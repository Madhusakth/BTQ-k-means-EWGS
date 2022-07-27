import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv']

class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x_in, num_levels, scaling_factor):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)

        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in-x_out)
        return x_out
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        if delta > 0:
            print("scaling")
        scale = 1 + delta * torch.sign(g)*diff
        return g * scale, None, None

class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None

class BQ_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x_in):
        return x_in
    @staticmethod
    def backward(ctx, g):
        return g, None

# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply

        self.BQ_discretizer = BQ_discretizer.apply

        self.training_flag = args.training_flag
    
        if self.quan_weight:
            self.weight_levels = args.weight_levels
            self.uW = nn.Parameter(data = torch.tensor(0).float())
            self.lW = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())

        if self.quan_act:
            self.act_levels = args.act_levels
            self.uA = nn.Parameter(data = torch.tensor(0).float())
            self.lA = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(data = torch.tensor(1).float())
        
        self.hook_Qvalues = False
        self.buff_weight = None
        self.buff_act = None

    def weight_quantization(self, weight):
        weight = (weight - self.lW) / (self.uW - self.lW)
        weight = weight.clamp(min=0, max=1) # [0, 1]

        if not self.baseline:
            weight = self.EWGS_discretizer(weight, self.weight_levels, self.bkwd_scaling_factorW)
        else:
            weight = self.STE_discretizer(weight, self.weight_levels)
            
        if self.hook_Qvalues:
            self.buff_weight = weight
            self.buff_weight.retain_grad()
        
        weight = (weight - 0.5) * 2 # [-1, 1]

        return weight

    def act_quantization(self, x):
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1) # [0, 1]
        
        if not self.baseline:
            x = self.EWGS_discretizer(x, self.act_levels, self.bkwd_scaling_factorA)
        else:
            x = self.STE_discretizer(x, self.act_levels)
            
        if self.hook_Qvalues:
            self.buff_act = x
            self.buff_act.retain_grad()

        return x

    def initialize(self, x):
        # self.init.data.fill_(0)
        Qweight = self.weight
        Qact = x
        
        if self.quan_weight:
            self.uW.data.fill_(self.weight.std()*3.0)
            self.lW.data.fill_(-self.weight.std()*3.0)
            Qweight = self.weight_quantization(self.weight)

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self.act_quantization(x)

        Qout = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        out = F.conv2d(x, self.weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        self.output_scale.data.fill_(out.abs().mean() / Qout.abs().mean())

    def B_Q_weight(self, weight, grad):
        bit = 5
        bins = 2**bit
        #d1,d2,d3,d4 = weight.shape
        #weight = weight.reshape(-1,1)
        weight_np = weight.cpu().detach().numpy()
        max_val = max(weight_np.reshape(-1,1))[0]
        min_val = min(weight_np.reshape(-1,1))[0]
        #if grad != None:
        #    grad = grad.cpu().detach().numpy()
        #    print(np.unique(grad))
        #    print(len(np.unique(grad)))

        #'''

        if grad != None and len(np.unique(grad.cpu().detach().numpy()))>1:
            #import pdb
            #pdb.set_trace()
            grad_np = grad.cpu().detach().numpy()
            mean = np.mean(weight_np.reshape(-1,1))
            std = np.std(weight_np.reshape(-1,1))
            RangeValues = [min_val, mean - std, mean+std, max_val]
            
            to_modify = 0
            #bin_grad = []
            while(len(RangeValues)<bins+1):
                bin_grad = []
                #if len(RangeValues) > 4:
                #    bin_grad.pop(to_modify)
                for times in range(len(RangeValues) - 1):
                    indices = np.where(np.logical_and(weight_np>=RangeValues[times], weight_np<=RangeValues[times+1]))
                    bin_grad.append(sum(abs(grad_np[indices])))

                to_modify = np.array(bin_grad).argmax()
                middle_bin = (RangeValues[to_modify] + RangeValues[to_modify+1]) / 2
                RangeValues.insert(to_modify+1, middle_bin)
            for x in range(len(RangeValues) - 1):
                v2 = weight.cpu().detach().numpy()
                indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
                weight[indices] = (RangeValues[x] + RangeValues[x+1])/2



        else:
            gap = (abs(max_val) + abs(min_val))/bins
            RangeValues = [min_val]
            for i in range(1,bins+1):
                RangeValues.append(min_val+gap*i)

            for x in range(len(RangeValues) - 1):
                v2 = weight.cpu().detach().numpy()
                indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
                weight[indices] = (RangeValues[x] + RangeValues[x+1])/2
        #'''
        #weight = weight.reshape(d1,d2,d3,d4)
        weight = self.BQ_discretizer(weight)
        return weight


    def forward(self, x):
        ep = x[1]
        x = x[0]
        if self.init == 1:
            self.initialize(x)
        Qweight = self.weight
        grad = Qweight.grad

        #if Qweight.grad != None:
        #    print(np.unique(Qweight.grad.cpu().detach().numpy()))
        
        if self.quan_weight:
            #if ep <=1:
            #    Qweight = self.weight_quantization(Qweight)
            if ep == 20:
                Qweight = self.B_Q_weight(Qweight,grad)

        Qact = x
        if self.quan_act:
            Qact = self.act_quantization(Qact)

        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups) * torch.abs(self.output_scale)

        return [output,ep]
