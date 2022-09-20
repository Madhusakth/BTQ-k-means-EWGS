import argparse
import logging
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import torch.nn as nn

from custom_models import *
from custom_modules import *
from utils import *

from mobilenet_models import *

from sklearn.cluster import KMeans
import numpy as np

best_acc1 = 0
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")
# data and model
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), 
                                 help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--arch', type=str, default='resnet20_quant', help='model architecture resnet20_quant, mobilenetv2_quant')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
parser.add_argument('--lr_m', type=float, default=1e-3, help='learning rate for model parameters')
parser.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--lr_q_end', type=float, default=0.0, help='final learning rate for quantizer parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--decay_schedule_q', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--lr_scheduler_q', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

# arguments for quantization
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scsaling factor using Hessian trace')
parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='../results/ResNet20_CIFAR10/W1A1/')
parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='../results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth', 
                                       help='path for pretrained full-preicion model')

parser.add_argument('--btq', default=False, type=str2bool, help='BTQ setting')

parser.add_argument('--training_flag', type=str2bool, default=False, help='B_Q_code')
parser.add_argument('--eval', type=str2bool, default=False, help='eval the resumed model')
parser.add_argument('--weighted', type=str2bool, default=False, help='weighted_k_means')


parser.add_argument('--bits', type=int, default=5, help='bits/weight')
parser.add_argument('--cv_block_size', type=int, default=6, help='3x3 kernel block size')
parser.add_argument('--pw_fc_block_size', type=int, default=4, help='1x1 fc kernel block size')

parser.add_argument('--random_sens', type=str2bool, default=False, help='random sensitivity')
parser.add_argument('--grad_sens', type=str2bool, default=False, help='gradient sensitivity')

parser.add_argument('--debug', type=str2bool, default=False, help='debug')
args = parser.parse_args()
arg_dict = vars(args)

def reshape_weights(weight):
    """
    C_out x C_in x k x k -> (C_in x k x k) x C_out.
    """
    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.view(C_out, C_in * k * k).t()
    else:
        return weight.t()


def reshape_back_weights(weight, k=3, conv=True):
    """
    (C_in x k x k) x C_out -> C_out x C_in x k x k.
    """

    if conv:
        C_in_, C_out = weight.size()
        C_in = C_in_ // (k * k)
        return weight.t().view(C_out, C_in, k, k)
    else:
        return weight.t()


def unroll_weights(M, n_blocks):
    """
    Unroll weights.
    """
    return torch.cat(M.chunk(n_blocks, dim=0), dim=1)

def roll_weights(M, n_blocks):
    return torch.cat(M.chunk(n_blocks, dim=1), dim=0)


def grad_sens(weight, grad,name=None):
    with torch.no_grad():

        weight_np = weight.cpu().detach().numpy()
        grad = grad.cpu().detach().numpy()
        size = weight.numel()
        d1, d2, d3, d4 = weight_np.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]
        weight_np = weight_np.reshape(-1,1)
        grad = grad.reshape(1,-1)
        indices = torch.argsort(grad,descending=True)
        weight_np[indices[0][:int(size*0.3)]] += np.random.normal(0,0.05)
        weight_np = weight_np.reshape(d1,d2,d3,d4)

        weight = torch.from_numpy(weight_np.T).float().to(cuda_device)

        return weight

def random_sens(weight, grad,name=None):
    with torch.no_grad():
        weight_np = weight.cpu().detach().numpy()
        size = weight.numel()
        d1, d2, d3, d4 = weight_np.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]
        weight_np = weight_np.reshape(-1,1)
        indices = random.sample(range(0,size),int(size*0.3))
        weight_np[indices[0][:int(size*0.3)]] += np.random.normal(0,0.05)
        weight_np = weight_np.reshape(d1,d2,d3,d4)

        weight = torch.from_numpy(weight_np.T).float().to(cuda_device)

        return weight


### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

### train/test datasets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset = dsets.CIFAR10(root='../data/CIFAR10/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR10(root='../data/CIFAR10/',
                            train=False, 
                            transform=transform_test)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset = dsets.CIFAR100(root='../data/CIFAR100/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR100(root='../data/CIFAR100/',
                            train=False, 
                            transform=transform_test)
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)

### initialize model
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)

num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path)
    #missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.pretrain_path))
    #print("missing keys", missing_keys)
    #print("unexpected_keys", unexpected_keys)
    #'''
    current_dict = model.state_dict()
    print("Pretrained full precision weights are initialized")
    #print(trained_model)
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)
    #'''

if args.eval:
    model.eval()
    #for name,param in model.named_parameters():
    #    print(name, len(np.unique(param.cpu().detach().numpy())),np.unique(param.cpu().detach().numpy()))

    with torch.no_grad():
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            if args.training_flag:
                pred = model([images,30])
            else:
                pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Test accuracy: {}%".format(test_acc))
        logging.info("Test accuracy: {}%".format(test_acc))
    exit(0)

# initialize quantizer params
init_quant_model(model, train_loader, device)

### initialize optimizer, scheduler, loss function
trainable_params = list(model.parameters())
model_params = []
quant_params = []
for m in model.modules():
    if isinstance(m, QConv):
        model_params.append(m.weight)
        if m.bias is not None:
            model_params.append(m.bias)
        if m.quan_weight:
            quant_params.append(m.lW)
            quant_params.append(m.uW)
        if m.quan_act:
            quant_params.append(m.lA)
            quant_params.append(m.uA)
        if m.quan_act or m.quan_weight:
            quant_params.append(m.output_scale)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        model_params.append(m.weight)
        if m.bias is not None:
            model_params.append(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.affine:
            model_params.append(m.weight)
            model_params.append(m.bias)
print("# total params:", sum(p.numel() for p in trainable_params))
print("# model params:", sum(p.numel() for p in model_params))
print("# quantizer params:", sum(p.numel() for p in quant_params))
logging.info("# total params: {}".format(sum(p.numel() for p in trainable_params)))
logging.info("# model params: {}".format(sum(p.numel() for p in model_params)))
logging.info("# quantizer params: {}".format(sum(p.numel() for p in quant_params)))
if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
    raise Exception('Mismatched number of trainable parmas')

# optimizer for model params
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)
# optimizer for quantizer params
if args.optimizer_q == 'SGD':
    optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
elif args.optimizer_q == 'Adam':
    optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)
    
# scheduler for model params
if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
# scheduler for quantizer params
if args.lr_scheduler_q == "step":
    if args.decay_schedule_q is not None:
        milestones_q = list(map(lambda x: int(x), args.decay_schedule_q.split('-')))
    else:
        milestones_q = [args.epochs+1]
    scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones_q, gamma=args.gamma)
elif args.lr_scheduler_q == "cosine":
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)

print(model)

def eval(model, test_loader):
    model.eval()
    #if ep == 0:
    #    for name,param in model.named_parameters():
    #        print(name, len(np.unique(param.cpu().detach().numpy())))
    correct_classified = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        if args.training_flag:
            pred = model([images,30])
        else:
            pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")

### train
#'''
total_iter = 0
best_acc = 0
for ep in range(args.epochs):
    model.train()
    ### update grad scales
    if ep % args.update_every == 0 and ep != 0 and not args.baseline and args.use_hessian:
        update_grad_scales(model, train_loader, criterion, device, args)
    ###
    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], ep)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_m.zero_grad()
        optimizer_q.zero_grad()
        if args.training_flag:
            pred = model([images,ep+1])
        else:
            pred = model(images)
        loss_t = criterion(pred, labels)
        
        loss = loss_t
        loss.backward()
        
        optimizer_m.step()
        optimizer_q.step()

        if args.per_iter_btq and i%100 == 0:
            state_dict = model.state_dict()
            total_bits = 0
            total_comp_bits = 0

            for name, param in model.named_parameters():
                if ('layer' in name or 'weight' in name) and len(param.shape) >2:
                    state_dict[name], layer_bits = k_means_pq(state_dict[name],param.grad,name)
                    total_bits += state_dict[name].numel()#*5
                    total_comp_bits += layer_bits
                elif 'bias' in name:
                    total_bits += state_dict[name].numel()
                    total_comp_bits += state_dict[name].numel()*8 ######32
                else:
                    total_bits += state_dict[name].numel()
                    total_comp_bits += state_dict[name].numel()*8
            model.load_state_dict(state_dict)
            print("Bit ratio for compressed layers:", total_comp_bits/total_bits) #total_bits/total_comp_bits)

        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1
        if args.debug:
            if total_iter >=1: ########################
                break
            print("******************** done training ******************")

        print("Original model accuracy:")
        eval(model, test_loader)

        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if ('layer' in name or 'weight' in name) and len(param.shape) >2:   #>=2
                state_dict = model.state_dict()
                print(name, param.shape)
                orig_weights = state_dict[name].copy()

                if args.random_sens:
                    state_dict[name] = random_sens(state_dict[name],param.grad,name)
                elif args.grad_sens:
                    state_dict[name] = grad_sens(state_dict[name],param.grad,name)

                model.load_state_dict(state_dict)
                eval(model, test_loader)
                #load original weights to model
                state_dict[name] = orig_weights
                model.load_state_dict(state_dict)
                print("Orig model accuracy re-check")
                eval(model, test_loader)
    exit(0)

    
    scheduler_m.step()
    scheduler_q.step()

    with torch.no_grad():
        #'''  #########################
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            if args.training_flag:
                pred = model([images,20+i])
            else:
                pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()

        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        #if ep == 0:
        #    for name,param in model.named_parameters():
        #        print(name, len(np.unique(param.cpu().detach().numpy())))
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            if args.training_flag:
                pred = model([images,30])
            else:
                pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
        logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
        writer.add_scalar('test/acc', test_acc, ep)

        torch.save({
            'epoch':ep,
            'model':model.state_dict(),
            'optimizer_m':optimizer_m.state_dict(),
            'scheduler_m':scheduler_m.state_dict(),
            'optimizer_q':optimizer_q.state_dict(),
            'scheduler_q':scheduler_q.state_dict(),
            'criterion':criterion.state_dict()
        }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'optimizer_q':optimizer_q.state_dict(),
                'scheduler_q':scheduler_q.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))
    layer_num = 0

    for m in model.modules():
        if isinstance(m, QConv):
            layer_num += 1
            if args.QWeightFlag:
                writer.add_scalar("z_{}th_module/lW".format(layer_num), m.lW.item(), ep)
                logging.info("lW: {}".format(m.lW))
                writer.add_scalar("z_{}th_module/uW".format(layer_num), m.uW.item(), ep)
                logging.info("uW: {}".format(m.uW))
                writer.add_scalar("z_{}th_module/bkwd_scaleW".format(layer_num), m.bkwd_scaling_factorW.item(), ep)
                logging.info("grad_scaleW: {}".format(m.bkwd_scaling_factorW.item()))
            if args.QActFlag:
                writer.add_scalar("z_{}th_module/lA".format(layer_num), m.lA.item(), ep)
                logging.info("lA: {}".format(m.lA))
                writer.add_scalar("z_{}th_module/uA".format(layer_num), m.uA.item(), ep)
                logging.info("uA: {}".format(m.uA))
                writer.add_scalar("z_{}th_module/bkwd_scaleA".format(layer_num), m.bkwd_scaling_factorA.item(), ep)
                logging.info("grad_scaleA: {}".format(m.bkwd_scaling_factorA.item()))
            if args.QActFlag or args.QWeightFlag:
                writer.add_scalar("z_{}th_module/output_scale".format(layer_num), m.output_scale.item(), ep)
                logging.info("output_scale: {}".format(m.output_scale))
            logging.info('\n')
#'''
### Test accuracy @ last checkpoint
trained_model = torch.load(os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))
model.load_state_dict(trained_model['model'])
print("The best checkpoint is loaded")
logging.info("The best checkpoint is loaded")
model.eval()
#for name,param in model.named_parameters():
#    print(name, len(np.unique(param.cpu().detach().numpy())),np.unique(param.cpu().detach().numpy()))

with torch.no_grad():
    correct_classified = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        if args.training_flag:
            pred = model([images,30])
        else:
            pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print("Test accuracy: {}%".format(test_acc))
    logging.info("Test accuracy: {}%".format(test_acc))


