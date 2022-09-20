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
from mobilenet_models import *

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
parser.add_argument('--arch', type=str, default='resnet20_fp', help='model architecture resnet20_fp, mobilenetv2_fp, resnet56_fp')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--lr_m', type=float, default=1e-1, help='learning rate for model parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='test/')

#resume
parser.add_argument('--resume', type=str, default='')

parser.add_argument('--random_sens', type=str2bool, default=False, help='random sensitivity')
parser.add_argument('--grad_sens', type=str2bool, default=False, help='gradient sensitivity')

parser.add_argument('--debug', type=str2bool, default=False, help='debugging mode')

args = parser.parse_args()
arg_dict = vars(args)

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


def grad_sens(weight, grad,name=None):
    with torch.no_grad():
        
        weight_np = weight.cpu().detach().numpy()
        #grad = grad.cpu().detach().numpy()
        size = weight.numel()
        d1, d2, d3, d4 = weight_np.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]
        weight_np = weight_np.reshape(-1,1)
        grad = abs(grad.reshape(-1,1))
        indices = torch.argsort(grad[:,0],descending=True).cpu().detach().numpy()
        weight_np[indices[:int(size*0.5)]] += np.random.normal(0,0.05)
        weight_np = weight_np.reshape(d1,d2,d3,d4)

        weight = torch.from_numpy(weight_np).float().to(device)

        return weight

def random_sens(weight, grad,name=None):
    with torch.no_grad():
        weight_np = weight.cpu().detach().numpy()
        size = weight.numel()
        d1, d2, d3, d4 = weight_np.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]
        weight_np = weight_np.reshape(-1,1)
        indices = random.sample(range(0,size),int(size*0.3))
        weight_np[indices[:int(size*0.5)]] += np.random.normal(0,0.05)
        weight_np = weight_np.reshape(d1,d2,d3,d4)

        weight = torch.from_numpy(weight_np).float().to(device)

        return weight


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

### initialize optimizer, scheduler, loss function
# optimizer for model params
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model.parameters(), lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model.parameters(), lr=args.lr_m, weight_decay=args.weight_decay)
    
# scheduler for model params
if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)

### train
total_iter = 0
best_acc = 0
start_epoch = 0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    print(args.resume)
    model.load_state_dict(checkpoint['model'])
    #best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print(start_epoch)

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
        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print(test_acc)
    #print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")


#print(model)
for ep in range(start_epoch,args.epochs):
    model.train()
    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_m.zero_grad()
            
        pred = model(images)
        loss_t = criterion(pred, labels)
        
        loss = loss_t
        loss.backward()
        
        optimizer_m.step()
        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1
        if args.debug==True:
            if total_iter %10 == 0:
                break
    
    scheduler_m.step()
    print("Original model accuracy:")
    eval(model,test_loader)

    param_size = []
    param_name = []
    for name,param in model.named_parameters():
        print(name, param.shape)
        #if ('layer' in name or 'weight' in name) and len(param.shape) >2:
        if len(param.shape) >2:
            param_size.append(param.numel())
            param_name.append(name)
    sorted_idx = sorted(range(len(param_size)), key=lambda k: param_size[k])
    sorted_idx = sorted_idx[::-1] #flip for descending order

    
    state_dict = model.state_dict()
    for idx in sorted_idx:
        curr_name = param_name[idx]
        for name, param in model.named_parameters():
                #if ('layer' in name or 'weight' in name) and len(param.shape) >2:   #>=2
                if name == curr_name:
                    state_dict = model.state_dict()
                    print(name, param.shape)
                    orig_weights = state_dict[name].clone().detach()

                    state_dict[name] = random_sens(orig_weights,param.grad,name)
                    model.load_state_dict(state_dict)
                    print("Random sensitivity result:")
                    eval(model, test_loader)


                    state_dict[name] = grad_sens(orig_weights,param.grad,name)
                    model.load_state_dict(state_dict)
                    print("Gradient sensitivity result:")
                    eval(model, test_loader)

                    #load original weights to model
                    state_dict[name] = orig_weights
                    model.load_state_dict(state_dict)
                    #print("Orig model accuracy re-check")
                    #eval(model, test_loader)
    exit(0)


    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
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
            'criterion':criterion.state_dict(),
            'best_acc':best_acc,
        }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'criterion':criterion.state_dict(),
                'best_acc':best_acc,
                
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))  
    

### Test accuracy @ last checkpoint
trained_model = torch.load(os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
model.load_state_dict(trained_model['model'])
print("The last checkpoint is loaded")
logging.info("The last checkpoint is loaded")
model.eval()
with torch.no_grad():
    correct_classified = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print("Test accuracy: {}%".format(test_acc))
    logging.info("Test accuracy: {}%".format(test_acc))
