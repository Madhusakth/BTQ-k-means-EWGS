# PyTorch implementation of Bin-Train and Quant using k-means and EWGS. 



EWGS code reference: https://github.com/cvlab-yonsei/EWGS

## Requirements
* Python >= 3.6
* PyTorch >= 1.3.0

## Datasets
* CIFAR-10 (will be automatically downloaded when you run the code)
* ImageNet (ILSVRC-2012) available at [http://www.image-net.org](http://www.image-net.org/download)


## References
* ImageNet training code: [[PyTorch official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py)]
* ResNet-18/34 models: [[PyTorch official code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)]
* ResNet-20 model: [[ResNet on CIFAR10](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)] [[IRNet](https://github.com/XHPlus/IR-Net/blob/master/resnet-20-cifar10/1w1a/resnet.py)]
* Quantized modules: [[DSQ](https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18)]
* Estimating Hessian trace: [[PyHessian](https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py#L160)]
