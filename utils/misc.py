'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os, errno
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'get_device',
    'get_cifar_models',
    'set_seed',
    'get_mean_and_std',
    'init_params',
    'mkdir_p',
    'format_time',
    'AverageMeter'
]


def get_device(use_gpu):
    available_gpu = use_gpu and torch.cuda.is_available()
    return torch.device('cuda') if available_gpu else torch.device('cpu'), available_gpu


def get_cifar_models():
    import models.cifar as cifar_models
    return cifar_models


def set_seed(args, use_gpu):
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    print("===> Using random seed {}".format(args.manual_seed))

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.manual_seed)


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    """

    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """
    Init layer parameters.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    """
    Make dir if not exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count