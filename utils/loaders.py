import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .misc import get_cifar_models
from collections import OrderedDict

__all__ = [
    'load_optimizer',
    'load_learning_rate_schedule',
    'load_checkpoint',
    # cifar
    'load_transform',
    'load_dataset',
    'load_model',
    # detection
    'load_state_dict_path',
    'load_checkpoint_path',
    'load_ensemble_path',
]


def __process_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:7] == "module." else k
        new_state_dict[name] = v
    return new_state_dict


# General loaders compatible with cifar and imagenet
def load_optimizer(args, model):
    # Get optimiser name
    opt_name = args.optim.lower()

    # Print message to LOG
    print("==> Creating '{}' optimiser".format(opt_name))

    # Supports only SGD and RMSprop
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay,
            nesterov = "nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay,
            eps = 0.0316,
            alpha = 0.9,
        )
    else:
        msg = "Invalid optimizer {}. Only SGD and RMSprop are supported."
        raise RuntimeError(msg.format(args.opt))

    return optimizer


def load_learning_rate_schedule(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()

    # Print message to LOG
    print("==> Creating '{}' learning rate scheduler".format(args.lr_scheduler))

    # Supports only MultiStep and Step and Exponential schedules
    if args.lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = args.schedule, gamma = args.gamma)
    elif args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.schedule_step, gamma = args.gamma)
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma = args.gamma)
    elif args.lr_scheduler == "cycliclr":
        step_size_up = args.total_steps // 2
        step_size_down = args.total_steps - step_size_up
        main_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr = args.base_lr, max_lr = args.max_lr,
            step_size_up=step_size_up, step_size_down=step_size_down)
    else:
        raise RuntimeError(
            "Invalid lr scheduler '{}'. Only MultiStepLR, StepLR and ExponentialLR "
            "are supported.".format(args.lr_scheduler)
        )
    return main_lr_scheduler


# Use this when training models
def load_checkpoint(args, model, optimizer, reset = False):
    # Defaults
    best_acc = 0.0
    start_epoch = 0

    # Load checkpoint
    # args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)

    # Extract information
    if not reset:
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']

    # For dataparallell and loading issues
    try: model.load_state_dict(__process_state_dict(checkpoint['state_dict']))
    except RuntimeError: model.model.load_state_dict(__process_state_dict(checkpoint['state_dict']))

    # optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, best_acc, start_epoch


# Loaders only compatible with cifar
def load_transform(args):

    # Let the normalisation layer be different for daf
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if args.arch.startswith('daf'): normalize = transforms.Normalize((0.50, 0.50, 0.50), (0.50, 0.50, 0.50))

    # Default transformation
    transform_train = transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # And with data augmentation
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    return transform_train, transform_test


def load_dataset(args, transform_train, transform_test, return_sets = False):
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainloader = None
    if transform_train is not None:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testloader = None
    if transform_test is not None:
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    if return_sets: return trainloader, testloader, num_classes, (trainset, testset)
    return trainloader, testloader, num_classes


def load_model(args, models, num_classes):

    if 'densenet' in args.arch:
        model = models.__dict__[args.arch](args = args)

    else:
        raise ValueError("==> Model architecture can not be loaded.")

    return model


# These loaders are used for detection
def load_state_dict_path(path):
    # Load checkpoint
    assert os.path.isfile(path) or os.path.islink(path), 'Error: no checkpoint directory found!'

    # Get checkpoint dict
    checkpoint = torch.load(path)

    # Get attributes
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    return __process_state_dict(state_dict), {'best_acc': best_acc, 'start_epoch': start_epoch}


def load_checkpoint_path(args, num_classes, path, use_cuda):
    # Get model directory
    if 'cifar' in args.dataset:
        models = get_cifar_models()
        model = load_model(args, models, num_classes)

    if use_cuda: model = model.cuda()

    # Get state dict
    state_dict, info = load_state_dict_path(path)

    model.load_state_dict(state_dict)
    return model


def load_ensemble_path(args, num_classes, path, use_cuda):

    # Load every model in ensemble
    ensemble = []
    for file in os.listdir(path):

        # Create full path to file
        filepath = os.path.join(path, file)

        print("Loading model from:", filepath)
        ensemble.append(load_checkpoint_path(args, num_classes, filepath, use_cuda))

    return ensemble
