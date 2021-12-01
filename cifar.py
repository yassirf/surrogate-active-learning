'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import os
import time
import numpy as np

import torch.utils.data as data
import torch.backends.cudnn as cudnn
import models.cifar as models

from utils import train as train_epoch
from utils import test as test_epoch
from utils import adjust_learning_rate, save_checkpoint

from utils import Logger, mkdir_p, savefig, get_args, get_device, set_seed
from utils import load_transform, load_dataset, load_model, load_checkpoint, save_dataset
from utils import load_optimizer, load_learning_rate_schedule
from utils import format_time

from al.selection import BaseSelector
from al import ActiveLearningDataset, load_selection


def setup_task(models):
    """
    Setting up task and necessary arguments
    """

    # Get all model names
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # Get arguments
    args, state = get_args(model_names=model_names)

    # Get device
    device, use_gpu = get_device(use_gpu=args.use_gpu)

    # Set seeds
    set_seed(args, use_gpu)

    # Validate dataset
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

    return args, state, device, use_gpu


def setup_model(args, state, device, use_gpu):

    # Tracking best accuracy of model
    best_acc = 0.0

    # Start from epoch 0 or last checkpoint epoch
    start_epoch = args.start_epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Training and Testing datasets
    print('==> Using device {}'.format(device))
    print('==> Preparing dataset %s' % args.dataset)

    # Dataset output size
    num_classes = 10 if args.dataset == 'cifar10' else 100

    # Load model
    print("==> Creating model '{}'".format(args.arch))
    model = load_model(args, models, num_classes).to(device)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Initialise optimiser
    cudnn.benchmark = True
    optimizer = load_optimizer(args, model)
    scheduler = load_learning_rate_schedule(args, optimizer)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!\n{}'.format(args.resume)
        model, _, best_acc, start_epoch = load_checkpoint(args, model, optimizer, reset=True)

        # Reinitialise after resuming
        optimizer = load_optimizer(args, model)
        scheduler = load_learning_rate_schedule(args, optimizer)

    state['lr'] = scheduler.get_last_lr()[0]
    # Loggers for training
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Log location for saved models
    print('==> Checkpoints saved to: {}'.format(args.checkpoint))

    return start_epoch, best_acc, model, optimizer, scheduler, logger


# Setting up task
args, state, device, use_gpu = setup_task(models)


def main():

    # Setting up model structure
    start_epoch, best_acc, model, optimizer, scheduler, logger = \
        setup_model(args, state, device, use_gpu)

    # Time tracking
    t0 = time.time()

    # Load transforms
    transform_train, transform_test = load_transform(args)

    # Load test set
    _, testloader, num_classes, (trainset, _) = load_dataset(
        args=args,
        transform_train=transform_train,
        transform_test=transform_test,
        return_sets=True
    )

    # Initialise the active learning dataset
    active_dataset = ActiveLearningDataset(set = trainset, idx = [])

    # Define the selection mechanism
    selector: BaseSelector = load_selection(args, use_gpu)

    # Randomly acquire datapoints
    active_dataset = selector.random(args, model, active_dataset)

    for aliter in range(args.acquisition_iterations + 1):

        # Reinitialise model
        model = model.reinitialise().to(device)

        # Convert labelled into a training set
        trainloader = data.DataLoader(active_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        # Train and validation epoch loop
        for epoch in range(start_epoch, args.epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            train_loss, train_acc = train_epoch(args, state, trainloader, model, scheduler, optimizer, epoch, use_gpu)
            test_loss, test_acc = test_epoch(args, state, testloader, model, scheduler, epoch, use_gpu)

            # Update learning rate epoch wise
            if not args.batch_lr_update: adjust_learning_rate(state, scheduler)

            # Early termination if nan and it happens early in the training process
            if (np.isnan(train_loss) or np.isnan(test_loss)) and (epoch < args.epochs//2):
                print("\n\n\n----- Early stopping of training -----")
                return

            # Append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            # Model performance
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            # Model saving
            filename = "checkpoint.al{}.pth.tar".format(aliter + 1)
            if (args.save_every > 0) and ((epoch + 1) % args.save_every == 0):
                filename = "checkpoint_{}.al{}.pth.tar".format(epoch + 1, aliter + 1)

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint, filename=filename)

            t1 = time.time()
            print('Current: {:.4f}  Best: {:.4f};    Elapsed: {:}.'.format(test_acc, best_acc, format_time(t1-t0)))

        logger.close()
        logger.plot()
        savefig(os.path.join(args.checkpoint, 'log.al{}.eps'.format(aliter + 1)))

        print('Best acc:')
        print(best_acc)

        # Store the chosen dataset
        save_dataset(
            active_dataset,
            checkpoint=args.checkpoint,
            filename='dataset.al{}.pth.tar'.format(aliter + 1)
        )

        # Select data using model
        active_dataset = selector.select(args, model, active_dataset)


if __name__ == '__main__':
    main()
