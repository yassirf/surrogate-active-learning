import os
import shutil
import numpy as np

import time
import torch
import torch.nn as nn

from utils import AverageMeter
from utils import (
    accuracy,
    get_maxprob_metric,
    get_entropy_metric,
    get_eceloss_metric,
)

__all__ = [
    'save_dataset',
    'save_checkpoint',
    'adjust_learning_rate',
    'train',
    'train_daf',
    'test',
    'test_daf'
]


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best: shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_dataset(dataset, checkpoint='checkpoint', filename='dataset.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(dataset.lab_idx, filepath)


def adjust_learning_rate(state, lr_scheduler):
    lr_scheduler.step()
    state['lr'] = lr_scheduler.get_last_lr()[0]


def update_args(args, state, lr_scheduler):
    # Self-Distribution Distillation
    args.temperature_scale_est = max(1.0, args.temperature_scale_est * args.temperature_scale_est_gamma)
    args.temperature_scale_num = max(1.0, args.temperature_scale_num * args.temperature_scale_num_gamma)
    if args.batch_lr_update: adjust_learning_rate(state, lr_scheduler)


def batch_repetition(inputs, labels, repetitions: int):
    """
    Repeating examples in the batch for training subnetworks
    """
    # Repeat examples in the batch
    shuffled_inputs = inputs.repeat(repetitions, 1, 1, 1)
    shuffled_labels = labels.repeat(repetitions)

    # Perform shuffling
    perm_mask = torch.randperm(shuffled_inputs.size(0))
    return shuffled_inputs[perm_mask], shuffled_labels[perm_mask]


def input_repetition(inputs, labels, repetitionp: float):
    """
    Sampling examples in the batch for training subnetworks.
    This is only compatible when batch repetitions are set to 2
    """
    for i in range(0, inputs.size(0), 2):

        # Given that the sample is within probability
        if torch.rand(1).item() < repetitionp:

            # Set inputs to be equal
            inputs[i] = inputs[i+1]
            labels[i] = labels[i+1]

    return inputs, labels


def train(args, state, trainloader, model, scheduler, optimizer, epoch, use_cuda):
    # Switch to training mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # Batch level data augmentation
        inputs, targets = batch_repetition(inputs, targets, repetitions = args.batch_repetitions)
        inputs, targets = input_repetition(inputs, targets, repetitionp = args.input_repetitions)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Compute output
        outputs, extra = model(inputs)
        loss, extra = model.get_loss(args, outputs, extra, targets)

        # Terminate early if model has convergence issues
        if np.isnan(loss.data.item()):
            return np.nan, np.nan

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Update arguments for schedules of parameters
        update_args(args, state, scheduler)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg


def train_daf(args, state, trainloader, model, scheduler, optimizer, epoch, use_cuda):
    # Switch to training mode
    model.train()

    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # Batch level data augmentation
        inputs, targets = batch_repetition(inputs, targets, repetitions = args.batch_repetitions)
        inputs, targets = input_repetition(inputs, targets, repetitionp = args.input_repetitions)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Compute output
        outputs, extra = model(inputs)
        loss, extra = model.get_loss(args, outputs, extra, targets)

        # Terminate early if model has convergence issues
        if np.isnan(loss.data.item()):
            return np.nan, np.nan

        # Record loss
        losses.update(loss.data.item(), inputs.size(0))

        # Update arguments for schedules of parameters
        update_args(args, state, scheduler)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, 0.0


@torch.no_grad()
def test(args, state, testloader, model, scheduler, epoch, use_cuda):

    # Switch to evaluate mode
    model.eval()

    # Evaluate performance on cross-entropy
    criterion = nn.CrossEntropyLoss(reduction = 'mean')

    losses = AverageMeter()
    entropy = AverageMeter()
    maxprob = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # save predictions
    saved_probs, saved_targets = [], []

    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Compute output and ensemble
        outputs, _ = model(inputs)
        outputs = model.ensemble(outputs) if outputs.dim() == 3 else outputs
        loss = criterion(outputs, targets)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # Measure other quantities
        log_probs = torch.log_softmax(outputs, axis=-1)
        maxprob.update(get_maxprob_metric(log_probs).item(), inputs.size(0))
        entropy.update(get_entropy_metric(log_probs).item(), inputs.size(0))

        # Map to probabilities and save for ece loss
        saved_probs.append(torch.exp(log_probs))
        saved_targets.append(targets)

    # Concatenate
    saved_probs = torch.cat(saved_probs, axis=0)
    saved_targets = torch.cat(saved_targets, axis=0)

    # Compute calibration loss
    eceloss = get_eceloss_metric(saved_probs, saved_targets, n_bins=15)

    # Printing message
    msg = "Loss: {loss:.4f} | " \
          "top1: {top1: .4f} | " \
          "top5: {top5: .4f} | " \
          "entropy: {ent: .4f} | " \
          "max prob: {maxp: .4f} | " \
          "ece loss: {ece: .4f} | " \
          "time {time: .2f}"

    msg = msg.format(
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg,
        ent=entropy.avg,
        maxp=maxprob.avg,
        ece=eceloss,
        time=time.time()-t0
    )
    print(msg)

    return losses.avg, top1.avg


@torch.no_grad()
def test_daf(args, state, testloader, model, scheduler, epoch, use_cuda):

    # Switch to evaluate mode
    model.eval()

    # Evaluate performance on cross-entropy
    criterion = nn.MSELoss(reduction = 'mean')

    losses = AverageMeter()

    # save predictions
    saved_probs, saved_targets = [], []

    t0 = time.time()
    for batch_idx, (inputs, _) in enumerate(testloader):

        if use_cuda: inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)

        # Compute output and ensemble
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)

        # Measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))

    # Printing message
    msg = "Loss: {loss:.4f} | " \
          "time {time: .2f}"

    msg = msg.format(
        loss=losses.avg,
        time=time.time()-t0
    )
    print(msg)

    return losses.avg, 0.0

