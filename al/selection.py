import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import scipy as sp
import scipy.special

from al import ActiveLearningDataset

__all__ = [
    'load_selection'
]


class BaseSelector(object):
    def __init__(self, args, use_cuda, **kwargs):
        self.use_cuda = use_cuda
        self.init_fraction = args.initial_size
        self.acqn_fraction = args.acquisition_size

    def is_empty(self, args, model: nn.Module, active_set: ActiveLearningDataset):

        # Set into training mode
        active_set.train()

        # Check if the dataset has an empty labelled set
        return len(active_set) == 0

    @torch.no_grad()
    def generate(self, args, model: nn.Module, active_set: ActiveLearningDataset):

        # Ensure active learning set is in evaluation
        active_set.eval()

        # Define the dataset loader
        active_loader = data.DataLoader(active_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        # Store all predictions
        storage = {'outputs': []}

        # Iterate over unlabelled and generate predictions
        for batch_idx, (inputs, targets) in enumerate(active_loader):

            if self.use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # Compute outputs and store
            outputs, _ = model(inputs)

            # Store output predictions
            storage['outputs'].append(outputs.detach().clone().cpu())

        # Stack predictions
        storage['outputs'] = torch.cat(storage['outputs'], dim = 0)

        return storage

    def select(self, args, model: nn.Module, active_set: ActiveLearningDataset):
        raise NotImplementedError

    @staticmethod
    def random(args, model: nn.Module, active_set: ActiveLearningDataset):

        # Ensure active learning set is in evaluation
        active_set.eval()

        # Initial size for random acquisition
        init_size = int(len(active_set.set) * args.initial_size)

        # Generate random subselection of data
        indices = active_set.random_selection(init_size)

        # Update active learning set
        return active_set.create(indices)


class SelectionGreedyLeastConfidence(BaseSelector):
    def __init__(self, args, use_cuda, **kwargs):
        super(SelectionGreedyLeastConfidence, self).__init__(args, use_cuda, **kwargs)

    def compute_single(self, args, outputs):
        values, _ =  torch.log_softmax(outputs, dim = -1).max(-1)
        return -values

    def select(self, args, model: nn.Module, active_set: ActiveLearningDataset):

        # Selection size
        size = int(self.acqn_fraction * len(active_set.set))

        # Generate outputs
        storage = self.generate(args, model, active_set)

        # Generate metrics
        metrics = self.compute_single(args, storage['outputs'])

        # Map metrics to numpy
        metrics = metrics.numpy()

        # Choose the predictions with largest metric
        indices = np.argpartition(metrics, -size)[-size:]

        # Combine dataset
        updated = indices + active_set.lab_idx

        # Return combined dataset
        return active_set.create(updated)


class SelectionGreedyMaxEntropy(BaseSelector):
    def __init__(self, args, use_cuda, **kwargs):
        super(SelectionGreedyMaxEntropy, self).__init__(args, use_cuda, **kwargs)

    def compute_single(self, args, outputs):
        log_probs = torch.log_softmax(outputs, dim = -1)
        return -(log_probs * np.exp(log_probs)).sum(-1)

    def select(self, args, model: nn.Module, active_set: ActiveLearningDataset):
        # Selection size
        size = int(self.acqn_fraction * len(active_set.set))

        # Generate outputs
        storage = self.generate(args, model, active_set)

        # Generate metrics
        metrics = self.compute_single(args, storage['outputs'])

        # Map metrics to numpy
        metrics = metrics.numpy()

        # Choose the predictions with largest metric
        indices = np.argpartition(metrics, -size)[-size:]

        # Combine dataset
        updated = indices + active_set.lab_idx

        # Return combined dataset
        return active_set.create(updated)


class SelectionGreedyMaxRecon(BaseSelector):
    def __init__(self, args, use_cuda, **kwargs):
        super(SelectionGreedyMaxRecon, self).__init__(args, use_cuda, **kwargs)

    @torch.no_grad()
    def generate(self, args, model: nn.Module, active_set: ActiveLearningDataset):

        # Ensure active learning set is in evaluation
        active_set.eval()

        # Define the dataset loader
        active_loader = data.DataLoader(active_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        # Store all predictions
        storage = {'outputs': [], 'latents': [], 'inputs': []}

        # Iterate over unlabelled and generate predictions
        for batch_idx, (inputs, targets) in enumerate(active_loader):

            if self.use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # Compute outputs and store
            outputs, extra = model(inputs)

            # Store output predictions
            storage['outputs'].append(outputs.detach().clone().cpu())
            storage['latents'].append(extra['latent'].detach().clone().cpu())
            storage['inputs'].append(extra['input'].detach().clone().cpu())

        # Stack predictions
        storage['outputs'] = torch.cat(storage['outputs'], dim = 0)
        storage['latents'] = torch.cat(storage['latents'], dim = 0)
        storage['inputs'] = torch.cat(storage['inputs'], dim = 0)

        return storage

    def compute_single(self, args, outputs, inputs):
        diff = outputs - inputs
        diff = diff.view(diff.size(0), -1)
        return (diff * diff).mean(-1)

    def select(self, args, model: nn.Module, active_set: ActiveLearningDataset):
        # Selection size
        size = int(self.acqn_fraction * len(active_set.set))

        # Generate outputs
        storage = self.generate(args, model, active_set)

        # Generate metrics
        metrics = self.compute_single(args, storage['outputs'], storage['inputs'])

        # Map metrics to numpy
        metrics = metrics.numpy()

        # Choose the predictions with largest metric
        indices = np.argpartition(metrics, -size)[-size:]

        # Combine dataset
        updated = indices + active_set.lab_idx

        # Return combined dataset
        return active_set.create(updated)


def load_selection(args, use_cuda, **kwargs):

    if args.selector.endswith('lc'):
        return SelectionGreedyLeastConfidence(args, use_cuda, **kwargs)

    if args.selector.endswith('entropy'):
        return SelectionGreedyMaxEntropy(args, use_cuda, **kwargs)

    if args.selector.endswith('recon'):
        return SelectionGreedyMaxRecon(args, use_cuda, **kwargs)

    raise ValueError("Selection method '{}' has not been defined".format(args.selector))