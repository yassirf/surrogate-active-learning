import numpy as np

import torch
from torch.distributions import normal
from torch.distributions import dirichlet

from typing import List, Dict

__all__ = [
    'ensemblecategoricals',
    'ensembledirichlets',
    'diagonalgaussianoverlogits',
    'isotropicgaussianoverlogits',
    'diagonalgaussianoverlogdirichlets',
]


class BaseClass(object):
    def __init__(self):
        pass

    @staticmethod
    def store(key, value, store_in: Dict):
        if key not in store_in:
            store_in[key] = []
        store_in[key].append(value)

    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:
        """
        Computes uncertainties and samples possible outputs
        The return should be a dictionary containing a key 'outputs'
        """
        raise NotImplementedError()


class EnsembleCategoricals(BaseClass):
    def __init__(self):
        super(EnsembleCategoricals, self).__init__()

    @staticmethod
    def compute_log_confidence(log_probs):
        log_probs = torch.logsumexp(log_probs, dim = 1) - np.log(log_probs.size(1))
        return log_probs.max(axis = -1).values

    @staticmethod
    def compute_entropy(log_probs):
        entropy = - log_probs * torch.exp(log_probs)
        return entropy.sum(-1)

    def compute_expected_entropy(self, log_probs):
        entropies = self.compute_entropy(log_probs)
        return entropies.mean(-1)

    def compute_entropy_expected(self, log_probs):
        log_probs = torch.logsumexp(log_probs, dim = 1) - np.log(log_probs.size(1))
        return self.compute_entropy(log_probs)

    @torch.no_grad()
    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:

        # Assert temperature parameter exists
        assert hasattr(args, "ood_temperature")

        # If number of outputs is 1, only compute certain uncertainties
        n = len(outputs)

        # Combine all outputs into a single tensor
        outputs = torch.stack(outputs, dim = 1)

        # Temperature anneal
        outputs = outputs/args.ood_temperature

        # Normalise results
        outputs = torch.log_softmax(outputs, dim = -1)

        self.store('unc_conf', self.compute_log_confidence(outputs), store_in)
        self.store('unc_tu', self.compute_entropy_expected(outputs), store_in)

        if n > 1:
            self.store('unc_du', self.compute_expected_entropy(outputs), store_in)
            self.store('unc_ku', store_in['unc_tu'][-1] - store_in['unc_du'][-1], store_in)

        info = {'outputs': outputs}
        return info


class EnsembleDirichlets(EnsembleCategoricals):
    def __init__(self, samples = 100):
        super(EnsembleDirichlets, self).__init__()
        self.samples = samples
        self.eps = 1e-8

    def compute_expected_entropy(self, log_alphas):
        alphas = torch.exp(log_alphas)
        alpha0 = torch.sum(alphas, dim=-1)

        entropy = torch.digamma(alpha0 + 1)
        entropy -= torch.sum(alphas * torch.digamma(alphas + 1), dim=-1) / alpha0

        return entropy.mean(1)

    @torch.no_grad()
    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:
        # Combine all outputs into a single tensor
        log_alphas = torch.stack(outputs, dim=1)
        log_probs = torch.log_softmax(log_alphas, dim=-1)

        self.store('unc_conf', self.compute_log_confidence(log_probs), store_in)
        self.store('unc_tu', self.compute_entropy_expected(log_probs), store_in)
        self.store('unc_du', self.compute_expected_entropy(log_alphas), store_in)
        self.store('unc_ku', store_in['unc_tu'][-1] - store_in['unc_du'][-1], store_in)

        info = {'outputs': log_probs}
        return info


class DiagonalGaussianOverLogits(EnsembleCategoricals):
    def __init__(self):
        super(DiagonalGaussianOverLogits, self).__init__()

    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:

        # Number of samples from gaussian
        self.num = args.num_samples

        # Create gaussians
        self.gaussians = [normal.Normal(*output) for output in outputs]

        # Sample from gaussians
        outputs = [gaussian.sample() for gaussian in self.gaussians for _ in range(self.num)]

        return super(DiagonalGaussianOverLogits, self).__call__(
            args = args,
            outputs = outputs,
            store_in = store_in
        )


class IsotropicGaussianOverLogits(DiagonalGaussianOverLogits):
    def __init__(self):
        super(IsotropicGaussianOverLogits, self).__init__()

    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:
        # Number of samples from gaussian
        self.num = args.num_samples

        # Create gaussians
        self.gaussians = [normal.Normal(*output) for output in outputs]

        # Sample from gaussians
        samples = [gaussian.sample() for gaussian in self.gaussians for _ in range(self.num)]

        _ = super(IsotropicGaussianOverLogits, self).__call__(
            args=args,
            outputs=samples,
            store_in=store_in
        )

        info = {'outputs': torch.stack(outputs, dim = 1)}
        return info


class DiagonalGaussianOverLogDirichlets(EnsembleDirichlets):
    def __init__(self):
        super(DiagonalGaussianOverLogDirichlets, self).__init__()

    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:
        # Number of samples from gaussian
        self.num = args.num_samples

        # Create gaussians
        self.gaussians = [normal.Normal(*output) for output in outputs]

        # Sample from gaussians
        outputs = [gaussian.sample() for gaussian in self.gaussians for _ in range(self.num)]

        return super(DiagonalGaussianOverLogDirichlets, self).__call__(
            args=args,
            outputs=outputs,
            store_in=store_in
        )


def ensemblecategoricals():
    return EnsembleCategoricals()


def ensembledirichlets():
    return EnsembleDirichlets()


def diagonalgaussianoverlogits():
    return DiagonalGaussianOverLogits()


def isotropicgaussianoverlogits():
    return IsotropicGaussianOverLogits()


def diagonalgaussianoverlogdirichlets():
    return DiagonalGaussianOverLogDirichlets()
