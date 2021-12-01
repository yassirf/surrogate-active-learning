from __future__ import print_function, absolute_import
import torch

__all__ = [
    'accuracy',
    'get_eceloss_metric',
    'get_maxprob_metric',
    'get_entropy_metric',
]


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = (pred == target.view(1, -1).expand_as(pred))
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_eceloss_metric(probs, target, n_bins = 15):
    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute confidence and prediction
    confidences, predictions = torch.max(probs, 1)

    # Get accuracy
    accuracies = predictions.eq(target)

    # ECE loss
    ece = 0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get bin mask
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

        # Get proportion of predictions in bin
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:

            # Average accuracy in bin
            accuracy_in_bin = accuracies[in_bin].float().mean()

            # Average confidence in bin
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Calculate ece loss for bin
            ece += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item()

    return ece


def get_maxprob_metric(log_probs):
    """
    Computes the average max probability
    """

    max_lp = log_probs.max(axis = -1).values
    return torch.exp(max_lp).mean()


def get_entropy_metric(log_probs):
    """
    Computes the average entropy
    """
    entropy = -torch.exp(log_probs) * log_probs
    entropy = entropy.sum(axis = -1)
    return entropy.mean()