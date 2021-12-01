import random
from torch.utils.data import Dataset

__all__ = [
    'ActiveLearningDataset'
]


class ActiveLearningDataset(Dataset):
    def __init__(self, set, idx):

        # Original dataset
        self.set = set

        # Current dataset
        self.idx = idx

        # Dataset containing all indices corresponding to labelled
        self.lab_idx = idx

        # Create unlabbeled dataset
        self.__create_unlabelled()

        # Decides if in training mode or not
        self.training = True

    def train(self):
        self.training = True
        self.idx = self.lab_idx

    def eval(self):
        self.training = False
        self.idx = self.unlab_idx

    def __create_unlabelled(self):
        # Get original dataset length
        self.unlab_idx = list(range(len(self.set)))
        for val in self.lab_idx: self.unlab_idx.remove(val)

    def random_selection(self, num):
        return random.sample(self.idx, num)

    def create(self, idx):
        return ActiveLearningDataset(set = self.set, idx = idx)

    def update(self, idx):
        # Update the labelled index set
        for i in idx: self.lab_idx.append(self.unlab_idx[i])
        return ActiveLearningDataset(set=self.set, idx=self.lab_idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.set[self.idx[i]]