import pickle

__all__ = [
    'save_object_pickle',
    'load_object_pickle',
]


def save_object_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
