import pickle


def save_pkl(filename, save_object):
    with open(filename, 'wb') as writer:
        pickle.dump(save_object, writer)


def load_pkl(filename):
    with open(filename, 'rb') as loader:
        return pickle.load(loader)
