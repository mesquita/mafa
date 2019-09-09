import os
import pickle


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    dir_to_create = os.path.dirname(name)
    if (not os.path.exists(dir_to_create)):
        os.makedirs(dir_to_create)
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
