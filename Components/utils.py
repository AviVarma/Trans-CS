import os
import torch
import pickle
import pandas as pd
import json
from Components import enviroment_variables as env
from torchtext.legacy import data


def save(filename, dataframe: pd.DataFrame = None, vocab: data = None, model=None):
    """
    function will save both modified datasets and models
    param filename: the file path eg: ./here/file.json
    param dataframe: default is none, provide a dataframe to save.
    param model: default is none, provide a model to save.
    """
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if dataframe is not None:
        dataframe.to_json(filename, orient='records')
    if model is not None:
        torch.save(model.state_dict(), filename)
    if vocab is not None:
        import pickle
        output = open(filename, 'wb')
        pickle.dump(vocab, output)
        output.close()


def load(filename, dataframe=False, vocab=False, model=False):
    if dataframe:
        return pd.read_json(filename)
    if vocab:
        return pickle.load(open(filename, 'rb'))
