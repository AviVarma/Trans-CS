import os
import torch
import pickle
import pandas as pd
import torch.nn as nn
import json
import Components.enviroment_variables as env
from torchtext.legacy import data


def save(filename, dataframe: pd.DataFrame = None, vocab: data = None, model=None):
    """
    Save both modified datasets and models.

    :param filename: the file path eg: ./here/file.json
    :param dataframe: provide a dataframe to save.
    :param vocab: provide a vocabulary to save.
    :param model: provide a model to save.
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
    """
    Load datasets byt providing the file path and setting dataframe to true.

    :param filename: the file path eg: ./here/file.json
    :param dataframe: Boolean, if you want to load a dataframe.
    :param vocab: Boolean, if you want to load a vocabulary.
    :param model: Boolean, if you want to load a model.
    :return: Respective data from the path provided.
    """

    if dataframe:
        return pd.read_json(filename)
    if vocab:
        return pickle.load(open(filename, 'rb'))
    if model:
        return torch.load(filename)


def initialize_weights(w):
    """
    Xavier uniform weights initialization.

    :param w: weights input. [model.apply(initialize_weights)]
    """

    if hasattr(w, 'weight') and w.weight.dim() > 1:
        nn.init.xavier_uniform_(w.weight.data)


def make_trg_mask(trg, TRG_PAD_IDX):
    """
    Create a mask for the <pad> tokens like in source mask.
    Then create a subsequent mask which is a diagonal matrix where:
    1. the elements above the diagonal will be zero.
    2. the elements below the diagonal will be set to the value in the input tensor.
    The subsequent mask is now concatenated with the padding mask using "AND" operator to combine
    the two masks ensuring both the subsequent tokens and the padding tokens cannot be attended to.

    :param trg: Target sequence. [batch size, trg len]
    :param TRG_PAD_IDX: Target sequence tokenized and changed to integers for mask creation.
    (elements below the diagonal matrix will be set to the value in the input tensor).
    :return trg_mask: Target mask. [batch size, 1, trg len, trg len]
    """

    # trg_pad_mask = [batch size, 1, 1, trg len]
    trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)

    trg_len = trg.shape[1]

    # trg_sub_mask = [trg len, trg len]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=env.DEVICE)).bool()

    # trg_mask = [batch size, 1, trg len, trg len]
    trg_mask = trg_pad_mask & trg_sub_mask

    return trg_mask
