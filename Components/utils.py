import os
import torch


def save(filename, dataframe=None, vocab=None, model=None):
    """
    function will save both modified datasets and models
    param filename: the file path eg: ./here/file.json
    param dataframe: default is none, provide a dataframe to save.
    param model: default is none, provide a model to save.
    """
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    elif dataframe is not None:
        dataframe.to_json(filename, orient='records')
    elif model is not None:
        torch.save(model.state_dict(), filename)
    elif vocab is not None:
        import pickle
        output = open(filename, 'wb')
        pickle.dump(vocab, output)
        output.close()
