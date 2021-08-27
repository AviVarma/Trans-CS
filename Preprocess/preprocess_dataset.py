import os
import pickle
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Components.utils import save
from Components import enviroment_variables as env
import random
import keyword
import pandas as pd
import numpy as np
import torch
from tokenize import tokenize
import io
from torchtext.legacy import data


def load_dataset(filepath):
    """
    load the dataset from the provided filepath and return the dataset as a list of dictionaries.
    :param filepath: the file path for python to english dataset.
    :return data_points: list containing a dictionary for all questions and their subsequent solutions.
    """
    file = open(filepath, "r", encoding="utf-8")
    file_lines = file.readlines()

    data_points = []
    dp = None
    for line in file_lines:
        if line[0] == "#":
            if dp:
                dp["solution"] = ''.join(dp["solution"])
                data_points.append(dp)
            dp = {"question": line[1:], "solution": []}
        else:
            dp["solution"].append(line)

    return data_points


def tokenize_python(src):
    """
    Using python's default tokenize library, extract the token type and the token string.
    :param src: python source code in utf-8 from the data_points list
    :return tokenized_output: list for the source code tokenized format: [python token type, python token]
    """
    tokenized_output = []
    py_tokens = list(tokenize(io.BytesIO(src.encode('utf-8')).readline))

    for i in range(0, len(py_tokens)):
        tokenized_output.append((py_tokens[i].type, py_tokens[i].string))

    return tokenized_output


def mask_tokenize_python(src, mask_factor=0.3):
    """
    randomly pick variables and mask them with  'var_1, 'var_2' etc to make sure the model does not
    fixate on the way variables are named amd understands the program's logic. When randomly picking
    the variables, python's reserved keyword literals, control structures and object properties are
    ignored.
    :param src: python source code in utf-8 from the data_points list
    :param mask_factor: chance of a variable being masked default: 0.3
    :return tokenized_output: list for the source code tokenized format [python token type, python token]
    """
    var_dict = {}  # Dictionary that stores masked variables

    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip',
                 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    counter = 1
    py_tokens = list(tokenize(io.BytesIO(src.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(py_tokens)):

        if py_tokens[i].type == 1 and py_tokens[i].string not in skip_list:
            # avoid masking modules, functions and error literals
            if i > 0 and py_tokens[i - 1].string in ['def', '.', 'import', 'raise', 'except', 'class']:
                skip_list.append(py_tokens[i].string)
                tokenized_output.append((py_tokens[i].type, py_tokens[i].string))
            # if variable is already masked
            elif py_tokens[i].string in var_dict:
                tokenized_output.append((py_tokens[i].type, var_dict[py_tokens[i].string]))
            # randomly mask variables
            elif random.uniform(0, 1) > 1 - mask_factor:
                var_dict[py_tokens[i].string] = 'var_' + str(counter)
                counter += 1
                tokenized_output.append((py_tokens[i].type, var_dict[py_tokens[i].string]))

            else:
                skip_list.append(py_tokens[i].string)
                tokenized_output.append((py_tokens[i].type, py_tokens[i].string))

        else:
            tokenized_output.append((py_tokens[i].type, py_tokens[i].string))

    return tokenized_output


def build_train_val_dataset(data_points):
    """
    Initialize a dataframe (python_data_frame) with data points as it's input.
    With this crate a train dataset and validation dataset.
    :param data_points: list containing a dictionary for all questions and their subsequent solutions.
    :return train dataframe, validation dataframe: .
    """
    python_data_frame = pd.DataFrame(data_points)

    np.random.seed(0)
    # split the dataset into train: 0.85 val: 0.15
    mask = np.random.rand(len(python_data_frame)) < 0.85

    train_df = python_data_frame[mask]
    val_df = python_data_frame[~mask]

    return train_df, val_df


def expand_vocabulary(train_df, val_df, fields, expansion_factor=100):
    """
    Apply data augmentations expansion_factor times so majority of the augmentations have been
    captured in the vocabulary. This will generalize and expand the vocabulary beyond the
    initial size.
    :param train_df: train dataframe from build_train_val_dataset()
    :param val_df: validation dataframe from build_train_val_dataset()
    :param fields: [('Input', Input),('Output', Output)]
    :param expansion_factor: number of times to apply data augmentations, default=100.
    :return train_examples, val_examples: list containing all tokens as torchtext.legacy.data.example.Example objects.
    """
    train_examples = []
    val_examples = []

    for j in range(expansion_factor):
        for i in range(train_df.shape[0]):
            try:
                ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                train_examples.append(ex)
            except:
                pass

    for i in range(val_df.shape[0]):
        try:
            ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
            val_examples.append(ex)
        except:
            pass

    return train_examples, val_examples


def main():
    print("Pre-processing...")
    random.seed(env.SEED)
    torch.manual_seed(env.SEED)
    torch.cuda.manual_seed(env.SEED)
    torch.backends.cudnn.deterministic = True

    data_points = load_dataset(env.DATASET_PATH)
    train_df, val_df = build_train_val_dataset(data_points)

    save(env.TRAIN_DF_PATH, dataframe=train_df)
    print("english_python_train.json saved!")
    save(env.VAL_DF_PATH, dataframe=val_df)
    print("english_python_val.json saved!")

    # create the vocabulary using torchtext
    Input = data.Field(tokenize=env.TOKENIZER,
                       init_token=env.INIT_TOKEN,
                       eos_token=env.EOS_TOKEN)

    Output = data.Field(tokenize=mask_tokenize_python,
                        init_token=env.INIT_TOKEN,
                        eos_token=env.EOS_TOKEN,
                        lower=False)

    fields = [('Input', Input), ('Output', Output)]
    train_examples, val_examples = expand_vocabulary(train_df, val_df, fields)

    train_data = data.Dataset(train_examples, fields)
    val_data = data.Dataset(val_examples, fields)

    Input.build_vocab(train_data, min_freq=env.MIN_FREQ)
    Output.build_vocab(val_data, min_freq=env.MIN_FREQ)

    # save the vocabs generated which will later be used by the model.
    save(env.VOCAB_INPUT, vocab=Input)
    save(env.VOCAB_OUTPUT, vocab=Output)
    print("Done! \n Vocabularies saved in:", os.path.dirname(env.VOCAB_INPUT))


if __name__ == '__main__':
    main()
