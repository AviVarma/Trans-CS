import random
import pandas as pd
import json
import os
import torch
import numpy as np
import keyword
import argparse
from tokenize import tokenize, untokenize
import io


def load_dataset(filepath):
    """
    load the dataset from the provided filepath and return the dataset as a list of dictionaries.
    :param filepath: the file path for python to english dataset.
    :return: array containing a dictionary for all questions and their subsequent solutions.
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
    :return: list for the source code tokenized format: [python token type, python token]
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
    :return: list for the source code tokenized format: [python token type, python token]
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
    python_data_frame = pd.DataFrame(data_points)

    np.random.seed(0)
    # split the dataset into train: 0.85 val: 0.15
    mask = np.random.rand(len(python_data_frame)) < 0.85

    train_df = python_data_frame[mask]
    val_df = python_data_frame[~mask]

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-raw_dir', required=True)
    parser.add_argument('-dataset_path', required=True)


if __name__ == '__main__':
    main()
