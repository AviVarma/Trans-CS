import pickle

import torch

SEED = 1234
INIT_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
TOKENIZER = 'spacy'
MIN_FREQ = 0
DATASET_PATH = 'Datasets/english_python_data.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_INPUT = "Vocabulary/Input.pkl"
VOCAB_OUTPUT = "Vocabulary/Output.pkl"
TRAIN_DF_PATH = "Datasets/english_python_dataframe/english_python_train.json"
VAL_DF_PATH = "Datasets/english_python_dataframe/english_python_val.json"
MODEL_SAVE_PATH = "Checkpoints/model.pt"
