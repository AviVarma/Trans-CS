import torch

SEED = 1234
INIT_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
TOKENIZER = 'spacy'
MIN_FREQ = 0
DATASET_PATH = '../Datasets/english_python_data.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
