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

# vars required for training
#INPUT_DIM = len(Input.vocab)
#OUTPUT_DIM = len(Output.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
N_EPOCHS = 100
CLIP = 1
