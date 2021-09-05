import torch

SEED = 1234
INIT_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
TOKENIZER = 'spacy'
MIN_FREQ = 0

#DATASET_PATH = 'Datasets/text_file_dataset.txt'
TRAIN_DF_PATH = "Datasets/conala-corpus/conala-train.json"
VAL_DF_PATH = "Datasets/conala-corpus/conala-test.json"

TRAIN_DF_MODIFIED_PATH = "Datasets/augmented-conala-corpus/conala-train.json"
VAL_DF_MODIFIED_PATH = "Datasets/augmented-conala-corpus/conala-test.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_INPUT = "Saved/Vocabulary/Input.pkl"
VOCAB_OUTPUT = "Saved/Vocabulary/Output.pkl"

MODEL_SAVE_PATH = "Saved/Model_Checkpoints/model.pt"
ATTENTION_PATH = "Saved/Evaluation_resources/attention_confusion_matrix.png"
