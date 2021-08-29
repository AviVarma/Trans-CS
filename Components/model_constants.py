import pickle
from Components.enviroment_variables import VOCAB_INPUT, VOCAB_OUTPUT
from Preprocess.preprocess_dataset import mask_tokenize_python

Input = pickle.load(open(VOCAB_INPUT, 'rb'))
Output = pickle.load(open(VOCAB_OUTPUT, 'rb'))

fields = [('Input', Input), ('Output', Output)]

INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)

SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]

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