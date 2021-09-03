import os
import pickle
import sys
from Components.enviroment_variables import VOCAB_INPUT, VOCAB_OUTPUT, DEVICE
from Model.Models import Seq2Seq, Encoder, Decoder

# Change the current directory to the parent directory so that Preprocess directory can be accessed.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Preprocess.preprocess_dataset import mask_tokenize_python

"""
Hyper-Parameters and Constants which require the Pre-processor to be first executed can be found here.
"""

# Load the saved vocabulary.
Input = pickle.load(open(VOCAB_INPUT, 'rb'))
Output = pickle.load(open(VOCAB_OUTPUT, 'rb'))

#
fields = [('Input', Input), ('Output', Output)]

# Calculate the total length for the saved vocabularies. Within the "Attention is all you need" transformer model
# this parameter is called n_src_vocab.
INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)
HID_DIM = 256
# The number of layers in the encoder does not have to be equal to the number
# of layers in the decoder.
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

N_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
CLIP = 1

# Please refer to the original function for definition in class Seq2Seq.
SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]
