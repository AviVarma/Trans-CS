import os
import pickle
import sys
from Components.enviroment_variables import VOCAB_INPUT, VOCAB_OUTPUT

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

# Please refer to the original function for definition in class Seq2Seq.
SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]
