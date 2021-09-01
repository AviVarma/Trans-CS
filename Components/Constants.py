import os
import pickle
import sys
from Components.enviroment_variables import VOCAB_INPUT, VOCAB_OUTPUT

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Preprocess.preprocess_dataset import mask_tokenize_python

Input = pickle.load(open(VOCAB_INPUT, 'rb'))
Output = pickle.load(open(VOCAB_OUTPUT, 'rb'))

fields = [('Input', Input), ('Output', Output)]

INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)

SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]