from Model.Models import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env
from Components.utils import load


def main():
    # load vocabularies
    Input = load(env.VOCAB_INPUT)
    Output = load(env.VOCAB_OUTPUT)
