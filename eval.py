import pickle

import spacy
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tokenize import tokenize, untokenize
from train import make_trg_mask
from tqdm import tqdm

from Model.Models import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env


# from train import initialize_weights
#
# Input = pickle.load(open(env.VOCAB_INPUT, 'rb'))
# Output = pickle.load(open(env.VOCAB_OUTPUT, 'rb'))
#
# fields = [('Input', Input), ('Output', Output)]
#
# INPUT_DIM = len(Input.vocab)
# OUTPUT_DIM = len(Output.vocab)
#
# SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
# TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]


def evaluate(model, iterator, criterion):
    model.eval()

    n_totals = 0
    print_losses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            mask_loss, nTotal = criterion(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    return sum(print_losses) / n_totals


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50000):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(30, 50))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    plt.savefig("myimage.png")


def eng_to_python(src, Input, Output, model):
    src = src.split(" ")
    translation, attention = translate_sentence(src, Input, Output, model, env.DEVICE)

    print(f'predicted trg: \n')
    print(untokenize(translation[:-1]).decode('utf-8'))

# def main():
#
#     enc = Encoder(INPUT_DIM, env.HID_DIM, env.ENC_LAYERS, env.ENC_HEADS,
#                   env.ENC_PF_DIM, env.ENC_DROPOUT, env.DEVICE)
#
#     dec = Decoder(OUTPUT_DIM, env.HID_DIM, env.DEC_LAYERS, env.DEC_HEADS,
#                   env.DEC_PF_DIM, env.DEC_DROPOUT, env.DEVICE)
#
#     # SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
#     # TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]
#
#     model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, env.DEVICE).to(env.DEVICE)
#     model.apply(initialize_weights)
#
#     src = "write a function that adds two numbers"
#     src = src.split(" ")
#     translation, attention = translate_sentence(src, Input, Output, model, env.DEVICE)
#
#     print(f'predicted trg sequence: ')
#     print(translation)
#     print("code: \n", untokenize(translation[:-1]).decode('utf-8'))
#
#     display_attention(src, translation, attention)
#
#
# if __name__ == '__main__':
#     main()
