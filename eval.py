import spacy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenize import untokenize
import matplotlib.ticker as ticker
import Components.Constants as Const
from Components.utils import make_trg_mask
from Components import enviroment_variables as env


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50000):
    """
    Generate translation from trained model by outputting translated sentence with <sos> removed.
    First tokenize the source sentence if not tokenized already. Append <sos> and <eos> tokens.
    numericalize source sentence and convert it to a tensor, then add a batch dimension.
    Create a source sentence mask. Feed both source sentence and mask into the encoder.
    Create a list for the output sentence initialized with <sos> token.

    :param sentence: source sentence to translate.
    :param src_field: Input vocabulary.
    :param trg_field: Output vocabulary.
    :param model: Trained model.
    :param device: run model on GPU or CPU.
    :param max_len: maximum length of output. (Default 50000)
    :return output: output sentence (with the <sos> token removed).
    :return attention: attention from the last layer.
    """

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


def save_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    """

    :param sentence:
    :param translation:
    :param attention:
    :param n_heads:
    :param n_rows:
    :param n_cols:
    :return:
    """

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

    plt.savefig(env.ATTENTION_PATH)


def eng_to_python(src, Input, Output, model):
    src = src.split(" ")
    translation, attention = translate_sentence(src, Input, Output, model, env.DEVICE)

    print(f'predicted trg: \n')
    print(untokenize(translation[:-1]).decode('utf-8'))


def evaluate(model, iterator, criterion):
    """

    :param model:
    :param iterator:
    :param criterion:
    :return:
    """

    model.eval()

    n_totals = 0
    print_losses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg, Const.TRG_PAD_IDX)

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


def main():
    model = Const.model.load_state_dict(torch.load(env.MODEL_SAVE_PATH))
    src = "write a function that adds two numbers"
    src = src.split(" ")
    translation, attention = translate_sentence(src, Const.Input, Const.Output, model, env.DEVICE)

    print(f'predicted trg sequence: ')
    print(translation)
    print("code: \n", untokenize(translation[:-1]).decode('utf-8'))
    save_attention(src, translation, attention)


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

if __name__ == '__main__':
    main()
