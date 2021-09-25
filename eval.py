import tokenize
from tokenize import untokenize
import pandas as pd
import spacy
import torch.optim
from matplotlib import pyplot as plt

from Preprocess.preprocess_dataset import tokenize_python
from Components.utils import is_dir, load
from Model.Model import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env
from Preprocess.preprocess_dataset import mask_tokenize_python
from Components.utils import make_trg_mask, initialize_weights
import Components.Constants as Const
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50000):
    """
    Generate translation from trained model by outputting translated sentence with <sos> removed.
    First tokenize the source sentence if not tokenized already. Append <sos> and <eos> tokens.
    numericalize source sentence and convert it to a tensor, then add a batch dimension.
    Create a source sentence mask. Feed both source sentence and mask into the encoder.
    Create a list for the output sentence initialized with <sos> token.

    :param sentence: Source sentence to translate.
    :param src_field: Input vocabulary.
    :param trg_field: Output vocabulary.
    :param model: Trained model.
    :param device: Run model on GPU or CPU.
    :param max_len: maximum length of output. (Default 50000)
    :return output: output sentence (with the <sos> token removed).
    :return attention: Attention from the last layer.
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
    Save the attention's confusion matrix over the source sentence for each step of the decoding.

    :param sentence: Source sentence tokenized.
    :param translation: Saved Input vocabulary
    :param attention: Normalized attention from model.
    :param n_heads: Number of heads in model. Default 8
    :param n_rows: Number of rows in graph.
    :param n_cols: Number of columns in graph.
    """

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(35, 55))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='inferno')

        ax.tick_params(labelsize=25)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    is_dir(env.ATTENTION_PATH)
    plt.savefig(env.ATTENTION_PATH)
    print("Attention confusion matrix saved at: " + env.ATTENTION_PATH)


def eng_to_python(src, model: Seq2Seq):
    """
    Execute translate_sentence function and return a formatted output.

    :param src: input query.
    :param model: pre-trained model.
    :return source code: source code in utf-8 format.
    """

    src = src.split(" ")
    translation, attention = translate_sentence(src, Const.Input, Const.Output, model, env.DEVICE)
    return untokenize(translation[:-1]).decode('utf-8')


def evaluate(model, iterator, NKLLLoss):
    """
    The evaluation loop, similar to the training loop, but without parameter updates and gradient
    calculation. calculate the Loss for the current model.

    :param model: Seq2Seq fully constructed transformer model.
    :param iterator: BucketIterator split into batches.
    :param NKLLLoss: NKLLLoss function.
    :return: Loss for current epoch.
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

            mask_loss, n_total = NKLLLoss(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    return sum(print_losses) / n_totals


def predict_queries(model: Seq2Seq):
    """
    For each query within validation dataset, predict and save code snippet to a text file.

    :param model: Trained transformer model
    """

    val_df = pd.read_json(env.VAL_DF_MODIFIED_PATH)
    print("writing predictions down...")
    with open(env.SAVED_PREDICTIONS, 'w') as f:
        for i in tqdm(val_df["intent"]):
            try:
                f.write("intent:\n")
                f.write(i + "\n" + "predicted code:\n")
                code = eng_to_python(i, model)
                f.write(code + "\n\n")
            except:
                pass
    print("Code predictions saved at: " + env.SAVED_PREDICTIONS)


def calculate_bleu(model: Seq2Seq):
    """
    For each query within validation dataset, predict sentence BLEU-4 score and return overall average.

    :param model: Trained transformer model
    """

    val_df = pd.read_json(env.VAL_DF_MODIFIED_PATH)

    references = []
    hyps = []
    print("Calculating corpus BLEU...")
    for i in tqdm(range(val_df.shape[0])):
        try:
            hypothesis = eng_to_python(val_df.intent[i], model)
            tokenize_hypothesis = tokenize_python(hypothesis)

            tokenized_ref = tokenize_python(val_df.snippet[i])
            references.append([tokenized_ref])
            hyps.append(tokenize_hypothesis)
        except:
            pass

    return corpus_bleu(references, hyps)


def calculate_sentence_bleu(query, ref, model: Seq2Seq):
    """
    Calculate sentence BLEU score for a specific query manually provided.

    :param query: Input question.
    :param ref: Dataframe containing intents relevant to input question.
    :param model: Trained transformer model
    """

    val_df = pd.read_json(env.VAL_DF_MODIFIED_PATH)
    df = val_df[val_df['intent'].str.contains(ref)]

    hypothesis = eng_to_python(query, model)
    tokenize_hypothesis = tokenize_python(hypothesis)

    references = []
    for i in df['snippet']:
        references.append(tokenize_python(i))

    return sentence_bleu(references, tokenize_hypothesis)


def evaluate_conala_sentence_bleu(model: Seq2Seq):
    r1 = calculate_sentence_bleu("split a multi-line string `inputString` into separate strings", "split", model)
    r2 = calculate_sentence_bleu("get rid of None values in dictionary?", "dictionary", model)
    r3 = calculate_sentence_bleu("download a file over HTTP", "HTTP", model)
    r4 = calculate_sentence_bleu("Merging two pandas dataframes", "pandas dataframe", model)
    r5 = calculate_sentence_bleu("Python how to combine two matrices in numpy", "numpy", model)

    # print(r1, r2, r3, r4, r5)
    return (r1 + r2 + r3 + r4 + r5) / 5


def init_transformer(is_eval: bool = True):
    """
    Initialize the Transformer model by constructing the encoder and decoder.
    Parse both the encoder and decoder into the Seq2Seq, initialize the weights, and load in the best trained
    parameters.

    :param is_eval: If function is being used for training or evaluating.
    :return model: model with pre-trained parameters.
    """
    enc = Encoder(Const.INPUT_DIM, Const.HID_DIM, Const.ENC_LAYERS, Const.ENC_HEADS,
                  Const.ENC_PF_DIM, Const.ENC_DROPOUT, env.DEVICE)

    dec = Decoder(Const.OUTPUT_DIM, Const.HID_DIM, Const.DEC_LAYERS, Const.DEC_HEADS,
                  Const.DEC_PF_DIM, Const.DEC_DROPOUT, env.DEVICE)

    model = Seq2Seq(enc, dec, Const.SRC_PAD_IDX, Const.TRG_PAD_IDX, env.DEVICE).to(env.DEVICE)

    model.apply(initialize_weights)

    if is_eval:
        model.load_state_dict(torch.load(env.MODEL_SAVE_PATH))

    return model


def main():
    model = init_transformer()

    val_df = pd.read_json(env.VAL_DF_MODIFIED_PATH)
    src = val_df.intent[0]
    src = src.split(" ")
    translation, attention = translate_sentence(src, Const.Input, Const.Output, model, env.DEVICE)
    save_attention(src, translation, attention)
    c_bleu = calculate_bleu(model)
    print("Corpus BLEU: " , c_bleu)
    predict_queries(model)

    # save the outputs:
    with open(env.SAVED_PERFORMANCE, 'w') as f:
        f.write("Corpus BLEU: ")
        f.write(str(c_bleu))
        f.write("\n")
    print("Code predictions saved at: " + env.SAVED_PERFORMANCE)


if __name__ == '__main__':
    main()
