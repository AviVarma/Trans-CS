from tokenize import untokenize

import torch.optim
from Model.Models import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env
from Preprocess.preprocess_dataset import mask_tokenize_python
from eval import evaluate, translate_sentence, save_attention
from Components.utils import load, save, make_trg_mask, initialize_weights
from torchtext.legacy.data import BucketIterator
from torchtext.legacy import data
import torch.nn as nn
import Model.CrossEntropyLoss as CEL
import Components.Constants as Const
from tqdm import tqdm
import time
import math


def count_parameters(model):
    """

    :param model:
    :return:
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def initialize_weights(w):
#     """
#
#     :param w:
#     :return:
#     """
#
#     if hasattr(w, 'weight') and w.weight.dim() > 1:
#         nn.init.xavier_uniform_(w.weight.data)


def maskNLLLoss(inp, target, mask):
    """

    :param inp:
    :param target:
    :param mask:
    :return:
    """

    # print(inp.shape, target.shape, mask.sum())
    nTotal = mask.sum()
    crossEntropy = CEL.CrossEntropyLoss(ignore_index=Const.TRG_PAD_IDX, smooth_eps=0.20)
    loss = crossEntropy(inp, target)
    loss = loss.to(env.DEVICE)
    return loss, nTotal.item()


def train(model, iterator, optimizer, criterion, clip):
    """

    :param model:
    :param iterator:
    :param optimizer:
    :param criterion:
    :param clip:
    :return:
    """

    model.train()

    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg, Const.TRG_PAD_IDX)
        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        mask_loss, nTotal = criterion(output, trg, trg_mask)

        mask_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    return sum(print_losses) / n_totals


def epoch_time(start_time, end_time):
    """

    :param start_time:
    :param end_time:
    :return:
    """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    # load training and validation dataset
    train_df = load(env.TRAIN_DF_PATH, dataframe=True)
    val_df = load(env.VAL_DF_PATH, dataframe=True)

    enc = Encoder(Const.INPUT_DIM, Const.HID_DIM, Const.ENC_LAYERS, Const.ENC_HEADS,
                  Const.ENC_PF_DIM, Const.ENC_DROPOUT, env.DEVICE)

    dec = Decoder(Const.OUTPUT_DIM, Const.HID_DIM, Const.DEC_LAYERS, Const.DEC_HEADS,
                  Const.DEC_PF_DIM, Const.DEC_DROPOUT, env.DEVICE)

    model = Seq2Seq(enc, dec, Const.SRC_PAD_IDX, Const.TRG_PAD_IDX, env.DEVICE).to(env.DEVICE)

    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=Const.LEARNING_RATE)

    #criterion = maskNLLLoss

    best_valid_loss = float('inf')

    for epoch in range(Const.N_EPOCHS):

        start_time = time.time()

        train_example = []
        val_example = []

        for i in range(train_df.shape[0]):
            try:
                ex = data.Example.fromlist([train_df.intent[i], train_df.snippet[i]], Const.fields)
                train_example.append(ex)
            except:
                pass

        for i in range(val_df.shape[0]):
            try:
                ex = data.Example.fromlist([val_df.intent[i], val_df.snippet[i]], Const.fields)
                val_example.append(ex)
            except:
                pass

        train_data = data.Dataset(train_example, Const.fields)
        valid_data = data.Dataset(val_example, Const.fields)

        train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size=Const.BATCH_SIZE,
                                                               sort_key=lambda x: len(x.Input),
                                                               sort_within_batch=True, device=env.DEVICE)

        train_loss = train(model, train_iterator, optimizer, maskNLLLoss, Const.CLIP)
        valid_loss = evaluate(model, valid_iterator, maskNLLLoss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save(env.MODEL_SAVE_PATH, model=model)
            #torch.save(Const.model.state_dict(), env.MODEL_SAVE_PATH)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # Evaluate the model.

    src = "write a function that adds two numbers"
    src = src.split(" ")
    translation, attention = translate_sentence(src, Const.Input, Const.Output, model, env.DEVICE)

    print(f'predicted trg sequence: ')
    print(translation)
    print("code: \n", untokenize(translation[:-1]).decode('utf-8'))
    save_attention(src, translation, attention)


if __name__ == '__main__':
    main()
