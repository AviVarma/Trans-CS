import pickle

import torch.optim
from Model.Models import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env
from Preprocess.preprocess_dataset import mask_tokenize_python
from Components.utils import load, save
from torchtext.legacy.data import Field, BucketIterator, Iterator
from torchtext.legacy import data
import torch.nn as nn
import Model.CrossEntropyLoss as CEL
from tqdm import tqdm
import time
import math

# Global variables.
Input = pickle.load(open(env.VOCAB_INPUT, 'rb'))
Output = pickle.load(open(env.VOCAB_OUTPUT, 'rb'))

fields = [('Input', Input), ('Output', Output)]

#print(Input)

INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)

SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(w):
    if hasattr(w, 'weight') and w.weight.dim() > 1:
        nn.init.xavier_uniform_(w.weight.data)


def make_trg_mask(trg):
    # trg = [batch size, trg len]

    trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)

    # trg_pad_mask = [batch size, 1, 1, trg len]

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=env.DEVICE)).bool()

    # trg_sub_mask = [trg len, trg len]

    trg_mask = trg_pad_mask & trg_sub_mask

    # trg_mask = [batch size, 1, trg len, trg len]

    return trg_mask

def maskNLLLoss(inp, target, mask):
    # print(inp.shape, target.shape, mask.sum())
    nTotal = mask.sum()
    crossEntropy = CEL.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, smooth_eps=0.20)
    loss = crossEntropy(inp, target)
    loss = loss.to(env.DEVICE)
    return loss, nTotal.item()

def train(model, iterator, optimizer, criterion, clip):
    model.train()

    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg)
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    # load training and validation dataset
    train_df = load(env.TRAIN_DF_PATH, dataframe=True)
    val_df = load(env.VAL_DF_PATH, dataframe=True)

    # load vocabularies
    # Input = load(env.VOCAB_INPUT)  # try adding these to environment vars.
    # Output = load(env.VOCAB_OUTPUT)
    #
    # fields = [('Input', Input), ('Output', Output)]
    #
    # INPUT_DIM = len(Input.vocab)
    # OUTPUT_DIM = len(Output.vocab)

    enc = Encoder(INPUT_DIM, env.HID_DIM, env.ENC_LAYERS, env.ENC_HEADS,
                  env.ENC_PF_DIM, env.ENC_DROPOUT, env.DEVICE)

    dec = Decoder(OUTPUT_DIM, env.HID_DIM, env.DEC_LAYERS, env.DEC_HEADS,
                  env.DEC_PF_DIM, env.DEC_DROPOUT, env.DEVICE)

    # SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
    # TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, env.DEVICE).to(env.DEVICE)
    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=env.LEARNING_RATE)

    criterion = maskNLLLoss

    best_valid_loss = float('inf')

    for epoch in range(env.N_EPOCHS):

        start_time = time.time()

        train_example = []
        val_example = []

        for i in range(train_df.shape[0]):
            try:
                ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                train_example.append(ex)
            except:
                pass

        for i in range(val_df.shape[0]):
            try:
                ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
                val_example.append(ex)
            except:
                pass

        train_data = data.Dataset(train_example, fields)
        valid_data = data.Dataset(val_example, fields)

        train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size=env.BATCH_SIZE,
                                                               sort_key=lambda x: len(x.Input),
                                                               sort_within_batch=True, device=env.DEVICE)

        train_loss = train(model, train_iterator, optimizer, criterion, env.CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save(env.MODEL_SAVE_PATH, model=model)
            # torch.save(model.state_dict(), '/content/drive/MyDrive/TheSchoolOfAI/EndCapstone/model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()
