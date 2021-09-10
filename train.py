import torch.optim
from Model.Model import Encoder, Decoder, Seq2Seq
from Components import enviroment_variables as env
from Preprocess.preprocess_dataset import mask_tokenize_python
from eval import evaluate, init_transformer
from Components.utils import load, save, make_trg_mask, initialize_weights
from torchtext.legacy.data import BucketIterator
from torchtext.legacy import data
import Model.CrossEntropyLoss as CEL
import Components.Constants as Const
from tqdm import tqdm
import time
import math


def count_parameters(model):
    """
    Count the number of parameters within the constructed transformer model.

    :param model: Fully constructed Transformer model (see Seq2Seq implementation).
    :return: Number of parameters within the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maskNLLLoss(inp, target, mask):
    """
    Implement Cross Entropy Loss.

    :param inp: input.
    :param target: target value.
    :param mask: mask respective to source or target (defined in Seq2Seq)
    :return loss: Loss value
    :return nTotal: Mask's sum.
    """

    nTotal = mask.sum()
    crossEntropy = CEL.CrossEntropyLoss(ignore_index=Const.TRG_PAD_IDX, smooth_eps=0.20)
    loss = crossEntropy(inp, target)
    loss = loss.to(env.DEVICE)
    return loss, nTotal.item()


def train(model, iterator, optimizer, criterion, clip):
    """
    Train the model where to iterate tqm is used to provide an interactive bar within the terminal when training.

    :param model: Constructed Transformer model (Seq2Seq).
    :param iterator: BucketIterator split into validation and train iterator.
    :param optimizer: optimizer of choice from torch.optim library.
    :param criterion: maskNLLLoss.
    :param clip: default = 1.
    :return: current epoch loss.
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
    Calculate elapsed minutes and seconds.

    :param start_time: Epoch start time.
    :param end_time: Epoch end time.
    :return elapsed_mins: elapsed minutes since start of epoch.
    :return elapsed_secs: elapsed seconds since start of epoch.
    """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():

    # load training and validation dataset
    train_df = load(env.TRAIN_DF_PATH, dataframe=True)
    val_df = load(env.VAL_DF_PATH, dataframe=True)

    model = init_transformer(eval=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=Const.LEARNING_RATE)

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

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    # test
    main()
