import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    """
    Encoder module from 'Attention is all you need'.
    """

    def __init__(self,  n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1,
                 n_position=200, scale_emb=False):
        super().__init__()

