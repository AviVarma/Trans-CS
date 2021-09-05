import torch.nn as nn
import torch


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, hid_dim, pf_dim, dropout):
        """
        Initialize linear layers for feedforward.

        :param hid_dim: Output dimension for encoder/ decoder (d_model from original transformer).
        :param pf_dim: Position-wise Feedforward Layer dimension, usually larger than hid_dim.
        :param dropout: nn.dropout()
        """

        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        The input is transformed from hid_dim to pf_dim.
        The original Transformer used a hid_dim of 512 and a pf_dim of 2048. The ReLU activation function and dropout
        are applied before it is transformed back into a hid_dim representation.

        :param x: [batch size, seq len, hid dim]
        :return: [batch size, seq len, hid dim]
        """

        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout, device):
        """
        Initialize query, key, value parameters for scaled dot product attention. All three of these are words from the
        input sequence that are meant to operate with each other in a certain pattern. Hence all three parameters are
        initialized as linear layers. W_o (W^O in original paper) is the linear layer applied at the end of the
        attention layer. head_i = Attention(W_q_i, W_k_i, W_v_i).
        Split the hid_dim into n_heads.
        Initialize the dropout layer.
        Initialize the scaling function as a tensor module.

        :param hid_dim: Output dimension for encoder/ decoder (d_model from original transformer).
        :param n_heads: Number of heads in multi-head attention.
        :param dropout: nn.dropout()
        :param device: Run model on GPU or CPU.
        """

        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        """
        The Transformer model uses scaled dot product attention. where the query and key are combined by taking the
        dot product between them, apply softmax and scale by d_k before multiplying by value.
        d_k is represented within this implementation as head_dim.

        Scaled dot-product attention is similar to Luong dot product attention, but is scaled by d_k which is used to
        stop results of the dot product from growing too large and causing gradient to become too small.

        To get Q, K, V, first calculate QW^Q, KW^K, VW^V with the initialized linear layers.
        Split the hid_dim into n_heads using .view(). Permute through them and multiply together.
        Calculate the energy by Q * K and scaling it by the square root of head_dim (head dim // n_heads).
        mask the energy so that attention is not given to sequences that should not not be covered; apply softmax and
        dropout. Apply this attention to value heads (V), then combine n_heads together.
        Multiply result with W_O.

        :param query: Words from the input sequence. [batch size, query len, hid dim]
        :param key: Words from the input sequence. [batch size, key len, hid dim]
        :param value: Words from the input sequence. [batch size, value len, hid dim]
        :param mask: Mask the energy so that attention is not given to sequences that should not not be covered.
        :return x: W_O after matrix multiplying the output with softmax.
        :return attention: Final calculated energy.
        """

        # lengths of the keys and values are always the same, thus when matrix multiplying the output of the softmax,
        # attention, with V we will always have valid dimension sizes for matrix multiplication.
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention
