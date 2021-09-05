import torch.nn as nn
from Model.SubLayers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        """
        Initialize multi-head attention, pointwise feedforward layer and dropout.

        :param hid_dim: Output dimension for encoder (d_model from original transformer).
        :param n_heads: Number of heads in multi-head attention.
        :param pf_dim: Position-wise Feedforward Layer dimension, usually larger than hid_dim.
        :param dropout: nn.Dropout()
        :param device: Run model on GPU or CPU.
        """

        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Pass the src sentence and it's mask (src_mask) into the multi-head attention layer and calculate
        attention over itself.
        Then perform dropout on it, apply a residual connection and pass this through the
        "layer normalization" layer defined as self_attn_layer_norm.
        Now pass the result through a position-wise feedforward layer and apply dropout.
        On the result apply a residual connection then layer normalization to get the output for the
        next layer. However, the parameters are not shared with the next layer.

        :param src: Source sentence. [batch size, src len, hid dim]
        :param src_mask: Source sentence mask (generated in Seq2Seq Class). [batch size, 1, 1, src len]
        :return src: Source sentence. [batch size, src len, hid dim]
        """

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        """
        Initialize the two attention layers normalization layers required: self_attention and encoder_attention.
        Initialize the "layer normalization" layer.
        Initialize the two attention layers required: self_attention and encoder_attention.
        Initialize the position-wise feedforward layer.
        Initialize the dropout.

        :param hid_dim: Output dimension for decoder (d_model from original transformer).
        :param n_heads: Number of heads in multi-head attention.
        :param pf_dim: Position-wise Feedforward Layer dimension, usually larger than hid_dim.
        :param dropout: nn.Dropout().
        :param device: Run model on GPU or CPU.
        """

        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        First perform self attention on the target sentence by using the decoder representation so far as the query,
        key and value. The target sequence mask (trg_mask) is applied to the self attention to present the decoder from
        paying attention to tokens ahead of the current position since all tokens are processed in parallel.
        Then apply dropout on the outputted target sentence followed by residual connection and layer normalization.

        Now the encoded source sentence (enc_src) is sent into the decoder using encoder attention. Within this
        multi-head attention layer, the query is the decoder representations; keys and values are encoder
        representations. the source mask, src_mask is used to prevent the multi-head attention layer from attending to
        <pad> tokens within the source sentence. hen apply dropout on the outputted target sentence followed by residual
        connection and layer normalization. Pass the output target sequence through a position-wise feedforward layer
        and another sequence of dropout, residual connection and layer normalization.

        :param trg: Target sequence. [batch size, trg len, hid dim]
        :param enc_src: Encoder Output. [batch size, src len, hid dim]
        :param trg_mask: Target sentence mask (generated in Seq2Seq Class). [batch size, 1, trg len, trg len]
        :param src_mask: Source sentence mask (generated in Seq2Seq Class). [batch size, 1, 1, src len]
        :return trg: Target sequence after layer normalization. [batch size, trg len, hid dim]
        :return attention: Decoder attention. [batch size, n heads, trg len, src len]
        """

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention where parameter are: query, key, value
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm: trg = [batch size, trg len, hid dim]
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
