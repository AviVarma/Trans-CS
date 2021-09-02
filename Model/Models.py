import torch
import torch.nn as nn
from Model.Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    """
    Encoder module from 'BERT (using positional embeddings)'
    """
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=1000):
        """
        Initialize the tokens through an embedding layer.
        Now the order of tokens must be recorded using the positional embedding layer (pos_embedding).
        Initialize the Encoder sub layers.
        Initialize the dropout and scale function.
        :param input_dim: size of input vocabulary, (n_src_vocab from original transformer).
        :param hid_dim: Output dimension for encoder (d_model from original transformer).
        :param n_layers: number of identical stack layers.
        :param n_heads: number of heads in multi-head attention.
        :param pf_dim: Position-wise Feedforward Layer dimension, usually larger than hid_dim.
        :param dropout: nn.Dropout()
        :param device: run model on GPU or CPU.
        :param max_length: maximum length of sequence default 1000
        """

        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)

        # positional embedding: position of the token within the sequence, starting with
        # the first token, (<sos>) in position 0. The position embedding has a "vocabulary"
        # size of 100, which means our model can accept sentences up to 100 tokens long.
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        """
        Used in the encoder layers to mask the multi-head attention mechanisms.
        This is to calculate and apply attention over the input query.
        the model does not pay attention to <pad>.

        The tok_embedding and pos_embedding are summed together elementwise which returns a vector
        containing information about the token and it's position within the sentence.
        The token embeddings are multiplied by scaling factor (hid_dim) to reduce variance in
        embeddings. Dropout is applied to the combined embeddings.
        The combined embeddings are then passed through N encoder layers to get src:
        [batch size, src len, hid dim]
        :param src: the source sentence. [batch size, src len]
        :param src_mask: Same shape as source sentence, but 1 when
         the token in src sentence is not <pad>, else, 0. [batch size, 1, 1, src len]
        :return src: combined embeddings are then passed through N encoder layers.
        [batch size, src len, hid dim]
        """

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch size, src len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch size, src len, hid dim]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        return src


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=10000):
        """
        Similar to encoder, the tokens and positional embedding layers are initialized.
        The decoder layer is initialized.
        The decoder representation after the Nth layer is then passed through a linear layer (fc_out).
        The softmax operation is contained within the loss function.
        :param output_dim: size of output vocabulary, (n_trg_vocab from original transformer).
        :param hid_dim: Output dimension for decoder (d_model from original transformer).
        :param n_layers: number of identical stack layers.
        :param n_heads: number of heads in multi-head attention.
        :param pf_dim: Position-wise Feedforward Layer dimension, usually larger than hid_dim.
        :param dropout: nn.Dropout().
        :param device: run model on GPU or CPU.
        :param max_length: maximum length of source code string default 10000.
        """
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        The decoder's combined embeddings are passed through N (DEC_LAYERS) decoder layers with the
        encoded source (enc_src) and the source and target masks.
        The decoder representation after the Nth layer is passed through a linear layer (fc_out).
        :param trg: target sequence. [batch size, trg len]
        :param enc_src: Encoded source sentence. [batch size, src len, hid dim]
        :param trg_mask: Target sequence mask, to prevent decoder from paying attention to tokens that are ahead of it's
         current position. [batch size, 1, trg len, trg len]
        :param src_mask: Same shape as source sentence, but 1 when
         the token in src sentence is not <pad>, else, 0. [batch size, 1, 1, src len]
        :return output: output probabilities.
        :return attention: Normalized attention values.
        """
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        """
        Encapsulates the encoder and decoder, as well as handling the creation of the masks.
        :param encoder: The encoder function.
        :param decoder: The decoder function.
        :param src_pad_idx: Source sequence tokenized and changed to integers for mask creation.
        :param trg_pad_idx: Target sequence tokenized and changed to integers for mask creation.
        (elements below the diagonal matrix will be set to the value in the input tensor).
        :param device: run model on GPU or CPU.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        The source mask is created by checking where the input sequence is not equal to a <pad> token.
        1 where the token is not a <pad> token, 0 otherwise.
        Unsqueeze to (1,2) singleton dimension so that energy can be applied on each item within src.
        :param src: Source sequence. [batch size, src len],
        :return src_mask: Source mask. [batch size, n heads, seq len, seq len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        Create a mask for the <pad> tokens like in source mask.
        Then create a subsequent mask which is a diagonal matrix where:
        1. the elements above the diagonal will be zero.
        2. the elements below the diagonal will be set to the value in the input tensor.
        The subsequent mask is now concatenated with the padding mask using "AND" operator to combine
        the two masks ensuring both the subsequent tokens and the padding tokens cannot be attended to.
        :param trg: Target sequence. [batch size, trg len]
        :return trg_mask: Target mask. [batch size, 1, trg len, trg len]
        """

        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        # trg_sub_mask = [trg len, trg len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_mask = [batch size, 1, trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        """
        construct the following from the functions above:
        1. source mask.
        2. target mask.
        3. The encoder.
        4. the Decoder.
        :param src: Source sequence. [batch size, src len]
        :param trg: Target sequence. [batch size, trg len]
        :return output: output probabilities. [batch size, trg len, output dim]
        :return attention: Normalized attention values. [batch size, n heads, trg len, src len]
        """

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # enc_src = [batch size, src len, hid dim]
        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
