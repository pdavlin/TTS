import torch
from torch import nn


class AdaDecoder(nn.Module):
    """Decoder with AdaTransformer. No change from FFT except to pass speaker embedding.

    Default params
            params={
                'hidden_channels_ffn': 1024,
                'num_heads': 2,
                "dropout_p": 0.1,
                "num_layers": 6,
            }

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels including Transformer layers.
        params (dict): dictionary for residual convolutional blocks.
    """

    def __init__(self, in_channels, out_channels, params):

        super().__init__()
        self.transformer_block = AdaTransformerBlock(in_channels, **params)
        self.postnet = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, x_mask=None, g=None, debug=False):  # pylint: disable=unused-argument
        if debug == True:
            print('in AdaDecoder')
            print(f'x:        {x.size()}')
            print(f'x_mask:   {x_mask.size()}')
            print(f'g:        {g.size()}')
        x_mask = 1 if x_mask is None else x_mask
        o = self.transformer_block(x, g=g, debug=debug) * x_mask
        o = self.postnet(o) * x_mask
        return o


# Custom FFT block for AdaSpeech to enable Conditional LayerNorm


class AdaTransformerBlock(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn, num_layers, dropout_p):
        super().__init__()
        self.fft_layers = nn.ModuleList(
            [
                AdaTransformer(
                    in_out_channels=in_out_channels,
                    num_heads=num_heads,
                    hidden_channels_ffn=hidden_channels_ffn,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, g=None, debug=False):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - mask:  :math:`[B, 1, T] or [B, T]`
        """
        if mask is not None and mask.ndim == 3:
            mask = mask.squeeze(1)
            # mask is negated, torch uses 1s and 0s reversely.
            mask = ~mask.bool()
        alignments = []
        for layer in self.fft_layers:
            x, align = layer(x, src_key_padding_mask=mask, g=g, debug=debug)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)
        return x


class AdaTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_fft=3, dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            in_out_channels, num_heads, dropout=dropout_p)

        padding = (kernel_size_fft - 1) // 2
        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn,
                               kernel_size=kernel_size_fft, padding=padding)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels,
                               kernel_size=kernel_size_fft, padding=padding)

        # AdaSpeech Conditional LayerNormalization
        self.norm1 = ConditionalLayerNorm(in_out_channels)
        self.norm2 = ConditionalLayerNorm(in_out_channels)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, g=None, debug=False):
        if debug == True:
            print('AdaTransformer forward')
            print(f'src:        {src.size()}')
            if g is not None:
                print(f'g:          {g.size()}')
        src = src.permute(2, 0, 1)
        src2, enc_align = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src + src2, g=g, debug=debug)
        # T x B x D -> B x D x T
        src = src.permute(1, 2, 0)
        src2 = self.conv2(nn.functional.relu(self.conv1(src)))
        src2 = self.dropout2(src2)
        src = src + src2
        src = src.transpose(1, 2)
        src = self.norm2(src, g=g, transpose=True, debug=debug)
        src = src.transpose(1, 2)
        return src, enc_align


class ConditionalLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):

        super().__init__()
        self.channels = channels
        self.eps = eps

        # TODO: Could these be a single layer
        self.gamma = nn.Linear(channels, channels)
        self.beta = nn.Linear(channels, channels)


    def forward(self, x, g=None, transpose=False, debug=False):
        if debug == True:
            print('ConditionalLayerNorm forward')
            print(f'x:          {x.size()}')
            print(f'g:          {g.size()}')
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        if g is not None:
            multiplier = self.gamma(g.transpose(1, 2)).transpose(
                0, 1)
            adder = self.beta(g.transpose(1, 2)).transpose(
                0, 1)
            # have to transpose in second layer because the channels in
            # `x` are in a different order
            if transpose == True: 
                multiplier = multiplier.permute(1, 0, 2)
                adder = adder.permute(1, 0, 2)
            if debug == True:
                print(f'multiplier: {multiplier.size()}')
                print(f'adder:      {adder.size()}')
            x = x * multiplier + adder

        return x
