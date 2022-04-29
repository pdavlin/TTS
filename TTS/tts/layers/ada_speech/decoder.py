import torch
from torch import nn

from TTS.tts.layers.generic.res_conv_bn import Conv1dBN, Conv1dBNBlock, ResidualConv1dBNBlock
from TTS.tts.layers.generic.transformer import FFTransformerBlock
from TTS.tts.layers.generic.wavenet import WNBlocks
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer

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

    def forward(self, x, x_mask=None, g=None):  # pylint: disable=unused-argument
        # TODO: handle multi-speaker
        x_mask = 1 if x_mask is None else x_mask
        o = self.transformer_block(x, g) * x_mask
        o = self.postnet(o) * x_mask
        return o


# Custom FFT block for AdaSpeech to enable Conditional LayerNorm

class AdaTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_fft=3, dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_out_channels, num_heads, dropout=dropout_p)

        padding = (kernel_size_fft - 1) // 2
        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn, kernel_size=kernel_size_fft, padding=padding)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels, kernel_size=kernel_size_fft, padding=padding)

        self.norm1 = nn.ConditionalLayerNorm(in_out_channels)
        self.norm2 = nn.ConditionalLayerNorm(in_out_channels)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """😦 ugly looking with all the transposing"""
        src = src.permute(2, 0, 1)
        src2, enc_align = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src + src2)
        # T x B x D -> B x D x T
        src = src.permute(1, 2, 0)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = self.dropout2(src2)
        src = src + src2
        src = src.transpose(1, 2)
        src = self.norm2(src)
        src = src.transpose(1, 2)
        return src, enc_align


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

    def forward(self, x, mask=None, g=None):  # pylint: disable=unused-argument
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
            x, align = layer(x, src_key_padding_mask=mask, g=g)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)
        return x

class ConditionalLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        """Layer norm for the 2nd dimension of the input.
        Args:
            channels (int): number of channels (2nd dimension) of the input.
            eps (float): to prevent 0 division

        Shapes:
            - input: (B, C, T)
            - output: (B, C, T)
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Linear(channels, channels)
        self.beta = nn.Linear(channels, channels)

        # self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        # self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, g=None):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma(g) + self.beta(g)
        return x