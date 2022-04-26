import torch
from torch import nn
from TTS.tts.layers.generic.normalization import LayerNorm
from typing import Optional
import torch.nn.functional as F


class UtteranceEncoder(nn.Module):

    def __init__(self, idim: int,
                 n_chans: int = 256,
                 kernel_size: int = 5,
                 dropout_rate: float = 0.5,
                 stride: int = 3):
        super(UtteranceEncoder, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                idim, # number of mels, 80
                n_chans, # hidden channels, (args.hidden_channels)
                kernel_size, # 5
                stride=stride, # 3
                padding=(kernel_size - 1) // 2, # 2
            ),
            torch.nn.ReLU(),
            LayerNorm(n_chans),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv1d(
                n_chans, # hidden channels, (args.hidden_channels)
                n_chans, # hidden channels, (args.hidden_channels)
                kernel_size, # 5
                stride=stride, # 3
                padding=(kernel_size - 1) // 2, # 2
            ),
            torch.nn.ReLU(),
            LayerNorm(n_chans), # 
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # print('---utterance encoder---')
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = F.avg_pool1d(xs, xs.size(-1))  # (B, C, 1)
        # print('utterance output size ' + str(xs.size()))
        # print('utterance encoder finished')
        return xs


class PhonemeLevelEncoder(nn.Module):
    def __init__(self, idim: int,
                 n_chans: int = 256,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.5,
                 stride: int = 1):
        super(PhonemeLevelEncoder, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                idim,  # number of mels, 80
                n_chans,  # number of channels (args.hidden_channels)
                kernel_size,  # 3
                stride=stride,  # 1
                padding=(kernel_size - 1) // 2,  # 1
            ),
            torch.nn.ReLU(),
            LayerNorm(n_chans),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv1d(
                n_chans,  # number of channels (args.hidden_channels)
                n_chans,  # number of channels (args.hidden_channels)
                kernel_size,  # 3
                stride=stride,  # 1
                padding=(kernel_size - 1) // 2,  # 1
            ),
            torch.nn.ReLU(),
            LayerNorm(n_chans),
            torch.nn.Dropout(dropout_rate),
        )
        self.linear = torch.nn.Linear(n_chans, n_chans) # needs to output same number of channels as input

    def forward(self,
                tensor: torch.Tensor,
                ) -> torch.Tensor:

        for f in self.conv:
            tensor = f(tensor)
        tensor = self.linear(tensor.transpose(1, 2))
        return tensor
