# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MLP(nn.Module):
    """
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, interim_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, interim_size)
        self.fc2 = nn.Linear(interim_size, output_size)

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PoseTransformer(nn.Module):

    def __init__(self, input_dim=2, output_dim=3, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model
        
        self.embed_layer = MLP(input_dim, d_model, d_model)
        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.pos_embed = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        self.pose_mlp = MLP(d_model, d_model, output_dim)

    def forward(self, src,
                mask: Optional[Tensor] = None):
        
        bs, t, n, d = src.shape
        src = src.view(bs * t * n, d)
        src = self.embed_layer(src).view(bs, t * n, self.d_model) * math.sqrt(self.d_model)

        # src.resize
        src = src.permute(1, 0, 2) # To handle batch_first=False in case of default transformer encoder
        src = self.pos_embed(src)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)
        # output = self.encoder(src, mask, src_key_padding_mask, pos)
        output = self.encoder(src, mask)
        output = output.permute(1, 0, 2).reshape(bs * t * n, self.d_model)
        output = self.pose_mlp(output)
        output = output.reshape(bs, t, n, 3)

        return output