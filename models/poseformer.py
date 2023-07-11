import copy
from typing import Optional, List

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.graformer import GraFormer
from utils import initialize_masks
# class MLP(nn.Module):
#     """
#     Args:
#         in_channels (int): number of input channels
#         representation_size (int): size of the intermediate representation
#     """

#     def __init__(self, in_channels, interim_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_channels, interim_size)
#         self.fc2 = nn.Linear(interim_size, output_size)

#     def forward(self, x):
#         # x = x.flatten(start_dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         return x


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, num_tokens, hid_dim=32, num_layers=4, nhead=8, normalize_before=False) -> None:
        super().__init__()

        self.embed_layer = nn.Linear(input_dim, hid_dim)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hid_dim))
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=False)
        temporal_encoder_norm = nn.LayerNorm(hid_dim) if normalize_before else None
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_layers, norm=temporal_encoder_norm)
    
    def forward(self, src):
        bs, t, d = src.shape
        src = src.view(bs * t, d)
        src = self.embed_layer(src)
        src = src.view(bs, t, d)
        src = src + self.temporal_pos_embed
        src = self.temporal_encoder(src)
        return src

class PoseFormer(nn.Module):

    def __init__(self, input_dim=2, output_dim=3, d_model=512, nhead=8, num_encoder_layers=4, num_kps=21, num_frames=9,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        # self.embed_layer = MLP(input_dim, d_model, d_model)
        self.embed_layer = nn.Linear(input_dim, d_model)

        # self.spatial_pos_embed = PositionalEncoding(d_model, dropout)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_kps, d_model))
        spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
        spatial_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.spatial_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=num_encoder_layers, norm=spatial_encoder_norm)

        # self.temporal_pos_embed = PositionalEncoding(d_model * num_kps, dropout)

        temporal_d_model = d_model * num_kps
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, temporal_d_model))
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=temporal_d_model, nhead=nhead, batch_first=False)
        temporal_encoder_norm = nn.LayerNorm(temporal_d_model) if normalize_before else None
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_encoder_layers, norm=temporal_encoder_norm)

        # self.pose_mlp = MLP(d_model, d_model, output_dim)
        self.avg_layer = torch.nn.Conv1d(in_channels=num_frames, out_channels=1, kernel_size=1)
        self.pose_head = nn.Sequential(nn.BatchNorm1d(temporal_d_model),
                                       nn.Linear(temporal_d_model, output_dim * num_kps)) #MLP(d_model, d_model, output_dim)

    def forward(self, src,
                spatial_mask: Optional[Tensor] = None, 
                temporal_mask: Optional[Tensor] = None):
        
        bs, t, n, d = src.shape
        
        src = src.view(bs * t * n, d)
        src = self.embed_layer(src)
        src = src.view(bs * t, n, self.d_model) #* math.sqrt(self.d_model)
        # src = self.spatial_pos_embed(src)

        spatial_features = src + self.spatial_pos_embed
        # spatial_mask = initialize_masks(bs * t, self.nhead, n, 0.2).to(src.device)
        # print(spatial_mask)
        spatial_features = self.spatial_encoder(src, spatial_mask)
        spatial_features = spatial_features.view(bs, t, n * self.d_model)        
        # spatial_features = self.temporal_pos_embed(spatial_features)

        temporal_features = spatial_features + self.temporal_pos_embed
        # temporal_mask = initialize_masks(bs, self.nhead, t, 0.2).to(src.device)
        temporal_features = self.temporal_encoder(temporal_features, temporal_mask)
        temporal_features = self.avg_layer(temporal_features).squeeze(1)
        output = self.pose_head(temporal_features)
        output = output.view(bs, n, self.output_dim)

        return output

class PoseGraFormer(nn.Module):

    def __init__(self, input_dim=2, output_dim=3, d_model=128, nhead=8, num_encoder_layers=4, num_kps=21, num_frames=9):
        
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.embed_layer = nn.Linear(input_dim, d_model)
        self.spatial_encoder = GraFormer(hid_dim=d_model, coords_dim=(input_dim, d_model), 
                                        num_layers=num_encoder_layers, n_head=nhead,  num_pts=num_kps)

        temporal_d_model = d_model * num_kps
        self.temporal_encoder = GraFormer(hid_dim=temporal_d_model, coords_dim=(temporal_d_model, output_dim * num_kps), 
                                        num_layers=num_encoder_layers, n_head=nhead,  num_pts=num_frames, temporal=True)

        # self.avg_layer = torch.nn.Conv1d(in_channels=num_frames, out_channels=1, kernel_size=1)
        # self.pose_head = nn.Sequential(nn.BatchNorm1d(temporal_d_model),
        #                                nn.Linear(temporal_d_model, output_dim * num_kps)) #MLP(d_model, d_model, output_dim)

    def forward(self, src,
                spatial_mask: Optional[Tensor] = None, 
                temporal_mask: Optional[Tensor] = None):
        
        bs, t, n, d = src.shape        
        src = src.view(bs * t, n, d)

        spatial_features = self.spatial_encoder(src)
        spatial_features = spatial_features.view(bs, t, n * self.d_model)        

        temporal_features = self.temporal_encoder(spatial_features)
        output = temporal_features.view(bs, t, n, self.output_dim)
        # temporal_features = self.avg_layer(temporal_features).squeeze(1)
        
        # output = self.pose_head(temporal_features)
        # output = output.view(bs, n, self.output_dim)

        return output