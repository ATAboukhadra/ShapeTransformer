from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
import scipy
from torch.nn.parameter import Parameter
# from mesh import Mesh

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def create_pose_edges(num_pts):
    hand_edges = [
        # Hand connectivity
        [0, 1], [1, 2], [2, 3], [3, 4], 
        [0, 5], [5, 6], [6, 7], [7, 8], 
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16], 
        [0, 17], [17, 18], [18, 19], [19, 20]]
    object_edges = [
        # Object connectivity
        [21, 22],[22, 24], [24, 23], [23, 21],
        [25, 26], [26, 28], [28, 27], [27, 25],
        [21, 25], [22, 26], [23, 27], [24, 28]
    ]

    edges = []
    if num_pts == 8:
        edges.extend(list(np.array(object_edges) - 21))
    elif num_pts == 21:
        edges.extend(hand_edges)
    elif num_pts == 29:
        edges.extend(hand_edges)
        edges.extend(object_edges)
    elif num_pts == 42: # Two Hands
        edges.extend(hand_edges)
        edges.extend(list(np.array(hand_edges) + 21))
    elif num_pts == 21 * 4: # Two Hands and Two Objects
        edges.extend(hand_edges)
        edges.extend(list(np.array(hand_edges) + 21))
        obj_edges = []
        for o in range(2):
            for i in range(21):
                for j in range(21):
                    obj_edges.append([42 + i + o * 21, 42 + j + o * 21])
        # print(obj_edges)
    return edges

def create_temporal_edges(num_pts):
    edges = [[i, i+1] for i in range(num_pts-1)]
    return edges

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adj_mx_from_edges(edges, num_pts=29):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx

class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _ResChebGC(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResChebGC, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x, self.adj)
        out = self.gconv2(out, self.adj)
        return residual + out

class ChebNet(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(ChebNet, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x):
        out = self.gconv1(x, self.adj)
        out = self.gconv2(out, self.adj)
        return out

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X

class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, num_pts=29, temporal=False):
        super(GraphNet, self).__init__()

        if temporal:
            edges = create_temporal_edges(num_pts)
        else:
            edges = create_pose_edges(num_pts)

        adj = adj_mx_from_edges(edges = edges, num_pts = num_pts) 

        self.A_hat = Parameter(adj.float(), requires_grad=True)   

        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1

class GraFormer(nn.Module):
    def __init__(self, hid_dim=128, coords_dim=(2, 3), num_layers=4, n_head=4,  dropout=0.1, num_pts=29, temporal=False):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        # self.n_pts = num_pts
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, num_pts=num_pts, temporal=temporal)
        self.temporal = temporal
        if temporal:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_pts, hid_dim))

        # Create a non-trainable copy from the same adjacency matrix used in LAM_Gconv to use in ChebGConv
        self.adj = Parameter(gcn.A_hat.data.clone(), requires_grad=False)
    
        self.mask = torch.tensor([[[True] * num_pts]])

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim, hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=coords_dim[1], K=2)

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

    def forward(self, x):
        self.mask = self.mask.to(x.device)
        out = self.gconv_input(x, self.adj)
        if self.temporal:
            out = out + self.temporal_pos_embed
            
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, self.mask)
            out = self.gconv_layers[i](out)
        out = self.gconv_output(out, self.adj)
        return out
