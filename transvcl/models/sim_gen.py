#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable
from .linear_attention import LinearAttention, FullAttention
import numpy as np
import cv2


INF = 1e9

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, file=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query)
        key = self.k_proj(key)
        query = query.view(bs, -1, self.nhead, self.dim) # [N, L, (H, D)]
        key = key.view(bs, -1, self.nhead, self.dim) # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.5, max_len=1200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """
            Args:
                x (torch.Tensor): [N, L, C]
            """
        _, _, d = x.shape
        return x + d ** -0.5 * self.pe[:, :x.size(1)]

class FeatureTransformer(nn.Module):
    """Feature Transformer module."""

    def __init__(self, config):
        super(FeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, cls_token=False, file=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        if cls_token:
            N, L, _ = feat0.shape
            _, S, _ = feat1.shape
            cls_token_0 = self.cls_token.repeat(N, 1, 1)
            cls_token_1 = self.cls_token.repeat(N, 1, 1)
            feat0 = torch.cat((cls_token_0, feat0), dim=1)
            feat1 = torch.cat((cls_token_1, feat1), dim=1)
            mask0 = torch.cat((torch.tensor([1]).repeat(N, 1).to(mask0.device), mask0), dim=1)
            mask1 = torch.cat((torch.tensor([1]).repeat(N, 1).to(mask1.device), mask1), dim=1)
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0, file[0] + '_self_feat0')
                feat1 = layer(feat1, feat1, mask1, mask1, file[0] + '_self_feat1')
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1, file[0] + '_cross_feat0')
                feat1 = layer(feat1, feat0, mask1, mask0, file[0] + '_cross_feat1')
            else:
                raise KeyError

        return feat0, feat1

class SimMapGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            conf_matrix (torch.Tensor): [N, L, S]
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    0)
            # if mask_c0 is not None:
            #     sim_matrix = torch.where((mask_c0.long()[..., None] * mask_c1.long()[:, None]).bool(), sim_matrix, torch.tensor([0.]))
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        else:
            assert self.match_type == 'sinkhorn', f'match type {self.match_type} is neither sinkhorn nor dual_softmax'
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

        return conf_matrix
