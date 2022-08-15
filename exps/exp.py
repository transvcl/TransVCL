#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch.nn as nn
import json
from transvcl.utils import wait_for_the_master, get_local_rank
from transvcl.models import TransVCL, YOLOPAFPN, YOLOXHead

class Exp(object):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"
        self.num_classes = 1
        self.eval_interval = 1

        self.default_feat_length = 1200
        self.default_feat_dim = 256

        self.vta_config = {
            'd_model': 256,
            'nhead': 8,
            'layer_names': ['self', 'cross'] * 1,
            'attention': 'linear',
            'match_type': 'dual_softmax',
            'dsmax_temperature': 0.1,
            'keep_ratio': False,
            'unsupervised_weight': 0.5
        }

    def get_model(self):
        def init_transvcl(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = TransVCL(self.vta_config, backbone, head)

        self.model.apply(init_transvcl)
        self.model.head.initialize_biases(1e-2)
        return self.model






