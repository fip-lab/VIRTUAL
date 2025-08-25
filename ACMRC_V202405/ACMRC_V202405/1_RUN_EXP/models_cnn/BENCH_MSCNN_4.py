#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : BENCH_2.py
@Author  : huanggj
@Time    : 2023/2/17 1:16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn import Conv1d
from torch.nn import MultiheadAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(config.pretrain_model_path)
        self.plm = BertModel.from_pretrained(config.pretrain_model_path, config=self.config)

        embedding_dim = self.plm.config.hidden_size  # 假设使用BERT编码，词嵌入维度为768
        kernel_sizes = [3, 5, 7]  # 卷积核大小列表
        num_filters = 64  # 每个卷积核的过滤器数量
        num_classes = 4  # 输出类别数量
        dropout_rate = 0.1  # Dropout概率

        # 卷积层，具有不同卷积核大小
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        # Add Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=num_filters * len(kernel_sizes), nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input):
        input1_ids = input[0]
        input1_attention_mask = input[1]
        input2_ids = input[2]
        input2_attention_mask = input[3]
        input3_ids = input[4]
        input3_attention_mask = input[5]
        input4_ids = input[6]
        input4_attention_mask = input[7]

        input1_representation = self.plm(input_ids=input1_ids,
                                         attention_mask=input1_attention_mask).last_hidden_state

        input2_representation = self.plm(input_ids=input2_ids,
                                         attention_mask=input2_attention_mask).last_hidden_state

        input3_representation = self.plm(input_ids=input3_ids,
                                         attention_mask=input3_attention_mask).last_hidden_state

        input4_representation = self.plm(input_ids=input4_ids,
                                         attention_mask=input4_attention_mask).last_hidden_state

        tensor_list = [input1_representation, input2_representation, input3_representation, input4_representation]

        representation = torch.cat(tensor_list, dim=1)

        x = representation.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # Max pooling
        x = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in x]

        # Concatenating multi-kernel convolution results
        x = torch.cat(x, 1)

        # Transformer Encoder computation
        batch_size = x.size(0)
        x = x.view(batch_size, -1,
                   64 * 3)  # [batch_size, seq_len, num_filters * len(kernel_sizes)]
        x = self.transformer_encoder(x)  # [batch_size, seq_len, num_filters * len(kernel_sizes)]

        # Pooling across the sequence dimension
        x = x.mean(dim=1)  # [batch_size, num_filters * len(kernel_sizes)]

        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]

        return x




