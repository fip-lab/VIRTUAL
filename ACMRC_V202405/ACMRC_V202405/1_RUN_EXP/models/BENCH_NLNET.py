#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : BENCH_2.py
@Author  : huanggj
@Time    : 2023/2/17 1:16
"""

import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn import functional as F
from transformers import XLNetModel, XLNetTokenizer
from torch.nn import TransformerEncoderLayer
# 构建XLNet模型的类
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.pretrain_model_path)
        self.cnn_layer0 = Conv1d(self.xlnet.config.hidden_size, config.cnn_channels, config.kernel_size,
                                 stride=config.stride)  # [batch_size, length_cnn_layer0, cnn_channels]
        length_cnn_layer0 = ((config.length - config.kernel_size) / config.stride) + 1
        print(length_cnn_layer0)
        self.cnn_layer1 = Conv1d(config.cnn_channels, config.cnn_channels, config.kernel_size,
                                 stride=config.stride)  # [batch_size, length_cnn_layer1, cnn_channels]
        length_cnn_layer1 = ((length_cnn_layer0 - config.kernel_size) / config.stride) + 1
        self.transformer_layer0 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.transformer_layer1 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(int(length_cnn_layer1 * config.cnn_channels), 4)

    def forward(self, input_tuple):
        input_ids = input_tuple[0::2]  # 取出偶数索引的元素作为input_ids
        attention_mask = input_tuple[1::2]  # 取出奇数索引的元素作为attention_mask

        outputs = []
        for i in range(4):
            x = self.xlnet(input_ids=input_ids[i], attention_mask=attention_mask[i])
            pooled_output = x.last_hidden_state  # 使用[CLS]的输出作为文本的表示
            outputs.append(pooled_output)

        # 拼接四个XLNet的输出
        concat_output = torch.cat(outputs, dim=1)
        union_representation = concat_output.permute(0, 2, 1)
        output_cnn_layer0 = self.cnn_layer0(union_representation)
        output_cnn_layer1 = self.cnn_layer1(output_cnn_layer0).permute(0, 2, 1)

        output_transformer_layer0 = self.transformer_layer0(output_cnn_layer1)
        output_transformer_layer1 = self.transformer_layer1(output_transformer_layer0)

        batch_size = output_transformer_layer1.shape[0]
        length = output_transformer_layer1.shape[1]
        hidden_size = output_transformer_layer1.shape[2]
        output_transformer_layer1 = torch.reshape(output_transformer_layer1, (batch_size, length * hidden_size))

        #output_dropout = self.dropout(output_transformer_layer1)
        #logits = self.classifier(output_dropout)
        logits = self.classifier(output_transformer_layer1)
        return logits




