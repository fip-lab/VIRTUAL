#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : BENCH_2.py
@Author  : huanggj
@Time    : 2023/2/17 1:16
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn import Conv1d
from torch.nn import TransformerEncoderLayer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(config.pretrain_model_path)
        self.bert = BertModel.from_pretrained(config.pretrain_model_path, config=self.config)
        self.cnn_layer0 = Conv1d(self.bert.config.hidden_size, config.cnn_channels, config.kernel_size,
                                 stride=config.stride)  # [batch_size, length_cnn_layer0, cnn_channels]
        length_cnn_layer0 = ((config.length - config.kernel_size) / config.stride) + 1

        self.cnn_layer1 = Conv1d(config.cnn_channels, config.cnn_channels, config.kernel_size,
                                 stride=config.stride)  # [batch_size, length_cnn_layer1, cnn_channels]
        length_cnn_layer1 = ((length_cnn_layer0 - config.kernel_size) / config.stride) + 1
        self.transformer_layer0 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.transformer_layer1 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(int(length_cnn_layer1 * config.cnn_channels), 4)

    def forward(self, input):
        question_ids = input[0]
        question_attention_mask = input[1]
        passage_ids = input[2]
        passage_attention_mask = input[3]

        question_representation = self.bert(input_ids=question_ids,
                                            attention_mask=question_attention_mask).last_hidden_state

        passage_one_representation = self.bert(input_ids=passage_ids,
                                               attention_mask=passage_attention_mask).last_hidden_state

        union_representation = torch.cat((question_representation, passage_one_representation), 1)

        union_representation = union_representation.permute(0, 2, 1)
        output_cnn_layer0 = self.cnn_layer0(union_representation)
        output_cnn_layer1 = self.cnn_layer1(output_cnn_layer0).permute(0, 2, 1)

        output_transformer_layer0 = self.transformer_layer0(output_cnn_layer1)
        output_transformer_layer1 = self.transformer_layer1(output_transformer_layer0)

        batch_size = output_transformer_layer1.shape[0]
        length = output_transformer_layer1.shape[1]
        hidden_size = output_transformer_layer1.shape[2]
        output_transformer_layer1 = torch.reshape(output_transformer_layer1, (batch_size, length * hidden_size))

        output_dropout = self.dropout(output_transformer_layer1)
        logits = self.classifier(output_dropout)
        return logits



