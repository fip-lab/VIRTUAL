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

        print(length_cnn_layer0)
        self.cnn_layer1 = Conv1d(config.cnn_channels, config.cnn_channels, config.kernel_size,
                                 stride=config.stride)  # [batch_size, length_cnn_layer1, cnn_channels]
        length_cnn_layer1 = ((length_cnn_layer0 - config.kernel_size) / config.stride) + 1
        print(length_cnn_layer1)
        self.transformer_layer0 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.transformer_layer1 = TransformerEncoderLayer(config.cnn_channels, config.nhead)
        self.dropout = nn.Dropout(config.dropout_rate)
        print(int(length_cnn_layer1 * config.cnn_channels))
        self.classifier = nn.Linear(int(length_cnn_layer1 * config.cnn_channels), 4)

    def forward(self, input):
        input1_ids = input[0]
        input1_type_ids = input[1]
        input1_attention_mask = input[2]
        input2_ids = input[3]
        input2_type_ids = input[4]
        input2_attention_mask = input[5]
        input3_ids = input[6]
        input3_type_ids = input[7]
        input3_attention_mask = input[8]
        input4_ids = input[6]
        input4_type_ids = input[7]
        input4_attention_mask = input[8]
        input5_ids = input[6]
        input5_type_ids = input[7]
        input5_attention_mask = input[8]

        input1 = self.bert(input_ids=input1_ids,
                           token_type_ids=input1_type_ids,
                           attention_mask=input1_attention_mask).last_hidden_state

        input2 = self.bert(input_ids=input2_ids,
                           token_type_ids=input2_type_ids,
                           attention_mask=input2_attention_mask).last_hidden_state

        input3 = self.bert(input_ids=input3_ids,
                           token_type_ids=input3_type_ids,
                           attention_mask=input3_attention_mask).last_hidden_state

        input4 = self.bert(input_ids=input4_ids,
                           token_type_ids=input4_type_ids,
                           attention_mask=input4_attention_mask).last_hidden_state

        input5 = self.bert(input_ids=input5_ids,
                           token_type_ids=input5_type_ids,
                           attention_mask=input5_attention_mask).last_hidden_state

        union_representation1 = torch.cat((input1, input2), 1)
        union_representation2 = torch.cat((union_representation1, input3), 1)
        union_representation3 = torch.cat((union_representation2, input4), 1)
        union_representation = torch.cat((union_representation3, input5), 1)

        # [batch_size, 512 + 512 + 512 , 768]
        union_representation = union_representation.permute(0, 2, 1)
        output_cnn_layer0 = self.cnn_layer0(union_representation)
        output_cnn_layer1 = self.cnn_layer1(output_cnn_layer0).permute(0, 2, 1)

        output_transformer_layer0 = self.transformer_layer0(output_cnn_layer1)
        output_transformer_layer1 = self.transformer_layer1(output_transformer_layer0)

        batch_size = output_transformer_layer1.shape[0]
        length = output_transformer_layer1.shape[1]
        hidden_size = output_transformer_layer1.shape[2]
        output_layer1 = torch.reshape(output_transformer_layer1, (batch_size, length * hidden_size))

        output_dropout = self.dropout(output_layer1)
        logits = self.classifier(output_dropout)
        return logits



