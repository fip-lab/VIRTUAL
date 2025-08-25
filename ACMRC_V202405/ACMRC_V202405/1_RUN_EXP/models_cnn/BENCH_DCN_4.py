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
from dcn_v2 import DCN

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        d_model = 768
        if 'macbert' in config.pretrain_model_path:
            d_model = 1024

        nhead = 8
        num_layers = 1
        self.config = BertConfig.from_pretrained(config.pretrain_model_path)
        self.plm = BertModel.from_pretrained(config.pretrain_model_path, config=self.config)
        #self.cnn_layer0 = DCN(768, 128, (3, 1), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.cnn_layer0 = DCN(in_channels=d_model, out_channels=d_model, stride=1,kernel_size=(1, 3), padding=(0, 1))
        #length_cnn_layer0 = ((config.length - config.kernel_size) / config.stride) + 1
        self.relu = nn.ReLU()
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,d_model))
        self.fc1 = nn.Linear(d_model, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)
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

        #print("\n embedding shape: ", representation.shape)

        union_representation = representation.permute(0, 2, 1).unsqueeze(3)

        #print("\n union_representation shape: ", union_representation.shape)

        x = self.cnn_layer0(union_representation.contiguous())

        #print("\n output_cnn_layer0 shape: ", x.shape)
        x = self.relu(x)
        x = x.squeeze(3).permute(2, 0, 1)
        #print("\n x transpose shape: ", x.shape)
        x = self.transformer_encoder(x)
        #print("\n x transformer_encoder shape: ", x.shape)
        x = x.permute(1, 0, 2)
        #print("\n x transpose shape: ", x.shape)
        x = self.global_avg_pool(x)
        #print("\n x global_avg_pool shape: ", x.shape)
        x = x.squeeze(1)
        #print("\n x squeeze shape: ", x.shape)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)

        return x



