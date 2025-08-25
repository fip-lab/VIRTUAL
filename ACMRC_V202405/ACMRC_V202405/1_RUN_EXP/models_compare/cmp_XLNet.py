#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : XLNet.py
@Author  : huanggj
@Time    : 2023/3/28 11:21
"""
import torch
import torch.nn as nn
from transformers import XLNetModel

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrain_model = XLNetModel.from_pretrained(config.pretrain_model_path)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.pretrain_model.config.hidden_size, 4)




    def forward(self, input):
        input_ids = input[0]
        attention_mask = input[1]

        representation = self.pretrain_model(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state

        output_dropout = self.dropout(representation)
        output_dropout = torch.max(output_dropout, dim=1)[0]
        logits = self.classifier(output_dropout)

        return logits

