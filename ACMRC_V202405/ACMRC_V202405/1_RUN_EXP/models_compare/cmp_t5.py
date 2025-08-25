# encoding=utf-8
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, T5EncoderModel
from torch.nn import Conv1d
from torch.nn import CrossEntropyLoss
from torch.nn import TransformerEncoderLayer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrain_model = T5EncoderModel.from_pretrained(config.pretrain_model_path) # [batch_size, length, 768]
        hidden_size = self.pretrain_model.config.hidden_size
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(hidden_size, 4)

    def forward(self, input):
        ids = input[0]
        attention_mask = input[1]

        question_representation = self.pretrain_model(input_ids=ids, attention_mask=attention_mask).last_hidden_state
        representation = question_representation[:, 0, :]

        output_dropout = self.dropout(representation)
        logits = self.classifier(output_dropout)

        return logits



