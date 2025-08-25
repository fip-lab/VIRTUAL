# encoding=utf-8
import torch
import torch.nn as nn
#from transformers import BertModel, BertConfig
from transformers import AutoModel
from torch.nn import Conv1d
from torch.nn import CrossEntropyLoss
from torch.nn import TransformerEncoderLayer

# Longformer
class Model(nn.Module):
    def __init__(self, config):
        print("longformer")
        super(Model, self).__init__()
        print(config.length)
        #self.config = BertConfig.from_pretrained(config.bert_path)
        self.pretrain_model = AutoModel.from_pretrained(config.pretrain_model_path)  # [batch_size, length, 768]
        self.dropout = nn.Dropout(config.dropout_rate)
        hidden_size = self.pretrain_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 4)

    def forward(self, input):
        all_ids = input[0]
        all_attention_mask = input[1]

        representation = self.pretrain_model(input_ids=all_ids,
                                   attention_mask=all_attention_mask)[1]


        output_dropout = self.dropout(representation)
        logits = self.classifier(output_dropout)

        return logits



