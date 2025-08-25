# encoding=utf-8
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn import Conv1d
from torch.nn import CrossEntropyLoss
from torch.nn import TransformerEncoderLayer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config1 = BertConfig.from_pretrained(config.pretrain_model_path)
        self.plm1 = BertModel.from_pretrained(config.pretrain_model_path, config=self.config1)

        # self.config2 = BertConfig.from_pretrained(config.pretrain_model_path)
        # self.plm2 = BertModel.from_pretrained(config.pretrain_model_path, config=self.config2)
        #
        # self.config3 = BertConfig.from_pretrained(config.pretrain_model_path)
        # self.plm3 = BertModel.from_pretrained(config.pretrain_model_path, config=self.config3)

        hidden_size = self.plm1.config.hidden_size
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(3 * hidden_size, 4)

    def forward(self, input):
        input1_ids = input[0]
        input1_attention_mask = input[1]
        input2_ids = input[2]
        input2_attention_mask = input[3]
        input3_ids = input[4]
        input3_attention_mask = input[5]

        output1 = self.plm1(input_ids=input1_ids,
                                            attention_mask=input1_attention_mask)[0][:, 0]

        output2 = self.plm1(input_ids=input2_ids,
                                            attention_mask=input2_attention_mask)[0][:, 0]

        output3 = self.plm1(input_ids=input3_ids,
                                            attention_mask=input3_attention_mask)[0][:, 0]

        # concat the outputs from the three models
        outputs = torch.cat((output1, output2, output3), dim=1)

        output_dropout = self.dropout(outputs)
        logits = self.classifier(output_dropout)
        return logits



