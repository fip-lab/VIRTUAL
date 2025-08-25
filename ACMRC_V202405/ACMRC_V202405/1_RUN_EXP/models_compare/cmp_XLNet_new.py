# #!/usr/local/bin/python3
# # -*- coding: utf-8 -*-
#
# """
# @File    : XLNet.py
# @Author  : huanggj
# @Time    : 2023/3/28 11:21
# """
# import torch
# import torch.nn as nn
# from transformers import XLNetModel, XLNetConfig
#
# class Model(nn.Module):
#     def __init__(self, num_classes=35, alpha=0.5):
#         self.alpha = alpha
#         super(Model, self).__init__()
#         self.net = XLNetModel.from_pretrained(xlnet_cfg.xlnet_path).cuda()
#         for name, param in self.net.named_parameters():
#             if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         self.MLP = nn.Sequential(
#             nn.Linear(768, 4, bias=True),
#         ).cuda()
#
#     def forward(self, x):
#         x = x.long()
#         x = self.net(x, output_all_encoded_layers=False).last_hidden_state
#         x = F.dropout(x, self.alpha, training=self.training)
#         x = torch.max(x, dim=1)[0]
#         x = self.MLP(x)
#         return torch.sigmoid(x)
#
#
#
