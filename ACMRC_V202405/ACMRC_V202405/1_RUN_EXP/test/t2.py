#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : t2.py
@Author  : huanggj
@Time    : 2023/4/28 23:11
"""
import json
import os

f = open("/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/base_data/acrc_train.json", "r", encoding="utf-8")

data = json.load(f)['data']

print(len(data))


