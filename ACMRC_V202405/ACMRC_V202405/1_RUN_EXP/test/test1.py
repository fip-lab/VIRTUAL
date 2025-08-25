#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : test1.py
@Author  : huanggj
@Time    : 2023/2/19 23:33
"""

from transformers import MBart50TokenizerFast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/BERT/")
# 加载预训练的mBART-50 tokenizer
#tokenizer = MBart50TokenizerFast.from_pretrained("/disk2/huanggj/ACMRC_EXPERIMENT/pretrain/mbart25/")

# 定义问题、选项和上下文
source_language = "en_XX"  # 英语
context = "The movie was directed by Christopher Nolan."
question = "Who directed the movie?"
options = ["Quentin Tarantino", "Christopher Nolan", "Steven Spielberg"]

# 将问题、选项和上下文拼接成一个字符串
input_strings = []
for option in options:
    input_string = f"{context} {question} {option}"
    input_strings.append(input_string)

# 对输入字符串进行编码
encoded_inputs = tokenizer(input_strings, return_tensors="pt", padding=True)

print("Encoded inputs:", encoded_inputs)





