#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : re_util.py
@Author  : huanggj
@Time    : 2022/12/6 20:40
"""
import re

def get_split_list(text):
    """
    返回字符串分割后，不为空的元素
    @param text:
    @return:
    """
    question_text_list = re.findall('[^。]*。', text)
    passage = '[CLS]'.join(question_text_list)
    return question_text_list, passage