#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_base.py
@Author  : huanggj
@Time    : 2022/8/4 14:41
"""
from abc import   abstractmethod
from typing import Dict, List, Tuple

# base class of extraction
class ExtractionBase:
    OPTION = 'OPTION'
    PASSAGE = 'PASSAGE'
    OPTION_A = 'A'
    OPTION_B = 'B'
    OPTION_C = 'C'
    OPTION_D = 'D'
    OPTION_ALL = 'ALL'
    def __init__(self, option_model_name='', passage_model_name=''):
        # name of pretrained language model
        self.option_model_name = option_model_name
        self.passage_model_name = passage_model_name
        # self.DEBUG_LOGGER = Logger(level='debug')
        # self.ERROR_LOGGER = Logger(level='error')

    #  定义抽象方法 : 每一种抽取实现类各自实现这个抽象方法
    @abstractmethod
    def extract(self, option : List[str], passage: str) -> str:
        return passage

    @abstractmethod
    def extract_rank(self, option: List[str], passage: str) -> List:
        return []

