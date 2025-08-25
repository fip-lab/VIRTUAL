#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : TFIDF.py
@Author  : huanggj
@Time    : 2022/11/3 10:49
"""
from __future__ import absolute_import
import os
import jieba
import jieba.posseg
import thulac
from jieba.analyse import *
import re
import pkuseg
from operator import itemgetter

_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__), path))
_get_abs_path = jieba._get_abs_path

DEFAULT_IDF = _get_module_path("idf.txt")
DEFAULT_STOP_WORlDS = _get_module_path("stop_words.txt")


class KeywordExtractor(object):

    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def set_stop_words(self, stop_words_path):
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader(object):

    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf

class TFIDF(KeywordExtractor):

    def __init__(self, idf_path=None):
        self.tokenizer = jieba.dt
        self.postokenizer = jieba.posseg.dt
        #self.pku_tokenizer = pkuseg.pkuseg()
        self.thu_tokenizer = thulac.thulac()
        self.stop_words = set()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def stop_words_list(self):
        stopword = [line.strip() for line in open(DEFAULT_STOP_WORlDS, 'r', encoding='utf8').readlines()]  # 以行的形式读取停用词表，同时转换为列表
        return stopword

    def pku_cut(self, text):
        rst = self.pku_tokenizer.cut(text)
        print(rst)

    def thu_cut(self, text):
        rst = self.thu_tokenizer.cut(text)
        print(rst)

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            #words = self.tokenizer.cut(sentence)
            words = self.thu_tokenizer.cut(sentence)
            words = [x[0]  for x in words]
            #print(words)

        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                #print(1)
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags




