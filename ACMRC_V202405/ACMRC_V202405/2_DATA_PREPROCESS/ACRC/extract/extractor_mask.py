import io, json, codecs, os
import random
import extractor_base
from typing import List, Tuple

ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)


import re
from tqdm import tqdm
import numpy as np
from pprint import pprint
import pickle as pk
import jieba 
from sentence_transformers import SentenceTransformer, util

# https://cloud.tencent.com/developer/article/1530340
# 获取文章句子列表
def sent_tokenize(para):
    # 单字符断句符
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    # 英文省略号
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    # 中文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 段尾如果有多余的\n就去掉它
    para = para.rstrip()
    # 很多规则中会考虑分号，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    para = para.split("\n")
    ret = []
    for s in para:
        if len(s) > 0:
            ret.append(s)
    return ret

class ExtractorCosine(extractor_base):
    def __init__(self):
        pass


    def extract_rank(self, option: List[str], passage: str) -> List:
        return []


   
if __name__ == '__main__':
    
    '''
    Loading external resources:
    1. Word vectors
    sgns.merge.char_word-list and sgns.merge.char_vector-list 
    are word vectors extracted from [1], 
    containing all the words appearing in VGaokao.
    [1] https://github.com/Embedding/Chinese-Word-Vectors

    2. Sentence-BERT
    We use the distiluse-base-multilingual-cased-v1 model in [2].
    [2] https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
    '''
    ''' 
    证据抽取
    '''


    # word vector 资源加载
    word_list = pk.load(open("../data/external/word_vectors/sgns.merge.char_word-list", 'rb'))
    print(len(word_list))
    vector_list = pk.load(open("../data/external/word_vectors/sgns.merge.char_vector-list", 'rb'))
    print("vector loaded")

    # embedding 对象
    #embedding = Embedding(word_list, vector_list)

    # # simple_stop_list : 特殊符号/词语
    # dense_embedder = DenseEmbedder(embedding, jieba.cut, simple_stop_list)
    # if os.path.exists('../data/external/distiluse-base-multilingual-cased-v1'):
    #     bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')
    # else:
    #     bert_embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    #
    # '''
    # Start extracting evidence
    # '''
    # iteratively_retrieve_evidence('../data/raw/train_acmrc.json', '../data/processed/train_mrc_iterative.json')
    # iteratively_retrieve_evidence('../data/raw/test_acmrc.json', '../data/processed/test_mrc_iterative.json')

