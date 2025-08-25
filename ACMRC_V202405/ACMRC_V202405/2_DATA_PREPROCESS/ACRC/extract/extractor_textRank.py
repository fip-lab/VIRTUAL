#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : extractor_textRank.py
@Author  : huanggj
@Time    : 2022/8/6 11:42
"""

# coding=utf-8
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba.analyse
from snownlp import SnowNLP
import pandas as pd
import numpy as np
import math

from extract.extractor_base import ExtractionBase
from mrc.Logger import Logger
# def print_arg(fn):
#     def sayspam(*args):
#         print(args)
#         fn(*args)
#
#     return sayspam

class ExtractorTextRank(ExtractionBase):

    POLICY_AROUND = 'POLICY_AROUND'
    POLICY_MAX_SCORE = 'POLICY_MAX_SCORE'


    # 初始化方法
    def __init__(self, policy):
        ExtractionBase.__init__(self)
        # allow_speech_tags   --词性列表，用于过滤某些词性的词
        self.tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
        #self.LOGGER = Logger()
        self.policy = policy
    # 抽取入口:
    def extract(self, option_list, passage):
        # 参数检查
        if option_list is None or len(option_list) != 4:
            print(option_list)
            print(passage)
            print('ERROR *** 选项个数错误 ***请检查')
            return
            #raise Exception

        if passage is None or passage == '':
            print('ERROR *** 文章为None ***请检查')
            print(passage)
            print(option_list)
            return
            #raise Exception

        # for op in option_list:
        #     op = op.replace('A','').replace('B','').replace('C','').replace('D','')
        self.LOGGER.logger.warning("###################### new question start ##########################")
        self.LOGGER.logger.warning(" ")
        # 选项A
        flagA, similar_sentences_list_of_option_A = self.get_similar_sentences(option_list[0].replace('A',''), passage, self.OPTION_A)
        self.LOGGER.logger.warning(" ")
        # 选项B
        flagB, similar_sentences_list_of_option_B = self.get_similar_sentences(option_list[1].replace('B',''), passage, self.OPTION_B)
        self.LOGGER.logger.warning(" ")
        # 选项C
        flagC, similar_sentences_list_of_option_C = self.get_similar_sentences(option_list[2].replace('C',''), passage, self.OPTION_C)
        self.LOGGER.logger.warning(" ")
        # 选项D
        flagD, similar_sentences_list_of_option_D = self.get_similar_sentences(option_list[3].replace('D',''), passage, self.OPTION_D)
        # 全部选项
        #flagALL, similar_sentences_list_of_option_ALL = self.get_similar_sentences(''.join(option_list).replace('A','').replace('B','').replace('C','').replace('D',''), passage, self.OPTION_ALL)

        # 把单个选项的相关句子结果合并成一个元组 ([], [], [], [])
        all_similar_sentences_tuple = (similar_sentences_list_of_option_A,
                                       similar_sentences_list_of_option_B,
                                       similar_sentences_list_of_option_C,
                                       similar_sentences_list_of_option_D)

        if flagB:
            similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_B)

        if flagC:
            similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_C)

        if flagD:
            similar_sentences_list_of_option_A.extend(similar_sentences_list_of_option_D)

        # 把单个选项的相近结果合并成一个字符串
        all_similar_sentences_str = ''.join(similar_sentences_list_of_option_A)

        #self.LOGGER.logger.warning(all_similar_sentences_str)
        self.LOGGER.logger.warning(" ")
        self.LOGGER.logger.warning("###################### new question end ##########################")
        self.LOGGER.logger.warning(" ")
        #return all_similar_sentences_str, all_similar_sentences_tuple, similar_sentences_list_of_option_ALL
        return all_similar_sentences_str

    # 返回相关的句子：计算方法  text Rank
    def get_similar_sentences(self, option, passage, option_name):
        #passage_sentence_list = list(filter(None,passage.split('。')))
        passage_sentence_list = [x for x in passage.split('。') if x]
        sentence_list_length = len(passage_sentence_list)
        self.LOGGER.logger.warning("文章句子个数( %d ), 最大索引号( %d )" %(sentence_list_length, sentence_list_length -1))
        self.LOGGER.logger.warning(passage)
        # 获取text rank得分最大句子的前后两句
        if self.POLICY_AROUND == self.policy:
            max_similarity_score = -1
            max_similarity_score_index = 0
            for index in range(sentence_list_length):
                sentence = passage_sentence_list[index]
                sents_similarity_score = self.text_rank_similarity(option, sentence)
                if sents_similarity_score > max_similarity_score:
                    max_similarity_score = sents_similarity_score
                    max_similarity_score_index = index

            return_index_tup = None
            if max_similarity_score_index == 0:
                return_index_tup = (0 ,1 ,2)
            elif max_similarity_score_index == sentence_list_length - 1:
                return_index_tup = (sentence_list_length - 3 ,sentence_list_length - 2 ,sentence_list_length - 1)
            else:
                return_index_tup = (max_similarity_score_index - 1, max_similarity_score_index, max_similarity_score_index + 1)

            if len(passage_sentence_list) < 3:
                self.LOGGER.logger.warning("选项( %s ): 文章句子小于3，直接返回原文")
                return False,passage_sentence_list
            self.LOGGER.logger.warning("选项( %s ): 最大Text Rank得分( %.2f), 句子索引号( %d ), 返回句子索引(%d, %d, %d)" %(option_name, max_similarity_score, max_similarity_score_index, return_index_tup[0] , return_index_tup[1], return_index_tup[2]))
            return_sentence_list = [passage_sentence_list[return_index_tup[0]],passage_sentence_list[return_index_tup[1]],passage_sentence_list[return_index_tup[2]]]
            self.LOGGER.logger.warning("选项文字 : %s"%option)
            self.LOGGER.logger.warning("相关句子 : %s"%return_sentence_list)
            #return return_sentence_list, return_index_tup, passage_sentence_list
            return True,return_sentence_list

        return None, None, None

    # 两个句子相似性 textRank
    def text_rank_similarity(self, option, sentence):
        #self.LOGGER.logger.warning(option)
        #self.LOGGER.logger.warning(sentence)
        counter = 0
        for sent in sentence:
            if sent in option:
                counter += 1
        sents_similarity=counter/(math.log(len(sentence))+math.log(len(option)))
        #self.LOGGER.logger.warning(sents_similarity)
        return sents_similarity

    # 关键词抽取
    def keywords_extraction(self, text):
        self.LOGGER.logger.warning("输入句子(%d)：%s" % (len(text), text))
        # self.DEBUG_LOGGER.logger.debug("输入句子(%d)：%s" % (len(text), text))
        # text    --  文本内容，字符串
        # window  --  窗口大小，int，用来构造单词之间的边。默认值为2
        # lower   --  是否将英文文本转换为小写，默认值为False
        # vertex_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点
        #                -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
        # edge_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边
        #              -- 默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数
        # pagerank_config  -- pagerank算法参数配置，阻尼系数为0.85
        self.tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                          pagerank_config={'alpha': 0.85, })
        # num           --  返回关键词数量
        # word_min_len  --  词的最小长度，默认值为1
        keywords = self.tr4w.get_keywords(num=6, word_min_len=2)
        self.LOGGER.logger.warning("关键词(%d)：%s" % (len(keywords), keywords))
        return keywords

    # 关键短语抽取
    def keyphrases_extraction(self,text):
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words', pagerank_config={'alpha': 0.85, })
        keyphrases = tr4w.get_keyphrases(keywords_num=8, min_occur_num=1)
        # keywords_num    --  抽取的关键词数量
        # min_occur_num   --  关键短语在文中的最少出现次数
        return keyphrases

    # 关键句抽取
    def keysentences_extraction(self,text):
        tr4s = TextRank4Sentence()
        tr4s.analyze(text, lower=True, source='all_filters')
        # text    -- 文本内容，字符串
        # lower   -- 是否将英文文本转换为小写，默认值为False
        # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
        # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
        # sim_func -- 指定计算句子相似度的函数
        # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
        keysentences = tr4s.get_key_sentences(num=4, sentence_min_len=3)
        return keysentences

    def keywords_textrank(self,text):
        keywords = jieba.analyse.textrank(text, topK=6)
        return keywords

    def debug_log(self,message):
        self.LOGGER.logger.warning(message)
        
    def error_log(self,message):
        self.LOGGER.logger.error(message)




# 关键词抽取
def keywords_extraction(text):

    #tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
    # allow_speech_tags   --词性列表，用于过滤某些词性的词
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.5, })
    # text    --  文本内容，字符串
    # window  --  窗口大小，int，用来构造单词之间的边。默认值为2
    # lower   --  是否将英文文本转换为小写，默认值为False
    # vertex_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点
    #                -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # edge_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边
    #              -- 默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数

    # pagerank_config  -- pagerank算法参数配置，阻尼系数为0.85
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    # num           --  返回关键词数量
    # word_min_len  --  词的最小长度，默认值为1
    print(text)
    return keywords

# 关键短语抽取
def keyphrases_extraction(text):

    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    keyphrases = tr4w.get_keyphrases(keywords_num=8, min_occur_num=1)
    # keywords_num    --  抽取的关键词数量
    # min_occur_num   --  关键短语在文中的最少出现次数
    print(text)
    return keyphrases

# 关键句抽取
def keysentences_extraction(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='all_filters')
    # text    -- 文本内容，字符串
    # lower   -- 是否将英文文本转换为小写，默认值为False
    # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
    # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # sim_func -- 指定计算句子相似度的函数

    # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
    keysentences = tr4s.get_key_sentences(num=4, sentence_min_len=3)
    return keysentences

def keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords

# 两个句子相似性 textRank
def two_sentences_similarity(sents_1, sents_2):
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    sents_similarity = counter / (math.log(len(sents_1)) + math.log(len(sents_2)))
    return sents_similarity



if __name__ == "__main__":
    text = "来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。"
    text3 = '戚继光，字元敬。幼倜傥负奇气。家贫，好读书，通经史大义。嘉靖中嗣职，用荐擢署都指挥佥事，备倭山东。三十六年，倭犯乐清、瑞安、临海，继光援不及，以道阻不罪。继光至浙时，见卫所军不习战，而金华、义乌俗称慓悍，请召募三千人，教以击刺法，长短兵迭用，由是继光一军特精。又以南方多薮泽，不利驰逐，乃因地形制阵法，审步伐便利，一切战舰、火器、兵械精求而更置之。“戚家军”名闻天下。四十年，倭大掠桃渚、圻头。继光急趋宁海，扼桃渚，败之龙山，追至雁门岭。贼遁去，乘虚袭台州。继光手歼其魁，蹙余贼瓜陵江尽死。先后九战皆捷，俘戴一千有奇，焚溺死者无算。明年，倭大举犯福建。闽中连告急，宗宪复檄继光剿之。先击横屿贼。人持草一束，填壕进。大破其巢，斩首二千六百。乘胜至福清，连克六十营，斩首千数百级。继光为将号令严，赏罚信，士无敢不用命。隆庆初，给事中吴时来以蓟门多警，请召大猷、继光专训边卒。继光乃议立车营。车一辆用四人推挽，战则结方阵，而马步军处其中。又制拒马器，体轻便利，遏寇骑冲突。寇至，火器先发，稍近则步军持拒马器排列而前，间以长枪、锒筅。寇奔，则骑军逐北。节制精明，器核犀利，蓟门军容遽为诸边冠。自嘉靖庚戌俺答犯京师，边防独重蓟。增兵益饷，骚动天下。继光在镇十六年，边备修伤，蓟门文然。继之者，置其成法，数十年得无事。亦赖当国大臣徐阶、高拱、张居正先后倚任之。居正尤事与商确，欲为继光难者，辄徙之去。居正及半岁，给事中张鼎思言继光不宜于北，当国者遵改之广东。继光悒悒不得志，强一赴，逾年即谢病。给事中张希皋等复勒之，竞罢归。居三年，御史得光宅疏存，反夺体。继光亦遂卒。'
    text1 = '以唐太宗纵囚的事例，证明死囚应约就死的难能可贵。'
    text2 = '无知而又自我欺骗，是默之愚。'
    option_d = '戚继光先荣后辱，晚景凄凉。戚继光曾经先后得到徐阶、高拱、张居正的信任倚重，煊赫一时，张居正去世后，他备受打击'
    option_c = '戚继光治军有方，边防安定。戚继光在蓟门练兵，设立车营，极大地提高了军队战斗力，蓟门边防得益于他的制度，数十年平安无事。'
    option_b = '戚继光抵御倭寇，战功卓著。嘉靖年间，倭寇进犯东南沿海，戚继光率军在浙江、福建连续九战九捷，杀敌无数，立下赫赫战功。'
    option_a = '戚继光幼好读书，通晓大义。戚继光家境贫寒，然从小就立有大志，热爱读书，通晓经史大义，后来代理都指挥佥事，到山东防御倭寇。'
    #extractor = ExtractorTextRank()
    #extractor.keywords_extraction(option_a)
    #extractor = ExtractorTextRank(ExtractorTextRank.POLICY_AROUND)
    # f = open('../dataset/test.csv')
    # lines = f.readlines()
    # for line in lines:
    #     arr = line.split(',')
    #     passage = arr[1]
    #     q_a = arr[0]
    #     a_list = q_a.split('[SEP]')[1:]
    #
    #     res = extractor.extract(a_list, passage)

    text_ = '刘长佑富有谋略，屡建战功。他虽然只是一个拔贡，但是他多次跟随江忠源作战，立下功劳，不断被提拔。刘长佑到云南就任云贵总督之后，边境少数民族杀了英国人马加理，刘长佑认为应该等云南境内的官民慢慢安定下来后，再派官员与英国人商议处理。刘长佑为政一方，善于治理。在广西匪情猖獗时，整顿吏治，强大军队，解决了军费问题，取得了很好的剿匪成果。刘长佑高瞻远瞩，深谋远虑。在法国军队窥视越南东京时，他就上疏说:“与其在失去越南边境后，再作守边考虑，不如趁法国军队才开始行动，就做好消除边境争端的谋划。”。'
    a = [
                  "李嗣业擅长陌刀，雄威显于敌阵。他担任过高仙芝的左陌刀将，军中信服他善用陌刀，在与叛军决战香积寺北时曾执陌刀阵前示威，一往无前，所向披靡。",
                  "李嗣业无惧艰险，孤胆降服强敌。攻打娑勒城，敌人依据险要地势，安营扎寨，居高临下，易守难攻，他孤身一人，走险路，突然出现在敌前，让敌军溃退。",
                  "李嗣业身先士卒，意志感奋军心。在与李归仁部战斗中，情势十分危急，他向郭子仪阐明意志之后，就又率先杀向敌军，前军之士随其后，终于让敌军溃败。",
                  "李嗣业建功立业，英名载人史册。安西征战，促使拂林、大食诸胡七十二国皆诚意来到边界归顺，贡献珍物；在平定“安史之乱”中也是战功卓著，深得倚重。"
               ]

    t = "".join(a)
    #print("textRank:")
    #print(two_sentences_similarity(text1,text2))
    #print('************************')
    # max_score = -1.0
    # max_score_text = ''
    # text = text.strip()
    # t = text.split("。")
    # for s in t:
    #     if s == '':
    #         continue
    #     s = s + '。'
    #     print(s)
    #     score = two_sentences_similarity(s,option_a)
    #     print(score)
    #     if score > max_score:
    #         max_score = score
    #         max_score_text = s
    #     print('-----------------')
    # print('************************')
    # print("max_score: %.2f"%max_score)
    # print("max_score_text : : %s"%max_score_text)
    # 关键词抽取
    print('***********key words*************')
    keywords = keywords_extraction(t)
    for k in keywords:

        print(k)


    keywords = keyphrases_extraction(t)
    print(keywords)
    # 关键短语抽取
    # keyphrases = keyphrases_extraction(text)
    # print(keyphrases)
    #
    # print('***********key words*************')
    # keyphrases = keyphrases_extraction(option_a)
    # print(keyphrases)
    # keyphrases = keyphrases_extraction(option_b)
    # print(keyphrases)
    # keyphrases = keyphrases_extraction(option_c)
    # print(keyphrases)
    # keyphrases = keyphrases_extraction(option_d)
    # print(keyphrases)

    # 关键句抽取
    # keysentences = keysentences_extraction(text)
    # print(keysentences)