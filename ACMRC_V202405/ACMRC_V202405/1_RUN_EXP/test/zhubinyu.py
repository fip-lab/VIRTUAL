#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : zhubinyu.py
@Author  : huanggj
@Time    : 2023/5/30 13:30
"""
# import spacy
#
# # 加载中文模型
# nlp = spacy.load("zh_core_web_sm")
#
# # 解析句子
# doc = nlp("庾衮留下来照顾染上瘟疫、生命垂危的二哥，疫势过后，庾衮无恙，父老连声感叹，认为瘟疫不传染。")
#
# # 找出主语和宾语
# for token in doc:
#     if "nsubj" in token.dep_:
#         print("主语:", token.text)
#     if "dobj" in token.dep_:
#         print("宾语:", token.text)

import spacy
import json
import random
import copy
import os

# Load Chinese language model
nlp = spacy.load("zh_core_web_sm")

# 换主语宾语
def swap_subject_object(sentence):
    # Parse the sentence
    doc = nlp(sentence)

    # Extract subject and object
    subj = [tok for tok in doc if (tok.dep_ in ['nsubj', 'nsubjpass'])]
    obj = [tok for tok in doc if (tok.dep_ in ['dobj', 'pobj'])]

    # Make sure we have at least one subject and one object
    if len(subj) == 0 or len(obj) == 0:
        return sentence, False

    # Check if the first word of subject and object is the same
    for s in subj:
        for o in obj:
            if s.text != o.text:
                # Swap the position of subject and object in the sentence
                sentence = sentence.replace(s.text, 'SUBJ_PLACEHOLDER')
                sentence = sentence.replace(o.text, s.text)
                sentence = sentence.replace('SUBJ_PLACEHOLDER', o.text)
                return sentence, True

    return sentence, False


# 获取聚类选项
def get_cluster_option(path, data_path):
    f = open(path, "r", encoding="utf-8")
    d = f.read().splitlines()
    dict_ = {}
    for line in d:
        cid, label = line.split(",")
        if label not in dict_.keys():
            dict_[label] = [cid]
        else:
            dict_[label].append(cid)

    f = open(data_path, "r", encoding="utf-8")
    d = json.load(f)['data']

    return_dict = {}
    for label in dict_.keys():
        cids = dict_[label]
        return_dict[label] = []
        for data in d:
            cid_ = str(data['cid'])
            if cid_ in cids:
                options = data['qas'][0]['options']
                return_dict[label].extend(options)

    return dict_, return_dict


def target_options(cid, cid_dict, data_dict):
    label_ = -1
    cid = str(cid)
    for label in cid_dict.keys():
        if cid in cid_dict[label]:
            label_ =  label

    option = random.choice(data_dict[label_])

    return  option


def write_json(data, des_json_path):
    data_dict = dict()
    data_dict['version'] = "acmrc-data"
    data_dict['data'] = data

    if os.path.exists(des_json_path):
        os.remove(des_json_path)

    with open(des_json_path, "w", encoding='utf8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=3, sort_keys=False)


def aug(data_list):

    label_list = ['A', 'B', 'C', 'D']
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    id2label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    count = 0
    cluster_path = '/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_train_cluster.txt'
    data_path = res_json_path
    cid_dict, data_dict = get_cluster_option(cluster_path, data_path)
    print("原始count: ", len(data_list))

    for item1 in data_list:
        q_type = item1['qas'][0]['question_type']
        # 如果是选正确的题型就跳过
        if q_type == 1:
            continue
        
        cid = item1['cid']

        item = copy.deepcopy(item1)


        options = item['qas'][0]['options']
        answer = item['qas'][0]['answer']
        answer_index = label2id[answer]

        idx_list = [0,1,2,3]
        idx_list.remove(answer_index)


        random_cluster_option_flag = False
        swap_subject_object_flag = False

        # 遍历选项
        for i in range(len(options)):
            # 如果是答案选项就跳过
            if i == answer_index:
                continue
            
            # 换主语、宾语
            if not swap_subject_object_flag:
                options[i],swap_subject_object_flag = swap_subject_object(options[i])
                if swap_subject_object_flag:
                    idx_list.remove(i)
                    continue

            if not random_cluster_option_flag:
                options[i] = target_options(cid,cid_dict, data_dict)
                random_cluster_option_flag = True
                idx_list.remove(i)
                continue


        # 两种增强方法都用了
        if not random_cluster_option_flag and not swap_subject_object_flag:
            continue


        #print(idx_list)

        item['qas'][0]['answer'] = id2label[idx_list[0]]
        item['qas'][0]['options'] = options
        item['qas'][0]['question_type'] = 1
        data_list.append(item)
        count += 1

    print("增强count: ", count)
    print("增强后count: ", len(data_list))

    return data_list


if __name__ == '__main__':

    set_ = 'train'

    res_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_{}.json".format(set_)
    des_json_path = "/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/data_aug_0605/acrc_{}.json".format(set_)
    # 读取
    data_list = json.load(open(res_json_path, 'r', encoding='UTF-8'))['data']

    # 数据增强
    data_list = aug(data_list)

    # 写入
    write_json(data_list, des_json_path)



