#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 10_fold_dataset_split.py
@Author  : huanggj
@Time    : 2023/6/20 16:47
"""
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold,train_test_split
import pandas as pd
import numpy as np
import json
import os

def k_fold_dataset_split(data_path, dest_path, tokenizer_path):
    # 使用 json.load 读取 JSON 文件
    # with open(data_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    # 将 JSON 数据转换为 Pandas 数据框架
    #df = pd.json_normalize(data)

    # 加载 JSON 文件
    #df = pd.read_json(data_path, lines=True)
    df = pd.read_json(data_path)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 计算token长度
    df['token_length'] = df['context'].apply(lambda x: len(tokenizer.encode(x, truncation=True)))

    # 根据文章长度范围，创建一个新的 'length_category' 列
    bins = [0, 512, 1024, 1536, np.inf]
    labels = ['0-512', '512-1024', '1024-1536', '1536+']
    df['length_category'] = pd.cut(df['token_length'], bins=bins, labels=labels)

    # 创建一个新的列来表示每个样本的类别和长度范围的组合
    df['type_length'] = df['qas'].apply(lambda x: x[0]['question_type']).astype(str) + "_" + df['length_category'].astype(str)

    # 过滤掉样本数少于10的组(如果该样本组小于10)
    group_counts = df['type_length'].value_counts()
    df = df[df['type_length'].isin(group_counts[group_counts >= 10].index)]

    # 使用 StratifiedKFold 进行10次划分
    skf = StratifiedKFold(n_splits=10)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['type_length'])):
        print(f'fold : {fold}')
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # 再将训练+验证集划分为训练集 和 验证集，比例为8:1
        train_df, val_df = train_test_split(train_val_df, test_size=1 / 9, stratify=train_val_df['type_length'],
                                            random_state=42)

        #a = train_df.to_dict(orient='records')


        # 将每次划分的结果保存为JSON文件
        #train_df.to_json(f'{dest_path}/fold_{fold}/train.json', orient='records', force_ascii=False, lines=True)
        #val_df.to_json(f'{dest_path}/fold_{fold}/dev.json', orient='records', force_ascii=False, lines=True)
        #test_df.to_json(f'{dest_path}/fold_{fold}/test.json', orient='records', force_ascii=False, lines=True)

        # 判断目录是否存在
        directory_path = f'{dest_path}/fold_{fold}'
        if not os.path.exists(directory_path):
            # 不存在则创建目录
            os.makedirs(directory_path)
            print(f"目录 '{directory_path}' 创建成功！")
        else:
            print(f"目录 '{directory_path}' 已经存在。")

        with open(f'{dest_path}/fold_{fold}/train.json', 'w',
                  encoding='utf-8') as merged_file:
            json.dump(train_df.to_dict(orient='records'), merged_file, ensure_ascii=False, indent=2)

        with open(f'{dest_path}/fold_{fold}/dev.json', 'w',
                  encoding='utf-8') as merged_file:
            json.dump(val_df.to_dict(orient='records'), merged_file, ensure_ascii=False, indent=2)

        with open(f'{dest_path}/fold_{fold}/test.json', 'w',
                  encoding='utf-8') as merged_file:
            json.dump(test_df.to_dict(orient='records'), merged_file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    data_path = '/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/acrc_all.json'
    dest_path = '/disk2/huanggj/ACMRC_EXP_V202306/3_DATASET/ACRC/4_inputs/10_Fold'
    tokenizer_path = '/disk2/huanggj/ACMRC_EXP_V202306/pretrain/BERT'
    k_fold_dataset_split(data_path, dest_path, tokenizer_path)
