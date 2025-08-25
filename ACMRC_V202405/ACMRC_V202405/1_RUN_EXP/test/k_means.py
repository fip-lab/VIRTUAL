#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : k_means.py
@Author  : huanggj
@Time    : 2023/5/31 21:30
"""


import jieba
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import json
import os

f = open("/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_train.json", "r", encoding="utf-8")

data = json.load(f)['data']


# 假设你的字典是这样的
articles_dict = {}

for d in data:
    pid = d['cid']
    passage = d['context']
    articles_dict[pid] = passage



# 将字典的值转化为列表
articles = list(articles_dict.values())

# 使用jieba分词
seg_articles = [list(jieba.cut(article)) for article in articles]

# 将分词结果连接为字符串
seg_articles_str = [' '.join(seg) for seg in seg_articles]

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(seg_articles_str)

# 在不同的聚类数量下计算肘部值
k_values = range(2, 10)  # 聚类数量的范围
inertias = []  # 用于存储肘部值

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 绘制肘部法则曲线
plt.plot(k_values, inertias, 'bx-')
plt.xlabel('聚类数量')
plt.ylabel('肘部值')
plt.title('肘部法则')
plt.show()

# 使用轮廓系数选择最佳的聚类数量
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

best_k = k_values[np.argmax(silhouette_scores)]

print("最佳的聚类数量是：", best_k)

# 执行K均值聚类
kmeans = KMeans(n_clusters=best_k)
kmeans.fit(X)

res = []
# 打印每个文章ID对应的聚类结果
for article_id, article in articles_dict.items():
    index = articles.index(article)
    cluster_label = kmeans.labels_[index]
    print(f"文章 {article_id} 属于聚类 {cluster_label+1}")
    res.append(str(article_id) + "," + str(cluster_label+1) + '\n')

f = open("/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_train_cluster.txt", "w", encoding="utf-8")
f.writelines(res)

