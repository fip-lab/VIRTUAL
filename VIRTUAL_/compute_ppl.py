import dashscope
from tqdm import tqdm
from http import HTTPStatus
import pandas as pd
import numpy as np
import os
from datasets import load_from_disk, Dataset
import jieba
from sklearn.metrics import (accuracy_score, 
                           classification_report, 
                           confusion_matrix)
import time
from collections import Counter, defaultdict
import math

top_m = 1
top_n = 3

class PATHS:
    test_path = '../data/test_data.json'
    train_path = '../data/train_data.json'
    model_name = 'qwen-plus-0806'
    dict_path = '../acrc_word_segmentation/high_freq_annotations.csv'
    test_with_evidence_path = f'../data/evidence_extraction_{top_m}_{top_n}.csv'
    ensemble_result_path = f'../result/virtual_{top_m}_{top_n}_one_shot.csv'
    output_path = '../result'

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)
    records = []
    for item in data:
        or_context = item['tran_context_tyqw']
        id = item['cid']
        source = item['source']
        difficulty = item['difficulty']
        for qa in item['qas']:
            question = qa['question']
            answer = qa['answer']
            options = qa['options']
            
            record = {
                "source": source,
                "question": question,
                "id": id,
                "or_text": or_context,
                "difficulty": difficulty,
                "answer": answer,
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df

def build_ngram_model(texts, n=2):
    """构建n-gram语言模型"""
    # 初始化计数器
    ngram_counts = defaultdict(Counter)
    context_counts = defaultdict(int)
    
    # 对每个文本进行处理
    for text in texts:
        # 分词
        words = list(jieba.cut(text))
        # 添加开始和结束标记
        words = ['<s>'] * (n-1) + words + ['</s>']
        
        # 统计n-gram
        for i in range(len(words)-n+1):
            ngram = tuple(words[i:i+n])
            context = tuple(words[i:i+n-1])
            ngram_counts[context][ngram[-1]] += 1
            context_counts[context] += 1
    
    return ngram_counts, context_counts

def calculate_ppl(text, ngram_counts, context_counts, n=2, alpha=0.1):
    """计算文本的困惑度"""
    words = list(jieba.cut(text))
    words = ['<s>'] * (n-1) + words + ['</s>']
    
    log_prob = 0
    count = 0
    vocab_size = len(set(word for context in ngram_counts for word in ngram_counts[context]))
    
    # 计算每个词的概率
    for i in range(len(words)-n+1):
        word = words[i+n-1]
        context = tuple(words[i:i+n-1])
        
        # 使用加法平滑
        numerator = ngram_counts[context][word] + alpha
        denominator = context_counts[context] + alpha * vocab_size
        
        # 处理分母为0的情况
        if denominator == 0:
            probability = 1.0 / vocab_size  # 使用均匀分布
        else:
            probability = numerator / denominator
        
        log_prob += math.log2(probability)
        count += 1
    
    # 处理count为0的情况
    if count == 0:
        return float('inf')
    
    # 计算困惑度
    ppl = 2 ** (-log_prob/count)
    return ppl

def main():
    # 加载训练集和测试集
    train_df = get_dataframe(PATHS.train_path)
    test_df = get_dataframe(PATHS.test_path)
    
    # 构建语言模型
    print("构建语言模型...")
    train_texts = train_df['or_text'].tolist()
    ngram_counts, context_counts = build_ngram_model(train_texts, n=2)
    
    # 计算测试集困惑度
    print("计算测试集困惑度...")
    ppls = []
    for text in tqdm(test_df['or_text']):
        ppl = calculate_ppl(text, ngram_counts, context_counts)
        ppls.append(ppl)
    
    # 添加困惑度到测试集
    test_df['ppl'] = ppls
    
    # 按困惑度分析准确率
    print("\n困惑度统计:")
    print(f"平均困惑度: {np.mean(ppls):.2f}")
    print(f"中位数困惑度: {np.median(ppls):.2f}")
    print(f"最大困惑度: {np.max(ppls):.2f}")
    print(f"最小困惑度: {np.min(ppls):.2f}")
    
    # 保存结果
    test_df.to_csv(f'{PATHS.output_path}/test_with_ppl.csv', index=False)
    
    preds = pd.read_csv(PATHS.ensemble_result_path)
    perplexity_acc = pd.concat([preds['final_answer'],test_df[['ppl', 'answer']]],axis=1).rename(columns={'final_answer':'pred'})
    # 首先计算预测是否正确
    perplexity_acc['is_correct'] = (perplexity_acc['pred'] == perplexity_acc['answer']).astype(int)

    # 使用pandas的cut函数将perplexity分成几个区间
    perplexity_acc['perplexity_bin'] = pd.qcut(perplexity_acc['ppl'], q=10)

    # 计算每个区间的平均准确率
    accuracy_by_perplexity = perplexity_acc.groupby('perplexity_bin')['is_correct'].mean().reset_index()

    # 绘制图表
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracy_by_perplexity)), accuracy_by_perplexity['is_correct'], marker='o')
    plt.xticks(range(len(accuracy_by_perplexity)), 
            [f'{x.left:.0f}-{x.right:.0f}' for x in accuracy_by_perplexity['perplexity_bin']], 
            rotation=45)
    plt.xlabel('Perplexity Range')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy vs Perplexity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{PATHS.output_path}/perplexity_acc.png')
    plt.close() 

if __name__ == "__main__":
    main()