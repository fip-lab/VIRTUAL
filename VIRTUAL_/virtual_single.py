import dashscope
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (accuracy_score, 
                           classification_report, 
                           confusion_matrix)
from http import HTTPStatus
import time
import os
import jieba

dashscope.api_key="sk-***" # 请替换为你的api_key

class PATHS:
    test_path = '../data/test_df_with_evidence.csv'
    model_name = 'qwen-plus-0806'
    dict_path = '../acrc_word_segmentation/high_freq_annotations.csv'
    test_with_evidence_path = '../data/evidence_extraction_2_1.csv'
    output_path = '../result'

def chat(messages):
    response = dashscope.Generation.call(
        model=PATHS.model_name,
        messages=messages,
        result_format='message')
    
    if response.status_code == HTTPStatus.OK:
        messages.append({'role': response.output.choices[0]['message']['role'],
                       'content': response.output.choices[0]['message']['content']})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        messages = messages[:-1]
    return response,messages

test_df = pd.read_csv(PATHS.test_with_evidence_path)
dict_train = pd.read_csv(PATHS.dict_path)
results = []
SS = "#"*25 + "\n"

for i in tqdm(range(len(test_df))):
    # Combine reading and answering into one round
    rawmessages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    text = test_df.loc[i, 'sent_context']
    
    # Use jieba for tokenization
    tokens = jieba.lcut(text)
    punctuation = set(['-', '，', '。', '"', '"', '：', '？','?' '！', ''', ''', '；'])
    cleaned_tokens = []
    seen_tokens = set()
    for token in tokens:
        if token not in seen_tokens and token not in punctuation:
            cleaned_tokens.append(token)
            seen_tokens.add(token)

    # Add annotations
    annotation_dict = dict(zip(dict_train['word'], dict_train['annotation']))
    annotations = [f"{token}: {annotation_dict[token]}" for token in cleaned_tokens if token in annotation_dict]
    annotations = '\n--------\n'.join(annotations)
    
    q = test_df.loc[i, 'question']
    options = [test_df.loc[i, 'A'], test_df.loc[i, 'B'], test_df.loc[i, 'C'], test_df.loc[i, 'D']]
    
    sys_prompt = "根据以下文言文原文和译文，选择一个符合题意的选项。只返回选项对应的字母A, B, C或D，不需要解释。"
    prompt = f"文章:\n{test_df.loc[i, 'or_text']}\n\n{SS}题目: {q}\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}\n\n{SS}\n\n{SS}证据:\n{text}\n{test_df.loc[i, 'sent_tr_context']}\n{test_df.loc[i, 'amr_tr_context']}\n{test_df.loc[i, 'amr_context']}\n\n{SS}注释:\n{annotations}\n\n只返回选项对应的字母A, B, C或D，不需要解释。"
    formatted_sample = sys_prompt + "\n\n" + prompt
    
    ans_mes = {'role': 'user', 'content': formatted_sample}
    rawmessages.append(ans_mes)
    res, mes = chat(rawmessages)
    
    if res.status_code == HTTPStatus.OK:
        response_text = res.output.choices[0].message.content
    else:
        print('Error in getting answer:', res.message)
        response_text = 'A'
    
    results.append({"id": test_df.loc[i, 'id'], "pred": response_text})
    time.sleep(1)

results = pd.DataFrame(results)
results.to_csv(os.path.join(PATHS.output_path,'virtul_single.csv'), index=False)

def evaluate(y_true, y_pred):
    labels = ['A', 'B', 'C', 'D']
    mapping = {'A': 0, 'B': 1, 'C':2, 'D': 3, 'none': 3}
    def map_func(x):
        return mapping.get(x, 2)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    unique_labels = set(y_true)
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3])
    print('\nConfusion Matrix:')
    print(conf_matrix)

evaluate(test_df['answer'], results['pred'])

test_df['correct'] = (test_df['answer'] == results['pred']).astype(int)
print(test_df.groupby('difficulty')['correct'].mean())