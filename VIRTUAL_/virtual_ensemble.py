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

top_m = 1
top_n = 3

class PATHS:
    test_path = '../data/test_data.json'
    model_name = 'qwen-plus-0806'
    dict_path = '../acrc_word_segmentation/high_freq_annotations.csv'
    test_with_evidence_path = f'../data/evidence_extraction_{top_m}_{top_n}.csv'
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
        personal_image = item['personal_image']
        reign_title = item['reign_title']
        dynasty = item['dynasty-tyqw']
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
                "personal_image": personal_image,
                "answer": answer,
                "reign_title": reign_title,
                "dynasty": dynasty,
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
            }
            records.append(record)

    df = pd.DataFrame(records)
    
    return df

data = get_dataframe(PATHS.test_path)

def chat(messages):
    response = dashscope.Generation.call(
        model=PATHS.model_name,
        messages=messages,
        result_format='message')
    
    if response.status_code == HTTPStatus.OK:
        messages.append({'role': response.output.choices[0]['message']['role'],
                       'content': response.output.choices[0]['message']['content']})
    else:
        print('Error in getting answer:', response.message)
        messages = messages[:-1]
    return response,messages

def rotate_options(options, rotation):
    """Rotate options by n positions"""
    n = len(options)
    rotated = options[rotation:] + options[:rotation]
    return rotated

def map_answer_back(answer, rotation):
    options = ['A', 'B', 'C', 'D']
    n = len(options)
    original_pos = (options.index(answer) + rotation) % n
    return options[original_pos]

results = []
all_responses = []
SS = "#"*25 + "\n"

dict_train = pd.read_csv(PATHS.dict_path) 
test_df = pd.read_csv(PATHS.test_with_evidence_path)
test_df['personal_image'] = data['personal_image']
test_df['reign_title'] = data['reign_title']
test_df['dynasty'] = data['dynasty']

dashscope.api_key="sk-***" # 请替换为你的api_key

# 修改这里的循环方式
for i in tqdm(range(len(test_df))):
    question_responses = {
        'question_id': i + 1,
        'stu1_answer': '',
        'stu2_answer': '',
        'stu3_answer': '',
        'final_answer': ''
    }
    votes = []
    
    for student_id in range(3):
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
        options = [
            test_df.loc[i, 'A'],
            test_df.loc[i, 'B'],
            test_df.loc[i, 'C'],
            test_df.loc[i, 'D']
        ]
        rotated_options = rotate_options(options, student_id)
        
        sys_prompt = "根据以下文言文原文和译文，选择一个符合题意的选项。只返回选项对应的字母A, B, C或D，不需要解释。"
        prompt = f"文章:\n{test_df.loc[i, 'or_text']}\n\n{SS}题目: {q}\nA:{rotated_options[0]}\nB:{rotated_options[1]}\nC:{rotated_options[2]}\nD:{rotated_options[3]}\n\n{SS}\n\n{SS}证据:\n{text}\n{test_df.loc[i, 'sent_tr_context']}\n{test_df.loc[i, 'amr_tr_context']}\n{test_df.loc[i, 'amr_context']}\n\n{SS}注释:\n{annotations}\n\n{SS}\n\n只返回选项对应的字母A, B, C或D，不需要解释。"
        formatted_sample = sys_prompt + "\n\n" + prompt
        
        ans_mes = {'role': 'user', 'content': formatted_sample}
        rawmessages.append(ans_mes)
        res, mes = chat(rawmessages)
        
        if res.status_code == HTTPStatus.OK:
            rotated_answer = res.output.choices[0].message.content
            answer = map_answer_back(rotated_answer, student_id) if student_id > 0 else rotated_answer
            votes.append(answer)
            question_responses[f'stu{student_id+1}_answer'] = answer
        else:
            answer = 'A'
            votes.append(answer)
            question_responses[f'stu{student_id+1}_answer'] = answer
        
        time.sleep(1)  # 添加延时避免请求过快
    
    final_answer = max(set(votes), key=votes.count)
    question_responses['final_answer'] = final_answer
    results.append(final_answer)
    all_responses.append(question_responses)

# 保存结果
responses_df = pd.DataFrame(all_responses)
responses_df.to_csv(f'virtual_{top_m}_{top_n}.csv', index=False)

def evaluate(y_true, y_pred):
    labels = ['A', 'B', 'C', 'D']
    mapping = {'A': 0, 'B': 1, 'C':2, 'D': 3, 'none': 3}
    def map_func(x):
        return mapping.get(x, 2)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report with zero_division parameter
    class_report = classification_report(
        y_true=y_true, 
        y_pred=y_pred, 
        zero_division=0,
        labels=[0, 1, 2, 3],
        target_names=['A', 'B', 'C', 'D']
    )
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3])
    print('\nConfusion Matrix:')
    print(conf_matrix)
    
    # Add difficulty level evaluation
    test_df['correct'] = (test_df['answer'] == results).astype(int)
    difficulty_accuracy = test_df.groupby('difficulty')['correct'].agg(['mean', 'count'])
    print("\nAccuracy by Difficulty Level:")
    print(difficulty_accuracy)

evaluate(test_df['answer'], results)