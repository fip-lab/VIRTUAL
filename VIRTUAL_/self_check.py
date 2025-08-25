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
import json

top_m = 2
top_n = 1

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

def self_check(messages, answer, question, options):
    check_prompt = f"""请检查我之前选择的答案是否正确。

    问题：{question}

    选项：
    A: {options[0]}
    B: {options[1]}
    C: {options[2]}
    D: {options[3]}

    我选择的答案是：{answer}

    请仔细思考后回答"确认"或"修改为X"（X为A、B、C、D中的一个）。只需回答这两种格式之一，不需要解释。"""

    check_mes = {'role': 'user', 'content': check_prompt}
    messages.append(check_mes)
    res, mes = chat(messages)
    
    if res.status_code == HTTPStatus.OK:
        check_result = res.output.choices[0].message.content
        if check_result.startswith("修改为"):
            return check_result[-1]  # 返回新答案
        else:
            return answer  # 保持原答案
    else:
        return answer  # 如果API调用失败，保持原答案

def get_question_type(messages, question, options):
    sys_prompt = "请判断这道题目属于'原文理解题'还是'概括理解题'。只需回答'原文理解题'或'概括理解题'，不需要解释。"
    prompt = f"题目: {question}\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}"
    
    type_mes = {'role': 'user', 'content': sys_prompt + "\n\n" + prompt}
    messages.append(type_mes)
    res, mes = chat(messages)
    
    if res.status_code == HTTPStatus.OK:
        question_type = res.output.choices[0].message.content
    else:
        question_type = "原文理解题"  # 默认类型
    
    return question_type

def get_example_prompt(question_type):
    with open('../data/example.json', 'r', encoding='utf-8') as file:
        examples = json.load(file)
    
    example = examples[question_type]
    example_prompt = f"""这是一个{question_type}的例子：

文章：{example['文章']}

题目：{example['问题']}

选项：
A: {example['选项']['A']}
B: {example['选项']['B']}
C: {example['选项']['C']}
D: {example['选项']['D']}

正确答案：{example['答案']}

解释：{example['解释']}

现在请你回答下面的题目：
"""
    return example_prompt

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
        'stu1_checked': '',
        'stu2_checked': '',
        'stu3_checked': '',
        'final_answer': ''
    }
    votes = []
    checked_votes = []
    
    # 获取当前题目的问题和选项
    q = test_df.loc[i, 'question']
    options = [
        test_df.loc[i, 'A'],
        test_df.loc[i, 'B'],
        test_df.loc[i, 'C'],
        test_df.loc[i, 'D']
    ]
    
    for student_id in range(3):
        rawmessages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        text = test_df.loc[i, 'sent_context']
        
        # 获取题型，传入问题和选项
        question_type = get_question_type(rawmessages.copy(), q, options)
        
        # 获取相应的例子
        example = get_example_prompt(question_type)
        
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
        
        rotated_options = rotate_options(options, student_id)
        
        sys_prompt = "根据以下文言文原文和译文，选择一个符合题意的选项。只返回选项对应的字母A, B, C或D，不需要解释。"
        prompt = f"{example}\n\n文章:\n{test_df.loc[i, 'or_text']}\n\n{SS}题目: {q}\nA:{rotated_options[0]}\nB:{rotated_options[1]}\nC:{rotated_options[2]}\nD:{rotated_options[3]}\n\n{SS}\n\n{SS}证据:\n{text}\n{test_df.loc[i, 'sent_tr_context']}\n{test_df.loc[i, 'amr_tr_context']}\n{test_df.loc[i, 'amr_context']}\n\n{SS}注释:\n{annotations}\n\n{SS}年号:\n{test_df.loc[i, 'reign_title']}\n\n{SS}朝代:\n{test_df.loc[i, 'dynasty']}\n\n只返回选项对应的字母A, B, C或D，不需要解释。"
        
        formatted_sample = sys_prompt + "\n\n" + prompt
        ans_mes = {'role': 'user', 'content': formatted_sample}
        rawmessages.append(ans_mes)
        res, mes = chat(rawmessages)
        
        if res.status_code == HTTPStatus.OK:
            rotated_answer = res.output.choices[0].message.content
            answer = map_answer_back(rotated_answer, student_id) if student_id > 0 else rotated_answer
            
            # 添加self-check
            checked_answer = self_check(rawmessages.copy(), answer, q, options)
            checked_votes.append(checked_answer)
            question_responses[f'stu{student_id+1}_checked'] = checked_answer
            
            votes.append(answer)
            question_responses[f'stu{student_id+1}_answer'] = answer
        else:
            answer = 'A'
            votes.append(answer)
            checked_votes.append(answer)
            question_responses[f'stu{student_id+1}_answer'] = answer
            question_responses[f'stu{student_id+1}_checked'] = answer
        
        time.sleep(1)
    
    final_answer = max(set(votes), key=votes.count)
    question_responses['final_answer'] = final_answer
    results.append(final_answer)
    all_responses.append(question_responses)

# 保存结果
responses_df = pd.DataFrame(all_responses)
responses_df.to_csv(f'virtual_{top_m}_{top_n}_self_check.csv', index=False)

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