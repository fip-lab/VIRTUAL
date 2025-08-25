import requests
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
import time

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=GDz5***&client_secret=GSLT****" # 请替换为你的client_id和client_secret
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class PATHS:
    train_path = "../../data/train_data.json"
    eval_path = "../../data/val_data.json"
    test_path = "../../data/test_data.json"
    all_path = "../../data/final_all_data.json"

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)
    records = []
    for item in data:
        or_context = item['context']
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

data = get_dataframe(PATHS.test_path)

results = []
for i in tqdm(range(len(data))):
    text = data.loc[i, 'or_text']
    q = data.loc[i, 'question']
    options = [data.loc[i, 'A'], data.loc[i, 'B'], data.loc[i, 'C'], data.loc[i, 'D']]
    user_query = f"你是文学汉语方面的专家。你的目标阅读一篇文章后回答问题并选择符合题意的选项，只需要输出选项对应的字符(A/B/C/D)，不需要做解释。\n### 文章：\n{text}\n\n### 问题：\n{q}\n\n### 选项：\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}"
    
    max_retries = 3
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
            
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            })
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.request("POST", url, headers=headers, data=payload)
            response_json = response.json()
            response_text = response_json['result']
            break
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(retry_delay * (retry + 1))
            else:
                print("Max retries reached. Moving to next item.")
                response_text = 'A'  # 默认答案
    
    results.append({"id": data.loc[i, 'id'], "pred": response_text})
    time.sleep(1)

results = pd.DataFrame(results)
results.to_csv("ernie_4_results.csv", index=False)

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

evaluate(data['answer'], results['pred'])

data['correct'] = (data['answer'] == results['pred']).astype(int)
print(data.groupby('difficulty')['correct'].mean())

