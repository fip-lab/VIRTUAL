import json
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, 
                           classification_report, 
                           confusion_matrix)
import requests
from tqdm import tqdm
import time

def get_access_token():
    """获取百度API access token"""
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=GDz5oJI6vojfo1Cdx1Y3T8yc&client_secret=GSLTr4a9ktznDOTBtIHk3Coh4pe00Yww"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class PATHS:
    test_path = "../../data/test_data.json"
    pred_path = "./o1-mini_results.csv"

def clean_prediction(text):
    """使用 ERNIE-4.0-8K 清理预测结果中的额外文本，只保留 A/B/C/D"""
    system_message = "你是一个助手。请从下面的文本中提取出预测的选项(A/B/C/D)。只需要输出一个字母，不需要任何解释。如果找不到明确的选项，输出D。"
    user_query = f"预测文本：{text}"
    
    max_retries = 3
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
            
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": system_message + "\n\n" + user_query
                    }
                ]
            })
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.request("POST", url, headers=headers, data=payload)
            response_json = response.json()
            cleaned_pred = response_json['result'].strip().upper()
            
            if cleaned_pred not in ['A', 'B', 'C', 'D']:
                cleaned_pred = 'D'
            return cleaned_pred
            
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(retry_delay * (retry + 1))
            else:
                print("Max retries reached. Using default answer.")
                return 'D'
    
    return 'D'

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)
    records = []
    for item in data:
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
                "difficulty": difficulty,
                "answer": answer,
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    return df

def evaluate(y_true, y_pred):
    labels = ['A', 'B', 'C', 'D']
    mapping = {'A': 0, 'B': 1, 'C':2, 'D': 3, 'none': 3}
    def map_func(x):
        return mapping.get(x, 2)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'总体准确率: {accuracy:.3f}')
    
    unique_labels = set(y_true)
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'选项 {labels[label]} 的准确率: {accuracy:.3f}')
    
    print('\n分类报告:')
    print(classification_report(y_true=y_true, y_pred=y_pred))
    
    print('\n混淆矩阵:')
    print(confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3]))

if __name__ == "__main__":
    # 读取数据
    data = get_dataframe(PATHS.test_path)
    pred_df = pd.read_csv(PATHS.pred_path)
    
    # 清理预测结果
    print("正在清理预测结果...")
    pred_df['cleaned_pred'] = pred_df['pred'].apply(clean_prediction)
    
    # 确保数据对齐
    data = data.sort_values('id').reset_index(drop=True)
    pred_df = pred_df.sort_values('id').reset_index(drop=True)
    
    # 评估模型性能
    evaluate(data['answer'], pred_df['cleaned_pred'])
    
    # 计算不同难度的准确率
    data['correct'] = (data['answer'] == pred_df['cleaned_pred']).astype(int)
    print("\n按难度级别的准确率:")
    print(data.groupby('difficulty')['correct'].mean())