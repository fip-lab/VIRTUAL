import dashscope
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, 
                           classification_report, 
                           confusion_matrix)
from http import HTTPStatus

dashscope.api_key="sk-ead2b1fb26e54500b6c4393f4ea03e8a"

class PATHS:
    test_path = "../../data/test_data.json"
    pred_path = "./ernie_4_results.csv"
    model_name = "qwen-plus"

def clean_prediction(text):
    """使用 Qwen-plus 清理预测结果中的额外文本，只保留 A/B/C/D"""
    system_message = "你是一个助手。请从下面的文本中提取出预测的选项(A/B/C/D)。只需要输出一个字母，不需要任何解释。如果找不到明确的选项，输出D。"
    user_query = f"预测文本：{text}"
    
    try:
        response = dashscope.Generation.call(
            model=PATHS.model_name,
            prompt=f"{system_message}\n\n{user_query}",
            top_p=0.9
        )
        
        if response.status_code == HTTPStatus.OK:
            cleaned_pred = response.output.text.strip().upper()
            if cleaned_pred not in ['A', 'B', 'C', 'D']:
                cleaned_pred = 'D'
            return cleaned_pred
        else:
            print(f"API Error: {response.code} - {response.message}")
            return 'D'
    except Exception as e:
        print(f"Error cleaning prediction: {str(e)}")
        return 'D'

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