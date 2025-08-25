import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, 
                           classification_report, 
                           confusion_matrix)

class PATHS:
    test_path = "../../../data/test_data.json"
    pred_path = "predictions_bench.csv"

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
    
    # 计算总体准确率
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'总体准确率: {accuracy:.3f}')
    
    # 计算每个选项的准确率
    unique_labels = set(y_true)
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'选项 {labels[label]} 的准确率: {accuracy:.3f}')
    
    # 打印分类报告
    print('\n分类报告:')
    print(classification_report(y_true=y_true, y_pred=y_pred, 
                              target_names=labels))
    
    # 打印混淆矩阵
    print('\n混淆矩阵:')
    print(confusion_matrix(y_true=y_true, y_pred=y_pred, 
                         labels=[0, 1, 2, 3]))

if __name__ == "__main__":
    # 读取数据
    data = get_dataframe(PATHS.test_path)
    pred_df = pd.read_csv(PATHS.pred_path)
    
    # 确保数据对齐
    data = data.sort_values('id').reset_index(drop=True)
    pred_df = pred_df.sort_values('id').reset_index(drop=True)
    
    # 评估模型性能
    evaluate(data['answer'], pred_df['pred'])
    
    # 计算不同难度的准确率
    data['correct'] = (data['answer'] == pred_df['pred']).astype(int)
    print("\n按难度级别的准确率:")
    print(data.groupby('difficulty')['correct'].mean())
