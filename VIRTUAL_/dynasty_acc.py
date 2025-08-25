from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

top_m = 1
top_n = 3

class PATHS:
    test_path = '../data/test_data.json'
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
                "dynasty": dynasty,
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
result = pd.read_csv(PATHS.ensemble_result_path)
data['pred'] = result['final_answer']

# 定义朝代到语言时期的映射
period_mapping = {
    '夏商周': '上古汉语',
    '春秋战国': '上古汉语',
    '秦朝': '上古汉语',
    '汉朝': '上古汉语',
    '三国': '中古汉语',
    '晋朝': '中古汉语',
    '南北朝': '中古汉语',
    '隋朝': '中古汉语',
    '唐朝': '中古汉语',
    '五代十国': '中古汉语',
    '宋朝': '近古汉语',
    '辽金元': '近古汉语',
    '明朝': '近古汉语',
    '清朝': '近古汉语'
}

# 添加语言时期列
data['period'] = data['dynasty'].map(period_mapping)

with open(f'{PATHS.output_path}/dynasty_acc.txt', 'w', encoding='utf-8') as file:
    # 计算整体准确率
    overall_accuracy = accuracy_score(data['answer'], data['pred'])
    file.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
    
    # 按朝代统计
    dynasty_accuracy = data.groupby('dynasty')[['answer', 'pred']].apply(lambda x: accuracy_score(x['answer'], x['pred']))
    dynasty_count = data.groupby('dynasty').size()
    dynasty_result = pd.DataFrame({
        'accuracy': dynasty_accuracy,
        'sample_count': dynasty_count
    })
    file.write("Dynasty-wise Accuracy:\n")
    file.write(str(dynasty_result))
    file.write("\n\n")
    
    # 按语言时期统计
    period_accuracy = data.groupby('period')[['answer', 'pred']].apply(lambda x: accuracy_score(x['answer'], x['pred']))
    period_count = data.groupby('period').size()
    period_result = pd.DataFrame({
        'accuracy': period_accuracy,
        'sample_count': period_count
    })
    file.write("Period-wise Accuracy:\n")
    file.write(str(period_result))
