import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import log_loss
import json  # 添加json模块导入


class PATHS:
    train_path = None
    eval_path = None
    test_path = None
    all_path = "../data/final_all_data.json"
    prob_path = "../data/qwen_72b_instruct_awq_prob_results.csv"

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=128):
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of train_ratio, val_ratio, and test_ratio must be 1."

    # 创建一个组合标签：难度_答案
    df['strat_label'] = df['difficulty'] + '_' + df['answer']
    
    # Step 1: 先划分训练集和临时集
    train_data, temp_data = train_test_split(
        df, 
        test_size=(1 - train_ratio), 
        stratify=df['strat_label'],  # 同时按照难度和答案的分布进行划分
        random_state=random_state
    )

    # Step 2: 再划分验证集和测试集
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        stratify=temp_data['strat_label'], 
        random_state=random_state
    )
    
    # 删除临时的组合标签列
    train_data = train_data.drop('strat_label', axis=1)
    val_data = val_data.drop('strat_label', axis=1)
    test_data = test_data.drop('strat_label', axis=1)

    return train_data, val_data, test_data

def save_to_json(df, original_json_path, output_path):
    # 读取原始JSON文件
    with open(original_json_path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)  
    
    current_ids = set(df['id'].unique())
    
    new_data = []
    for item in data:
        if item['cid'] in current_ids:
            # 获取该id对应的difficulty和log_loss
            sample_info = df[df['id'] == item['cid']].iloc[0]
            item = item.copy()
            item['difficulty'] = sample_info['difficulty']
            item['log_loss'] = float(sample_info['log_loss'])
            new_data.append(item)
    
    # 使用json模块保存
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=2)

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)
    records = []
    for item in data:
        or_context = item['context']
        id = item['cid']
        source = item['source']
        for qa in item['qas']:
            question = qa['question']
            answer = qa['answer']
            options = qa['options']
            
            record = {
                "source": source,
                "question": question,
                "id": id,
                "or_text": or_context,
                "answer": answer,
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
            }
            records.append(record)

    df = pd.DataFrame(records)
    
    return df

data = get_dataframe(PATHS.all_path)

print(data.head())

# 1. 首先计算log_loss和难度标签
prob_results = pd.read_csv(PATHS.prob_path)
y_pred = prob_results.loc[:, ['A', 'B', 'C', 'D']].values
y_true = data['answer'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3}).values

# 计算每个样本的 log_loss
individual_log_loss = []
for i in range(len(y_true)):
    true_one_hot = np.zeros(4)
    true_one_hot[y_true[i]] = 1
    pred_probs = y_pred[i]
    sample_loss = -np.sum(true_one_hot * np.log(np.clip(pred_probs, 1e-15, 1.0)))
    individual_log_loss.append(sample_loss)

# 将log_loss添加到数据框中
data['log_loss'] = individual_log_loss

# 计算准确率
pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, pred_labels)
print(f"\n总体准确率: {accuracy:.4f}")

# 按选项计算准确率
for label in range(4):
    mask = y_true == label
    label_accuracy = accuracy_score(y_true[mask], pred_labels[mask])
    print(f"选项 {['A', 'B', 'C', 'D'][label]} 的准确率: {label_accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_true, pred_labels, target_names=['A', 'B', 'C', 'D']))

# 打印混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_true, pred_labels))

# 计算log_loss的分位点，按3:5:2划分
x1 = np.percentile(data['log_loss'], 30)  # 前30%为simple
x2 = np.percentile(data['log_loss'], 80)  # 接下来50%为medium，最后20%为hard
x3 = data['log_loss'].max()

# 添加难度标签
def get_difficulty(x):
    if x <= x1:
        return 'simple'
    elif x <= x2:
        return 'medium'
    else:
        return 'hard'

data['difficulty'] = data['log_loss'].apply(get_difficulty)

# 2. 然后再进行数据集划分
train_data, val_data, test_data = split_dataset(data)

# 打印各个集合中的难度分布
print("\n训练集难度分布:")
print(train_data['difficulty'].value_counts(normalize=True))
print("\n验证集难度分布:")
print(val_data['difficulty'].value_counts(normalize=True))
print("\n测试集难度分布:")
print(test_data['difficulty'].value_counts(normalize=True))
print(test_data.head())

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")
print(f"Answer distribution in training set:\n{train_data['answer'].value_counts(normalize=True)}")

save_to_json(train_data, PATHS.all_path, "../data/train_data.json")
save_to_json(val_data, PATHS.all_path, "../data/val_data.json")
save_to_json(test_data, PATHS.all_path, "../data/test_data.json")



# prob_results = pd.read_csv(PATHS.prob_path)
# choices = ["A","B","C","D"]
# pred_answers = []

# for idx, row in prob_results.iterrows():
#     idx = np.argmax(row[choices])
#     pred_answers.append(choices[idx])

# def evaluate(y_true, y_pred):
#     labels = ['A', 'B', 'C', 'D']
#     mapping = {'A': 0, 'B': 1, 'C':2, 'D': 3, 'none': 3}
#     def map_func(x):
#         return mapping.get(x, 2)
    
#     y_true = np.vectorize(map_func)(y_true)
#     y_pred = np.vectorize(map_func)(y_pred)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
#     print(f'Accuracy: {accuracy:.3f}')
    
#     # Generate accuracy report
#     unique_labels = set(y_true)  # Get unique labels
    
#     for label in unique_labels:
#         label_indices = [i for i in range(len(y_true)) 
#                          if y_true[i] == label]
#         label_y_true = [y_true[i] for i in label_indices]
#         label_y_pred = [y_pred[i] for i in label_indices]
#         accuracy = accuracy_score(label_y_true, label_y_pred)
#         print(f'Accuracy for label {label}: {accuracy:.3f}')
        
#     # Generate classification report
#     class_report = classification_report(y_true=y_true, y_pred=y_pred)
#     print('\nClassification Report:')
#     print(class_report)
    
#     # Generate confusion matrix
#     conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3])
#     print('\nConfusion Matrix:')
#     print(conf_matrix)

# 使用布尔索引来获取相应的预测结果
# simple_mask = (data['difficulty']=='hard')
# evaluate(data[simple_mask]['answer'], [pred_answers[i] for i in range(len(pred_answers)) if simple_mask.iloc[i]])

# 这部分在kaggle上使用4×L4完成

# Acc for qwen_72b_instruct_awq

# Accuracy: 0.681
# Accuracy for label 0: 0.774
# Accuracy for label 1: 0.775
# Accuracy for label 2: 0.644
# Accuracy for label 3: 0.537

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.55      0.77      0.64      1061
#            1       0.68      0.77      0.72      1118
#            2       0.80      0.64      0.71      1123
#            3       0.81      0.54      0.65      1115

#     accuracy                           0.68      4417
#    macro avg       0.71      0.68      0.68      4417
# weighted avg       0.71      0.68      0.68      4417


# Confusion Matrix:
# [[821 128  64  48]
#  [168 866  45  39]
#  [227 124 723  49]
#  [285 161  70 599]]