#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : find_hyper_params.py
@Author  : huanggj
@Time    : 2023/6/25 17:47
"""
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
import torch
import optuna

# 数据准备
# 这里我们仅仅创建一些随机数据作为例子，你需要替换成真正的数据
train_encodings = {
    'input_ids': torch.tensor(np.random.randint(0, 1000, size=(100, 128)), dtype=torch.long),
    'attention_mask': torch.tensor(np.random.randint(0, 2, size=(100, 128)), dtype=torch.long)
}
train_labels = torch.tensor(np.random.randint(0, 1000, size=(100, 128)), dtype=torch.long)

val_encodings = {
    'input_ids': torch.tensor(np.random.randint(0, 1000, size=(20, 128)), dtype=torch.long),
    'attention_mask': torch.tensor(np.random.randint(0, 2, size=(20, 128)), dtype=torch.long)
}
val_labels = torch.tensor(np.random.randint(0, 1000, size=(20, 128)), dtype=torch.long)

class QuestionAnsweringDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = QuestionAnsweringDataset(train_encodings, train_labels)
val_dataset = QuestionAnsweringDataset(val_encodings, val_labels)

# 定义优化目标
def objective(trial):
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32])

    args = TrainingArguments(
        "test-squad",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_result = trainer.evaluate()

    return eval_result["eval_accuracy"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# 打印最优超参数
print(study.best_params)
