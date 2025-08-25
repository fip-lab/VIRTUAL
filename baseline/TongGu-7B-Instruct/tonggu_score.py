import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PATHS:
    train_path = None
    eval_path = None
    test_path = None
    all_path = "../../data/final_all_data.json"
    model_path = "./TongGu-7B-Instruct"

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

model = AutoModelForCausalLM.from_pretrained(PATHS.model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path, trust_remote_code=True)

results = []
for i in tqdm(range(len(data))):
    text = data.loc[i, 'or_text']
    q = data.loc[i, 'question']
    options = [data.loc[i, 'A'], data.loc[i, 'B'], data.loc[i, 'C'], data.loc[i, 'D']]
    system_message = "你是文学汉语方面的专家。你的目标阅读一篇文章后回答问题并选择符合题意的选项，只需要输出选项对应的字符(A/B/C/D)，不需要做解释。"
    user_query = f"### 问题：\n{q}\n\n### 选项：\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}\n\n### 文章：\n{text}"
    prompt = f"{system_message}\n<用户> {user_query}\n<通古> "
    inputs = tokenizer(prompt, return_tensors='pt')
    generate_ids = model.generate(
        inputs.input_ids.cuda(), 
        max_new_tokens=1
    )
    generate_text = tokenizer.batch_decode(
        generate_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0][len(prompt):]
    results.append({"id": data.loc[i, 'id'], "pred": generate_text})
    print(generate_text)

results = pd.DataFrame(results)
results.to_csv("tonggu_results.csv", index=False)