#fixed seed to get similar score
from transformers import set_seed
set_seed(42)

import os
import gc
import time
import warnings
import math

import pandas as pd
import numpy as np

import pandas as pd
import polars as pl

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 45) * 60

from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_pth = '/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1'

llm = LLM(
    llm_model_pth,
    max_model_len=32768,
    trust_remote_code=True,     
    tensor_parallel_size=4,      
    gpu_memory_utilization=0.85,  # 降低内存使用率
)

tokenizer = llm.get_tokenizer()

class PATHS:
    all_path = "/kaggle/input/virtual-v1/final_all_data.json"
    llm_model_pth = '/kaggle/input/qwq-32b-preview/transformers/default/1'

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

from transformers import LogitsProcessor
from typing import Any, Dict, List
choices = ["A","B","C","D"]

KEEP = []
for x in choices:
    c = tokenizer.encode(x,add_special_tokens=False)[0]
    KEEP.append(c)
print(f"Force predictions to be tokens {KEEP} which are {choices}.")

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_ids = KEEP
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores

sys_prompt = """你是文学汉语方面的专家。你的目标阅读一篇文章后回答问题并选择符合题意的选项，只需要输出选项对应的字符(A/B/C/D)，不需要做解释。"""

all_prompts = []
for i,row in data.iterrows():
    text = data.loc[i, 'or_text']
    q = data.loc[i, 'question']
    options = [data.loc[i, 'A'], data.loc[i, 'B'], data.loc[i, 'C'], data.loc[i, 'D']]
    prompt = f"### 文章：\n{text}\n\n### 问题：\n{q}\n\n### 选项：\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}"
    
    
    formatted_sample = sys_prompt + "\n\n" + prompt
    
    all_prompts.append( formatted_sample )

%%time
import vllm
from time import time

# 设置批次大小
BATCH_SIZE = 32  # 可以根据实际情况调整
all_results = []

start = time()
logits_processors = [DigitLogitsProcessor(tokenizer)]

# 分批处理
for i in range(0, len(all_prompts), BATCH_SIZE):
    # 清理内存
    clean_memory()
    
    batch_prompts = all_prompts[i:i + BATCH_SIZE]
    responses = llm.generate(
        batch_prompts,
        vllm.SamplingParams(
            n=1,
            top_p=0.9,
            temperature=0,
            seed=777,
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=logits_processors,
            logprobs=5
        ),
        use_tqdm=True
    )
    
    # 处理每个批次的结果
    for response in responses:
        try:
            x = response.outputs[0].logprobs[0]
            logprobs = []
            for k in KEEP:
                if k in x:
                    logprobs.append(math.exp(x[k].logprob))
                else:
                    logprobs.append(0)
                    print(f"bad logits {i}")
            logprobs = np.array(logprobs)
            logprobs /= logprobs.sum()
            all_results.append(logprobs)
        except:
            all_results.append(np.array([1/4., 1/4., 1/4., 1/4.]))
            errors += 1

    # 打印进度
    print(f"Processed {min(i + BATCH_SIZE, len(all_prompts))}/{len(all_prompts)} samples")

end = time()
elapsed = (end-start)/60. #minutes
print(f"Inference of {len(data)} samples took {elapsed} minutes!")
print(f"There were {errors} inference errors out of {len(all_prompts)} inferences")

# 将所有结果堆叠成一个数组
results = np.vstack(all_results)
results = pd.DataFrame(results, columns=["A", "B", "C", "D"])
results['id'] = data['id']
results.to_csv("qwen_72b_instruct_awq_prob_results.csv", index=False)
