import dashscope, os, re
from time import time
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_from_disk, Dataset

class PATHS:
    test_path = '/kaggle/input/trans-qw/acrc_test.json'
    model_name = 'qwen-plus-0806'
    stus_path = '/kaggle/input/trans-qw/student_responses.csv'
    test_with_evidence_path = '/kaggle/input/trans-qw/test_df_with_evidence.csv'
    dict_path = '/kaggle/input/acrc-dict/high_freq_annotations.csv'
    jiayan_path = '/kaggle/input/jiayan/jiayan.klm'

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)['data']
    records = []
    for item in data:
        if item.get('tran_context_tyqw') != None:
            tr_context = item.get('tran_context_tyqw')
        else:
            tr_context = item['tran_context_glm']
        or_context = item['context']
        dynasty_tyqw = item['dynasty-tyqw']
        for qa in item['qas']:
            question = qa['question']
            answer = qa['answer']
            options = qa['options']
            question_type = qa['question_type']
            
            record = {
                "tr_text": tr_context,
                "or_text": or_context,
                "question": question,
                "answer": answer,
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
                "question_type": question_type,
                "dynasty": dynasty_tyqw
#                 "question": question
            }
            records.append(record)

    df = pd.DataFrame(records)
    df['question'] = df['question_type'].map(qmap)
    
    return df

qmap = {0 : '以下对文章内容理解[错误]的一项是:', 1:'以下对文章内容理解[正确]的一项是:'}
test_df = get_dataframe(PATHS.test_path)

stus = pd.read_csv(PATHS.stus_path)
total_sum = 0

for answer, row in zip(test_df['answer'], stus[['stu1_answer','stu2_answer','stu3_answer']].values):
    if answer in row:
        total_sum += 1

print("Acc:", total_sum/399)

if PATHS.test_with_evidence_path is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('DMetaSoul/Dmeta-embedding', device=device)

def generate_faiss_index(text):
    index = None
    sentences = re.split(r'(?<=。)', text)
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device, show_progress_bar=False).cpu().numpy()
    
    if index is None:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
    index.add(embeddings)
        
    return sentences, index

or_texts = test_df['or_text'].tolist()
tr_texts = test_df['tr_text'].tolist()

def semantic_search(query, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False, device=device).cpu().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, top_k)  # Perform the search
    results = [{'index': i, 'text': sentences[i], 'distance': D[0][j]} for j, i in enumerate(I[0])]
    return results

def keyword_search(keyword, top_k=1):
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = [text for text in texts if pattern.search(text)]
    return matches[:top_k]

print("Semantic Searching")
if PATHS.test_with_evidence_path is None:
    for i in tqdm(range(len(test_df)), desc="Processing rows"):
        or_sem = []
        tr_sem = []
        option = ["A", "B", "C", "D"]
        seen_sentences = set()  # Set to track unique sentences
    
        for o in option:
            query = test_df.loc[i, o]
            sentences, index = generate_faiss_index(or_texts[i])
            semantic_results = semantic_search(query, top_k=2)
    
            # Collect unique sentences
            unique_context = []
            for result in semantic_results:
                sentence = result['text']
                if sentence not in seen_sentences:
                    unique_context.append(sentence)
                    seen_sentences.add(sentence)
    
            # Combine unique contexts into a formatted string
            or_context = "-" + "\n-".join(unique_context)
            or_sem.append(or_context)
            
            sentences, index = generate_faiss_index(tr_texts[i])
            semantic_results = semantic_search(query, top_k=1)
    
            # Collect unique sentences
            unique_context = []
            for result in semantic_results:
                sentence = result['text']
                if sentence not in seen_sentences:
                    unique_context.append(sentence)
                    seen_sentences.add(sentence)
    
            # Combine unique contexts into a formatted string
            tr_context = "-" + "\n-".join(unique_context)
            tr_sem.append(tr_context)
    
        # Assign the unique combined context back to the DataFrame
        test_df.loc[i, "context"] = "\n-".join(or_sem)
        test_df.loc[i, "tr_context"] = "\n-".join(tr_sem)
    test_df.to_csv('test_df_with_evidence_ec.csv', index=False)
else:
    test_df = pd.read_csv(PATHS.test_with_evidence_path)
print("Done!")

sys_prompt = "根据以上对话中阅读的文言文原文和译文，选择一个符合题意的选项，可以利用当前对话中的相关证据进行判断，但是请注意证据不一定有用，必须结合你自己的知识和文章进行判断。只返回选项对应的字母A, B, C或D，不需要解释。"
# sys_prompt = "请根据下面题目和选项，阅读文章，选择符合题意的一个选项，只返回该选项对应的字母A,B,C或者D，不需要解释。"
SS = "#"*25 + "\n"

all_prompts = []
for index, row in test_df.iterrows():
    example = row["context"]
    a = row['question']
    b = 'A:' + row['A'] + '\nB:' + row['B'] + '\nC:' + row['C'] + '\nD:' + row['D']
    c = row['tr_context']

    prompt = f"{SS}题目: "+row['question'] +"\n"+b+ f"\n\n{SS}证据:" + example + '\n' + c +"\n\n" 

    formatted_sample = sys_prompt + "\n\n" + prompt

    all_prompts.append( formatted_sample )

print(all_prompts[0])

from http import HTTPStatus
from tqdm import tqdm
dashscope.api_key="sk-ead2b1fb26e54500b6c4393f4ea03e8a"

from jiayan import load_lm, CharHMMTokenizer

dict_train = pd.read_csv(PATHS.dict_path) 

lm = load_lm(PATHS.jiayan_path) 
tokenizer = CharHMMTokenizer(lm)

# Tokenize the text
text = test_df.loc[0, 'context']
tokens = list(tokenizer.tokenize(text))

# Define punctuation marks to remove
punctuation = set(['-', '，', '。', '“', '”', '：', '？','?' '！', '‘', '’', '；'])

# Remove duplicates and punctuation
cleaned_tokens = []
seen_tokens = set()

for token in tokens:
    if token not in seen_tokens and token not in punctuation:
        cleaned_tokens.append(token)
        seen_tokens.add(token)

annotation_dict = dict(zip(dict_train['word'], dict_train['annotation']))

annotations = [f"{token}: {annotation_dict[token]}" for token in cleaned_tokens if token in annotation_dict]

annotations = '\n--------\n'.join(annotations)

def chat(messages):
    response = dashscope.Generation.call(
        model=PATHS.model_name,
        messages=messages,
        result_format='message')
    
    if response.status_code == HTTPStatus.OK:
        messages.append({'role': response.output.choices[0]['message']['role'],
                         'content': response.output.choices[0]['message']['content']})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
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

results = []
for i, prompt_text in enumerate(tqdm(all_prompts, desc="Generating responses")):
    response_text = 'A'
    
    example = test_df.loc[i, 'context']
    a = test_df.loc[i, 'question']
    b = 'A:' + test_df.loc[i, 'A'] + '\nB:' + test_df.loc[i, 'B'] + '\nC:' + test_df.loc[i, 'C'] + '\nD:' + test_df.loc[i, 'D']
    c = test_df.loc[i, 'tr_context']
    
    # Get student answers and explanations
    stu_answers = f"""
        学生作答情况：
        学生1选择了{stus.loc[i, 'stu1_answer']}，解释：{stus.loc[i, 'stu1_explain']}
        学生2选择了{stus.loc[i, 'stu2_answer']}，解释：{stus.loc[i, 'stu2_explain']}
        学生3选择了{stus.loc[i, 'stu3_answer']}，解释：{stus.loc[i, 'stu3_explain']}
        """
    
    #FIRST ROUND
    rawmessages = [{'role': 'system', 'content': 'You are a helpful assistant acting as a teacher.'}]
    read_text_mes = {'role': 'user', 'content': f"请阅读文言文译文，完成任务，任务我会在接下来的对话中给你，如果你阅读完了，请只回复阅读完毕。\n\n{SS}\n文言文译文:\n{test_df.loc[i, 'or_text']}"}
    rawmessages.append(read_text_mes)
    _, mes = chat(rawmessages)

    text = test_df.loc[i, 'context']
    tokens = list(tokenizer.tokenize(text))

    punctuation = set(['-', '，', '。', '“', '”', '：', '？','?' '！', '‘', '’', '；'])

    cleaned_tokens = []
    seen_tokens = set()

    for token in tokens:
        if token not in seen_tokens and token not in punctuation:
            cleaned_tokens.append(token)
            seen_tokens.add(token)

    annotation_dict = dict(zip(dict_train['word'], dict_train['annotation']))

    annotations = [f"{token}: {annotation_dict[token]}" for token in cleaned_tokens if token in annotation_dict]
    annotations = '\n--------\n'.join(annotations)
    

    #FINAL ROUND
    sys_prompt = """作为一位老师，请根据文言文，证据，注释和学生的答案，选择一个最符合题意的答案。注意：
        1. 学生的答案有参考价值，但不能过度依赖，可能全部都是错误的
        2. 少数服从多数原则有一定参考价值，但也不是绝对正确
        3. 只需返回选项字母A、B、C或D，不需要做任何解释。"""
    
    prompt = f"{SS}题目: "+a +"\n"+b+ f"\n\n{SS}证据:" + example + '\n' + c + f"\n\n{SS}注释:\n" + annotations + f"\n\n{SS}学生答案:\n" + stu_answers + "\n\n"
    formatted_sample = sys_prompt + "\n\n" + prompt
    
    ans_mes = {'role': 'user', 'content': formatted_sample}
    mes.append(ans_mes)
    res, mes = chat(mes)
    
    
    #GET THE ANSWER
    if res.status_code == HTTPStatus.OK:
        response_text = res.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            res.request_id, res.status_code,
            res.code, res.message
        ))
        
    results.append(response_text)

from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

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
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3])
    print('\nConfusion Matrix:')
    print(conf_matrix)

evaluate(test_df['answer'], results)