import dashscope, os, re
from time import time
import pandas as pd
import numpy as np
import faiss
import ast
from tqdm import tqdm
import torch
from jiayan import load_lm, CharHMMTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk, Dataset

class PATHS:
    test_path = '../../data/test_data.json'
    dict_path = '../acrc_word_segmentation/high_freq_annotations.csv'
    amr_path = '../Two-Stage-CAMRP/test_A/amr_clauses.txt'
    output_path = '../data/test_df_with_evidence.csv'

def get_dataframe(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)
    records = []
    for item in data:
        or_context = item['context']
        tr_context = item['tran_context_tyqw']
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
                "tr_text": tr_context,
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

test_df = get_dataframe(PATHS.test_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = SentenceTransformer('ethanyt/guwenbert-base', device=device)

# Create Faiss Index
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

def generate_faiss_index_with_split(text, split_chars='，。'):
    index = None
    sentences = re.split(f'(?<=[{re.escape(split_chars)}])', text)
    sentences = [s for s in sentences if s.strip()]  # Remove empty strings
    
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device, show_progress_bar=False).cpu().numpy()
    
    if index is None:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
    index.add(embeddings)
        
    return sentences, index

def semantic_search_with_threshold(query, sentences, index, top_k=1, threshold=0.8):
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False, device=device).cpu().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, top_k)
    results = []
    for j, i in enumerate(I[0]):
        similarity = 1 / (1 + D[0][j])
        
        if similarity >= threshold:  # Only consider results with similarity above the threshold
            results.append({'text': sentences[i], 'similarity': similarity, 'distance': D[0][j]})
    return results


# load amr data
def load_amr_data(amr_path):
    # 10_A_1303	['吕陶提出了对策和三点看法', '吕陶的理财言论不让陛下迷惑', '吕陶不嫌弃老成的谋略', '王安石刚刚掌权', '王安石在蜀州担任通判']
    # 10_B_1304	['吕陶担任侍御史殿中的职务', '吕陶上书罢黜了一些人', '这些人包括辜负先帝的人', '这些人还包括与哲宗和蔡确等有关的人', '吕陶议论了官员关于税收的弊端', '官员也被罢黜了', '被罢黜的官员包括河渡等']
    # 10_C_1305	['朱光庭弹劾了苏轼和吕陶', '朱光庭非议了先辈的功业', '朱光庭违背了原则', '朱光庭应该秉承台和谏官的原则', '朱光庭应该公正地奏议政事', '朱光庭担心从此可能出现朋党']
    # 10_D_1306	['哲宗深知吕陶护佑了自己', '吕陶在过去一直护佑了哲宗', '哲宗还是担心会有人的言论迷惑圣', '哲宗希望哲宗能够明察秋毫', '哲宗明察秋毫是为了维持国家安稳的局面']
    amr_data = []
    current_group = []
    current_id = None

    with open(amr_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 分割ID和内容
                idx, content = line.strip().split('\t', 1)
                cid = idx.split('_')[0]
                
                # 替换中文引号为英文引号
                content = content.replace(''', "'").replace(''', "'")
                content = content.replace('，', ',')
                
                # 处理分组
                if current_id is not None and cid != current_id:
                    if current_group:
                        amr_data.append((int(current_id), current_group))
                    current_group = []
                
                current_id = cid
                # 使用ast.literal_eval更安全地解析字符
                parsed_content = ast.literal_eval(content)
                current_group.extend(parsed_content)

            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {e}")
                continue

        # 处理最后一组
        if current_group:
            amr_data.append((int(current_id), current_group))

    return dict(amr_data)

amr_dict = load_amr_data(PATHS.amr_path)

# Search Faiss Index
for i in tqdm(range(len(or_texts)), desc="Processing rows"):
    or_amr_mappings = []  # New list for AMR evidence in original text
    tr_amr_mappings = []  # New list for AMR evidence in translated text
    or_sent_mappings = []  # New list for sentence evidence in original text
    tr_sent_mappings = []  # New list for sentence evidence in translated text
    option = [test_df.loc[i,'A'], test_df.loc[i,'B'], test_df.loc[i,'C'], test_df.loc[i,'D']]
    seen_sentences = set()

    # AMR evidence processing
    current_amr = amr_dict.get(test_df.loc[i,'id'], [])
    if current_amr:
        or_sentences, or_index = generate_faiss_index_with_split(or_texts[i])
        tr_sentences, tr_index = generate_faiss_index_with_split(tr_texts[i])
        
        for amr_sent in current_amr:
            or_results = semantic_search_with_threshold(amr_sent, or_sentences, or_index, top_k=1, threshold=0.3)
            for result in or_results:
                if result['text'] not in seen_sentences:
                    or_amr_mappings.append(f"- {result['text']}")
                    seen_sentences.add(result['text'])
            
            tr_results = semantic_search_with_threshold(amr_sent, tr_sentences, tr_index, top_k=1, threshold=0.3)
            for result in tr_results:
                if result['text'] not in seen_sentences:
                    tr_amr_mappings.append(f"- {result['text']}")
                    seen_sentences.add(result['text'])

    # Full sentence evidence processing
    seen_sentences.clear()  # Reset seen sentences for full sentence search
    for query in option:
        # Original text search
        sentences, index = generate_faiss_index(or_texts[i])
        semantic_results = semantic_search(query, top_k=2)
        
        for result in semantic_results:
            sentence = result['text']
            if sentence not in seen_sentences:
                or_sent_mappings.append(f"- {sentence}")
                seen_sentences.add(sentence)
        
        # Translated text search
        sentences, index = generate_faiss_index(tr_texts[i])
        semantic_results = semantic_search(query, top_k=1)
        
        for result in semantic_results:
            sentence = result['text']
            if sentence not in seen_sentences:
                tr_sent_mappings.append(f"- {sentence}")
                seen_sentences.add(sentence)

    # Store results in separate columns
    test_df.loc[i, "amr_context"] = "\n\n".join(or_amr_mappings)
    test_df.loc[i, "amr_tr_context"] = "\n\n".join(tr_amr_mappings)
    test_df.loc[i, "sent_context"] = "\n\n".join(or_sent_mappings)
    test_df.loc[i, "sent_tr_context"] = "\n\n".join(tr_sent_mappings)

test_df.to_csv(PATHS.output_path, index=False)

print("Done!")
