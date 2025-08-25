from jiayan import PMIEntropyLexiconConstructor
import pandas as pd
import numpy as np

class PATHS:
    train_path = "../data/train_data.json"
    eval_path = "../data/val_data.json"
    test_path = "../data/test_data.json"

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
                "or_text": or_context,
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
or_texts = test_df['or_text'].tolist()

with open('or_texts.txt', 'w', encoding='utf-8') as f:
    for text in or_texts:
        f.write(text + '\n')  

constructor = PMIEntropyLexiconConstructor()
lexicon = constructor.construct_lexicon('or_texts.txt')
constructor.save(lexicon, 'train_thesaurus.csv')