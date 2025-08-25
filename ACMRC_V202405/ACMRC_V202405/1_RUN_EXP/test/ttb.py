import spacy

nlp = spacy.load("zh_core_web_sm")


def create_adverb_list(choices):
    """
    Create a list of adverbs based on the choice options.

    Parameters:
    choices (List[str]): The list of choice options.

    Returns:
    List[str]: The list of adverbs.
    """
    adverbs = []
    for choice in choices:
        doc = nlp(choice)
        for token in doc:
            if token.pos_ == 'ADV':  # 'ADV' stands for adverb in spacy
                adverbs.append(token.text)
    return adverbs  # remove duplicates


# Your choice options
choices = ["这可能是个好主意", "这绝对是个好主意", "这可能不是个好主意", "这绝对不是个好主意"]

# Create the adverb list
# adverbs = create_adverb_list(choices)
# adverbs = set(adverbs)
# print(adverbs)

import json
import os

f = open("/disk2/huanggj/ACMRC_EXPERIMENT/dataset/ACRC/base_data/acrc_train.json", "r", encoding="utf-8")

data = json.load(f)['data']

print(len(data))

a = []
for d in data:
    options = d['qas'][0]['options']
    adverbs = create_adverb_list(options)
    b = set(adverbs)
    print(b)
    a = a + adverbs

a = set(a)
print(a)


