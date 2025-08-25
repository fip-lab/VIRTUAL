from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.data.data_collator import DataCollatorForTokenClassification, _torch_collate_batch
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
from torch.nn import ConstantPad2d
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch_relations(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `2d examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    # All relations are nxn list , n might be different for different example, we need to pad the 2d tensor

    examples = [torch.tensor(i) for i in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)


    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    

    # -100 denotes no compute loss
    result = examples[0].new_full([len(examples),max_length, max_length], -100).long()
    for i, example in enumerate(examples):
        result[i, :example.shape[0],:example.shape[0]] = example
    return result


class DualCollatorForSentenceAndRelation:
    def __init__(self,sentence_collator) -> None:
        self.sentence_collator=sentence_collator
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:



        batch = self.sentence_collator({"input_ids":[i["input_ids"] for i in features]})
        batch['relation'] = collate_batch_relations([i["relation"] for i in features],self.sentence_collator.tokenizer)
        batch['is_node_start'] = {"is_node_start":[i["is_node_start"] for i in features]}

        # token_ids = [{"input_ids":i["token_ids"]} for i in features]
        # for i in features:
        #     del i['token_ids']

        # batch = self.relation_collator(features)
        # batch['token_ids'] = sentence_batch['input_ids']
        
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}



    


