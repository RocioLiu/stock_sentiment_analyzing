from typing import List, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src import config


@dataclass
class SentimentDataset(Dataset):
    texts: List[List[str]]
    labels: List[int]
    vocab_to_id: Dict[str, int]
    cls_id: int
    sep_id: int
    max_len: int
    pad_on_right: bool = True

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"SentimentDataset with {self.__len__()} items."

    def __str__(self):
        return f"SentimentDataset with {self.__len__()} items."

    def get_max_len(self):
        return max([len(text) for text in self.texts])

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        input_ids = [self.vocab_to_id[word] for word in text]

        # max_len = self.get_max_len()
        max_len = self.max_len
        input_ids = input_ids[:max_len - 2]
        input_ids = [self.cls_id] + input_ids + [self.sep_id] # [CLS] + input_ids + [SEP]

        # attention masks
        # only the real token with value 1 are attends to, and the other
        # positions pads with 0.
        attention_mask = [1] * len(input_ids)

        # Segment the two sequences
        # token_type_ids = [0] * len(input_ids)

        ## Pad the inputs on the right hand side with the longest sample
        padding_len = max_len - len(input_ids)

        if self.pad_on_right:
            input_ids = input_ids + [0] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            # token_type_ids = token_type_ids + [0] * padding_len
        else:
            input_ids = [0] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            # token_type_ids = [0] * padding_len + token_type_ids

        assert len(input_ids) == max_len
        assert len(attention_mask) == max_len
        # assert len(token_type_ids) == max_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

