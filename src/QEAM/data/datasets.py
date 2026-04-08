import os
import pickle
from typing import List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


CACHE_DIR = "./cached_datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Callable, max_seq_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(self.texts[idx])

        if hasattr(tokens, "ids"): 
            tokens = tokens.ids
        elif isinstance(tokens, dict) and "input_ids" in tokens:
            tokens = tokens["input_ids"]
        tokens = list(tokens)

        tokens = tokens[:self.max_seq_len]
        padding_len = self.max_seq_len - len(tokens)
        tokens = tokens + [0] * padding_len  

        input_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # Return a tuple, NOT a dict
        return input_tensor, label_tensor


def get_tokenizer(max_seq_len: int) -> Callable:
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _tokenize(text: str):
        # return numeric token IDs
        enc = hf_tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)
        if hasattr(enc, "ids"):
            return enc.ids
        return enc["input_ids"]

    return _tokenize



def download_or_load_agnews(max_seq_len: int = 128, subset: Optional[int] = None) -> DatasetDict:
    cache_path = os.path.join(CACHE_DIR, f"agnews_{max_seq_len}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached AG_NEWS from {cache_path}")
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Downloading AG_NEWS from HuggingFace...")
        dataset = load_dataset("ag_news")
        if subset is not None:
            dataset["train"] = dataset["train"].select(range(subset))
            dataset["test"] = dataset["test"].select(range(subset))
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved dataset to cache: {cache_path}")
    return dataset



def load_agnews(tokenizer: Optional[Callable] = None,
                max_seq_len: int = 128,
                batch_size: int = 64,
                subset: Optional[int] = None) -> Tuple[DataLoader, DataLoader, int]:
    if tokenizer is None:
        tokenizer = get_tokenizer(max_seq_len)

    dataset = download_or_load_agnews(max_seq_len=max_seq_len, subset=subset)

   
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    num_classes = len(set(train_labels + test_labels))

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes