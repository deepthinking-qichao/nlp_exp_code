from torch.utils.data import Dataset, DataLoader
from sample import Sample, SampleList
import pandas as pd
from collections import defaultdict
import torch

class MyDataset(Dataset):
    def __init__(self, tokenizer, dataset_path):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(dataset_path)
        labels = self.df["label"]
        reviews = self.df["review"]
        self.anno_list = []
        for label, review in zip(labels, reviews):
            self.anno_list.append({"label": label, "review": review})
        print(f"{dataset_path} dataset length: {len(self.anno_list)}")

        self.max_length = 32

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):
        anno_dict = self.anno_list[idx]
        # self.df.iloc[index, :].tolist()
        return Sample(anno_dict)

    def collate_fn(self, data):
        data = SampleList(data)
        data.input_texts = self.tokenizer(data.review, padding=True, max_length=self.max_length,
        truncation=True, return_tensors="pt")
        return data
