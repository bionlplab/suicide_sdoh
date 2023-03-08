import torch
import numpy as np
from transformers import BertTokenizer, AutoTokenizer

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

class InputExample(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):        
        self.texts = []
        self.labels = []
        
        for idx, row in df.iterrows():
            CME, LE = str(row['NarrativeCME']), str(row['NarrativeLE'])
            # whichever report is shorter in length will be put first
            if len(CME) >= len(LE):
                report = LE + CME
            else:
                report = CME + LE
                            
            # tokenize the input texts
            self.texts.append(tokenizer(report, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"))
            label = torch.as_tensor(row[-7:].tolist())
            self.labels.append(label)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    