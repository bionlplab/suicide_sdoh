import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, bert, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes) 
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = self.classifier(output.pooler_output)
        return output