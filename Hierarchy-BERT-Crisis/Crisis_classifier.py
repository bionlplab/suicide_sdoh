import random
import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, BertModel
from torch.optim import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from data import Dataset
from model import BertClassifier
from utils import common_args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

NUM_CLASS = '3_class'
LABELS = {'3_class': ['Social community context', 'Behavior and lifestyle', 'Economic stability'],
          '5_class': ['Safety concern', 'Social support', 'Psychosocial factor', 'Behavior and lifestyle', 'Economic stability'],
          '7_class': ['Safety concern', 'Interpersonal support', 'Mental health problem', 'Adverse life experience', 'Stress', 'Substance use', 'Financial distress']}

DATASETS = {'3_class': {'train': '/data/3class/train.csv',
                        'test': '/data/3class/test.csv'},
            '5_class': {'train': '/data/hierarchy/train_5class.csv',
                        'test': '/data/hierarchy/test_5class.csv'},
            '7_class': {'train': '/data/hierarchy/train.csv',
                        'test': '/data/hierarchy/test.csv'}}

MODELS = {'BERT': "bert-base-uncased", 
          'PubmedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
          'BioBERT': "dmis-lab/biobert-base-cased-v1.2"}

def calculate_metrics(pred, target):
    return {'precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'accuracy': accuracy_score(y_true=target, y_pred=pred)
            }

def evaluate(args, test_dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    model_result, targets = [], []
    pred_scores = []

    for step, batch in enumerate(test_dataloader):
        report_input_ids, report_masks, labels = batch[0]['input_ids'].to(device), batch[0]['attention_mask'].to(device), batch[1].to(device)
            
        inputs = {'input_ids': report_input_ids.squeeze(1),
                  'attention_mask': report_masks.squeeze(1)}
        
        preds = model(**inputs)
        loss = criterion(preds, labels.float())
        total_loss = total_loss + loss.item()
        
        pred_scores.extend(preds.detach().cpu().numpy())
        preds = torch.sigmoid(preds)
        preds = torch.round(preds)
        model_result.extend(preds.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

    result = calculate_metrics(np.array(model_result), np.array(targets))
    avg_loss = total_loss / len(test_dataloader)
    
    with open(args.output_dir, 'a') as file:
        file.write('Epoch: {}, Evaluation loss: {}\n'.format(epoch, avg_loss))
        file.write('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}\n'.format(result['precision'], result['recall'], result['f1']))
        file.write('Accuracy: {:.3f} \n'.format(result['accuracy']))
        file.write('{}\n'.format(classification_report(np.array(targets), np.array(model_result), target_names=LABELS[NUM_CLASS], digits=4)))
    
    return avg_loss, result['accuracy']

def train(args, train_dataloader, test_dataloader, model, tokenizer, device):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.)).to(device)
    best_accuracy = 0
    
    for epoch in tqdm(range(args.num_train_epochs)):
        print('Training, Epoch: {}'.format(epoch))
        total_loss = 0
        model.train()
        
        for step, batch in tqdm(enumerate(train_dataloader)):
            report_input_ids, report_masks, labels = batch[0]['input_ids'].to(device), batch[0]['attention_mask'].to(device), batch[1].to(device)
            
            inputs = {'input_ids': report_input_ids.squeeze(1),
                      'attention_mask': report_masks.squeeze(1)}
            
            preds = model(**inputs)
            loss = criterion(preds, labels.float())
            total_loss = total_loss + loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)
        with open(args.output_dir, 'a') as file:
            file.write('**'*20)
            file.write('Epoch: {}, Training loss: {}.\n'.format(epoch, avg_loss))
        
        loss, acc = evaluate(args, test_dataloader, model, criterion, device, epoch)
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, './models_3/epoch_{}_accuracy_{:.3f}.pt'.format(epoch, acc))

if __name__ == '__main__':
    parser = common_args()
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda', args.gpu_device)
    print('Device:', device)
   
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.bert_model])
    model = BertClassifier(bert=MODELS[args.bert_model], n_classes=int(NUM_CLASS.split('_')[0]))
  
    model.to(device)
    print('*'*40)
    print('Model initialized.')
    
    train_df, test_df = pd.read_csv(DATASETS[NUM_CLASS]['train']), pd.read_csv(DATASETS[NUM_CLASS]['test'])
    print('*'*40)
    print('Dataset dataframe loaded. Training set size {}, test set size {}.'.format(len(train_df), len(test_df)))
    
    train_dataset = Dataset(train_df)
    test_dataset = Dataset(test_df)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False)
    print('*'*40)
    print('Dataset loader loaded.')
    
    train(args, train_dataloader, test_dataloader, model, tokenizer, device)
    
    
    
    
    
    
    
    