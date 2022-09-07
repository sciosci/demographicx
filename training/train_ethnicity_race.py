import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
import transformers
from tqdm.notebook import tqdm

census10 = pd.read_csv('./data/census_2010.csv')
census10 = census10.replace('(S)', 0)
LABELS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC']
L = []

for i in range(len(census10)):
    n = [float(j) for j in census10[['pctwhite', 'pctblack', 'pctapi', 'pcthispanic']].values[i]]
    L.append(LABELS[np.argmax(n)])census10['label'] = L

census10 = census10.drop_duplicates().dropna()

census10.drop_duplicates().dropna()['label'].value_counts()

census10_sampled = pd.concat([census10[census10['label'] == 'BLACK'].sample(n = 7500), 
                            census10[census10['label'] == 'ASIAN'].sample(n = 7500), 
                            census10[census10['label'] == 'HISPANIC'].sample(n = 7500),
                           census10[census10['label'] == 'WHITE'].sample(n = 7500)])[['name', 'label']].reset_index(drop = True)

def get_name_pair(s):
    s = str(s).lower()
    return(s, ' '.join(str(s)).replace('  ', ' ').replace('  ', ' '))

possible_labels = census10_sampled['label'].unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

census10_sampled['label'] = census10_sampled['label'].replace(label_dict)

training_split, testing_split = train_test_split(census10_sampled, test_size = 10000, random_state = 42)

torch.set_grad_enabled(True)

# Store the model we want to use
MODEL_NAME = "bert-base-cased"

# We need to create the model and tokenizer
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encoded_train = tokenizer.batch_encode_plus(training_split['name'].apply(get_name_pair).values.tolist(),
                                            return_attention_mask=True,
                                            padding=True,
                                            return_tensors='pt')

encoded_val = tokenizer.batch_encode_plus(testing_split['name'].apply(get_name_pair).values.tolist(),
                                            return_attention_mask=True,
                                            padding=True,
                                            return_tensors='pt')

input_ids_train = encoded_train['input_ids']
attention_masks_train = encoded_train['attention_mask']
labels_train = torch.tensor(training_split.label.values)

input_ids_val = encoded_val['input_ids']
attention_masks_val = encoded_val['attention_mask']
labels_val = torch.tensor(testing_split.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=4,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 50

dataloader_train = DataLoader(dataset_train, 
                sampler=RandomSampler(dataset_train), 
                batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                  #sampler=SequentialSampler(dataset_val), 
                  batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
          lr=1e-5, 
          eps=1e-8)
                  
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                        num_warmup_steps=0,
                        num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.cuda() for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):
    model.cuda()
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.cuda() for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    # torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

val_loss, predictions, true_vals = evaluate(dataloader_validation)

f1_score(true_vals, np.argmax(predictions, axis=1), average = None)