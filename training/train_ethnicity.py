import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
# Creating Dataloader
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification

from utils import f1_score_func


# The Function to Create both Word-level and Character-level Features for BERT
def get_name_pair(s):
    return s, ' '.join(str(s)).replace('  ', ' ').replace('  ', ' ')


# Loading datasets
race_sampled_train = pd.read_csv('./data/ethnic_mixed_train.csv')
race_sampled_val = pd.read_csv('./data/ethnic_mixed_val.csv')
race_sampled_train['name'] = race_sampled_train['name'].astype(str).apply(
    lambda x: x.lower())
race_sampled_val['name'] = race_sampled_val['name'].astype(str).apply(lambda x: x.lower())

# Check All Classes
possible_labels = race_sampled_train['label'].unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

race_sampled_train['label'] = race_sampled_train['label'].replace(label_dict)
race_sampled_val['label'] = race_sampled_val['label'].replace(label_dict)


torch.set_grad_enabled(True)

# Store the model we want to use
MODEL_NAME = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize Data
encoded_train = tokenizer.batch_encode_plus(
    race_sampled_train['name'].apply(get_name_pair).values.tolist(),
    return_attention_mask=True,
    padding=True,
    return_tensors='pt')

encoded_val = tokenizer.batch_encode_plus(
    race_sampled_val['name'].apply(get_name_pair).values.tolist(),
    return_attention_mask=True,
    padding=True,
    return_tensors='pt')

input_ids_train = encoded_train['input_ids']
attention_masks_train = encoded_train['attention_mask']
labels_train = torch.tensor(race_sampled_train.label.values)

input_ids_val = encoded_val['input_ids']
attention_masks_val = encoded_val['attention_mask']
labels_val = torch.tensor(race_sampled_val.label.values)

# Create datasets
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Loading Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

batch_size = 50

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   # sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup

# Optimizer
optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)

epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(
                                                dataloader_train) * epochs)

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Function to Evaluate Model Performance
def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.cuda() for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
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

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


# Training Model
for epoch in tqdm(range(1, epochs + 1)):
    model.cuda()
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False,
                        disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.cuda() for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(
            {'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    # torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals, )
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

# Model Performance
val_loss, predictions, true_vals = evaluate(dataloader_validation)
f1_score(true_vals, np.argmax(predictions, axis=1), average=None)

# Save Model
model.save_pretrained('../demographics_ethnicity/')
print('Model saved!')