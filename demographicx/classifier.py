import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer

__all__ = [
    'GenderEstimator',
    'EthnicityEstimator'
]


def get_name_pair(s):
    return s, ' '.join(str(s)).replace('  ', ' ').replace('  ', ' ')


def one_batch_name_predictor(encoder, model, name):
    encoded = encoder([get_name_pair(str(name).lower())],
                      return_attention_mask=True,
                      padding=True, return_tensors='pt')
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'],
                            torch.tensor([0]))
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))

    inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
              'labels': batch[2], }
    output = softmax(model(**inputs)[1].detach().tolist()[0])
    return output


class GenderEstimator:
    def __init__(self, name_or_path="liamliang/demographics_gender"):
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def predict(self, name):
        output = one_batch_name_predictor(self.tokenizer, self.model, name)
        res = {'male': output[0], 'unknown': output[1], 'female': output[2]}
        return res


class EthnicityEstimator:
    def __init__(self, name_or_path="liamliang/demographics_race"):
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def predict(self, name):
        output = one_batch_name_predictor(self.tokenizer, self.model, name)
        res = {'black': output[0], 'hispanic': output[1], 'white': output[2],
               'asian': output[3]}
        return res
