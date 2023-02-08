import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd

__all__ = [
    'GenderEstimator',
    'EthnicityEstimator'
]


def get_name_pair(s):
    """Creates space separated words and space separated character in a list

    Parameter
    ---------
    s : string
        the string of the name to make prediction
    
    Returns
    -------
    out : list
          list includes both space seperated words and space seperated character in a list
    """
    out = s, ' '.join(str(s)).replace('  ', ' ').replace('  ', ' ')
    return out


def _one_batch_name_predictor(encoder, model, name):
    """Helped function for predicting a name based on an encode and model"""
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


def _multi_batch_name_predictor(encoder, model, names, batch):
    """Helped function for predicting a name based on an encode and model"""
    encoded = encoder([get_name_pair(str(name).lower()) for name in names],
                      return_attention_mask=True,
                      padding=True, return_tensors='pt')
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'],
                            torch.tensor([0] * batch))
    dataloader = DataLoader(dataset, batch_size=batch)
    batch = next(iter(dataloader))

    inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
              'labels': batch[2], }
    output = softmax(model(**inputs).logits.detach().tolist(), axis = 1)
    return output


class GenderEstimator:
    """Gender estimator based on BERT sub-word tokenization.

    If no parameter is provided to the constructor, the model downloads a predictor
    from the `transformers`'s hub. If you train your own model,
    you can pass the path to the folder that contains it.

    Parameters
    ----------
    name_or_path : string
                   the path of the saved model, by default, it downloads a trained model.

    Example
    -------
    >>> from demographicx import GenderEstimator
    >>> gender_estimator = GenderEstimator()
    >>> gender_estimator.predict('Daniel')
    {'male': 0.9886190672823015,
     'unknown': 0.011367974526753396,
     'female': 1.2958190945360288e-05}
    """

    def __init__(self, name_or_path="liamliang/demographics_gender"):
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def predict(self, name):
        """Predicts gender based on a name.  

        Parameters
        ----------
        name: string
              A first name for which you want to predict the gender.
        batch: int
              The number of names passed for prediction
        
        Returns
        -------
        res : dictionary
              A dictionary that includes the predicted probability of the name being each
              gender for example: {'male': 0.9886190672823015,
                'unknown': 0.011367974526753396, 'female': 1.2958190945360288e-05}
        """
        
        if type(name) == str:
            batch = 1
        else:
            batch = len(name)
        
        if batch == 1:
            output = _one_batch_name_predictor(self.tokenizer, self.model, name)
            res = {'male': output[0], 'unknown': output[1], 'female': output[2]}
            return(res)
        elif batch > 1:
            output = _multi_batch_name_predictor(self.tokenizer, self.model, name, batch = batch)
            res = pd.DataFrame(output).rename({0: 'male',
                             1: 'unknown',
                             2: 'female'
                            }, axis = 1)
            return(res)


class EthnicityEstimator:
    """Ethnicity estimator based on BERT sub-word tokenization.

        If no parameter is provided to the constructor, the model downloads a predictor
        from the `transformers`'s hub. If you train your own model,
        you can pass the path to the folder that contains it.

         Parameters
        ----------
        name_or_path : string
                       the path of the saved model, by default, it downloads a trained
                       model.

        Example
        -------
        >>> from demographicx import EthnicityEstimator
        >>> ethnicity_estimator = EthnicityEstimator()
        >>> ethnicity_estimator.predict('lizhen liang')
        {'black': 2.1461191541442314e-06,
             'hispanic': 4.0070474029127346e-05,
             'white': 0.0002176521167431309,
             'asian': 0.999740131290074}
    """

    def __init__(self, name_or_path="liamliang/demographics_race_v2"):
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def predict(self, name):
        """Predicts ethnicity based on a name
        
        Parameters
        ----------
        name : string
               A full name for which you want to predict the ethnicity.               
        batch: int
              The number of names passed for prediction
        
        Returns
        -------
        res : dictionary
              A dictionary which includes the predicted probability of the name being
              each ethnicity. For example:
              {'black': 4.120965729769303e-06,
              'hispanic': 0.0023926903023342287,
              'white': 0.9963380370701861,
              'asian': 0.00126515166175015}
        """
        
        if type(name) == str:
            batch = 1
        else:
            batch = len(name)
            
        if batch == 1:
            output = _one_batch_name_predictor(self.tokenizer, self.model, name)
            res = {'black': output[2], 'hispanic': output[1], 'white': output[0],
                   'asian': output[3]}
            return(res)
        elif batch > 1:
            output = _multi_batch_name_predictor(self.tokenizer, self.model, name, batch = batch)
            res = pd.DataFrame(output).rename({0: 'white',
                                               1: 'hispanic',
                                               2: 'black',
                                               3: 'asian'
                                              }, axis = 1)
            return(res)

class EthnicityEstimatorCensus:
    """Ethnicity estimator based on BERT sub-word tokenization.

        If no parameter is provided to the constructor, the model downloads a predictor
        from the `transformers`'s hub. If you train your own model,
        you can pass the path to the folder that contains it.

         Parameters
        ----------
        name_or_path : string
                       the path of the saved model, by default, it downloads a trained
                       model.

        Example
        -------
        >>> from demographicx import EthnicityEstimator
        >>> ethnicity_estimator = EthnicityEstimator()
        >>> ethnicity_estimator.predict('lizhen liang')
        {'black': 2.1461191541442314e-06,
             'hispanic': 4.0070474029127346e-05,
             'white': 0.0002176521167431309,
             'asian': 0.999740131290074}
    """

    def __init__(self, name_or_path="liamliang/demographics_race_census"):
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def predict(self, name):
        """Predicts ethnicity based on a name
        
        Parameters
        ----------
        name : string
               A full name for which you want to predict the ethnicity.
        
        Returns
        -------
        res : dictionary
              A dictionary which includes the predicted probability of the name being
              each ethnicity. For example:
              {'black': 4.120965729769303e-06,
              'hispanic': 0.0023926903023342287,
              'white': 0.9963380370701861,
              'asian': 0.00126515166175015}
        """
        
        if type(name) == str:
            batch = 1
        else:
            batch = len(name)
            
        if batch == 1:
            output = _one_batch_name_predictor(self.tokenizer, self.model, name)
            res = {'black': output[0], 'hispanic': output[1], 'white': output[2],
                   'asian': output[3]}
            return(res)
        elif batch > 1:
            output = _multi_batch_name_predictor(self.tokenizer, self.model, name, batch = batch)
            res = pd.DataFrame(output).rename({0: 'black',
                                               1: 'hispanic',
                                               2: 'white',
                                               3: 'asian'
                                              }, axis = 1)
            return(res)

