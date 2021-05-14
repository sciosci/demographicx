import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer

__all__ = [
    'GenderEstimator',
    'EthnicityEstimator'
]


def get_name_pair(s):
    """Creates space seperated words and space seperated character in a list

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
        """Creates an estimator instance  
        
        Parameters
        ----------
        name_or_path : string
                       the path of the saved model, by default, it downloads a trained model.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def predict(self, name):
        """Predicts gender based on a name.  

        Parameters
        ----------
        name: string
              A first name for which you want to predict the gender.
        
        Returns
        -------
        res : dictionary
              A dictionary that includes the predicted probability of the name being each gender
              for example: {'male': 0.9886190672823015, 'unknown': 0.011367974526753396, 'female': 1.2958190945360288e-05}
        """
        output = one_batch_name_predictor(self.tokenizer, self.model, name)
        res = {'male': output[0], 'unknown': output[1], 'female': output[2]}
        return res


class EthnicityEstimator:
    def __init__(self, name_or_path="liamliang/demographics_race"):
        """Creates an estimator instance
        
        Parameters
        ----------
        name_or_path : string
                       the path of the saved model, by default, it downloads a trained model.
        """
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
              A dictionary which includes the predicted probability of the name being each ethnicity 
              for example: {'black': 4.120965729769303e-06, 'hispanic': 0.0023926903023342287, 'white': 0.9963380370701861, 'asian': 0.00126515166175015}
        """
        output = one_batch_name_predictor(self.tokenizer, self.model, name)
        res = {'black': output[0], 'hispanic': output[1], 'white': output[2],
               'asian': output[3]}
        return res
