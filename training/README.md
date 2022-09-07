# Training the models

The training of our models is based on three datasets:

- [Genni + Ethnea for Author-ity 2009 dataset](https://databank.illinois.edu/datasets/IDB-9087546)
- [Wikipedia dataset](https://github.com/appeler/ethnicolr/tree/master/ethnicolr/data/wiki)
- [Social Security Administration baby names](https://data.world/howarder/gender-by-name)
- [Frequently Occurring Surnames from the 2010 Census](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html)

## Obtaining the data

To download the models, execute the following commands:

```bash
# the data from the SSA is already in data
wget https://databank.illinois.edu/datafiles/xg3jn/download -O data/genni-ethnea-authority2009.tsv
wget https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/wiki/wiki_name_race.csv?raw=true -O data/wiki_name_race.csv
```

## Preprocessing

For preprocessing and training, you would need to install other packages

```
pip install -r requirements.txt
```

```bash
python preprocessing.py
```

## Training models

After running the following code, models will be saved to `../demographics_ethnicity`
and `../demographics_gender`

```bash
python train_gender.py
python train_ethnicity.py
```

## Trying your new model

You can try your new models as follows

```python
from demographicx import GenderEstimator

gender_estimator = GenderEstimator(name_or_path="../demographics_gender")
gender_estimator.predict('Daniel')
```
## Training environment  
The default model was trained using Nvidia GeForce RTX 2070 (8 Gb GDDR6)    
Driver Version: 450.119.03     
CUDA Version: 11.0   