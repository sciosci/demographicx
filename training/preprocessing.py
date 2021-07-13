# Packages
import pandas as pd
from sklearn.model_selection import train_test_split

# Import genni-ethnea-authority2009 dataset
names = pd.read_csv('./data/genni-ethnea-authority2009.tsv', sep='\t',
                    error_bad_lines=False)

# Get Labels (Estimated Gender)
gender_pred = names[['Genni', 'SexMac', 'SSNgender']]

# Get Majority Vote for Gender Labels
gender_gt = []
for i in range(len(gender_pred)):
    if (int(gender_pred.loc[i, 'Genni'] == 'M') + int(
        gender_pred.loc[i, 'SexMac'] == 'male') + int(
        gender_pred.loc[i, 'SexMac'] == 'mostly_male') + int(
        gender_pred.loc[i, 'SSNgender'] == 'M') >= 2):
        gender_gt.append('m')
    elif (int(gender_pred.loc[i, 'Genni'] == 'F') + int(
        gender_pred.loc[i, 'SexMac'] == 'female') + int(
        gender_pred.loc[i, 'SexMac'] == 'mostly_female') + int(
        gender_pred.loc[i, 'SSNgender'] == 'F') >= 2):
        gender_gt.append('f')
    elif (int(gender_pred.loc[i, 'Genni'] == '-') + int(
        gender_pred.loc[i, 'SexMac'] == 'andy') + int(
        gender_pred.loc[i, 'SSNgender'] == '-') >= 2):
        gender_gt.append('-')
    else:
        gender_gt.append('weak')

gender_pred['pred_gender_gt'] = gender_gt
gender_pred['name'] = names['name']
gender_df = gender_pred[gender_pred['pred_gender_gt'] != 'weak'][
    ['name', 'pred_gender_gt']]

# Take Ethnea as Label
# Removing Origins that Are Too Detailed
ethnic = names[['name', 'Ethnea']]
ethnic = ethnic[~ethnic['Ethnea'].str.contains('-')]
ethnic = ethnic[(ethnic['Ethnea'] != 'TOOSHORT') & (ethnic['Ethnea'] != 'UNKNOWN')]

# Aggregate Origins into Four Categories
ethnic['label_Ethnea'] = ethnic['Ethnea'].replace({'CHINESE': 'ASIAN',
                                                    'ARAB': 'ASIAN',
                                                    'INDIAN': 'ASIAN',
                                                    'INDONESIAN': 'ASIAN',
                                                    'ISRAELI': 'ASIAN',
                                                    'JAPANESE': 'ASIAN',
                                                    'KOREAN': 'ASIAN',
                                                    'POLYNESIAN': 'ASIAN',
                                                    'MONGOLIAN': 'ASIAN',
                                                    'THAI': 'ASIAN',
                                                    'VIETNAMESE': 'ASIAN',
                                                    'BALTIC': 'WHITE',
                                                    'CARIBBEAN': 'HISPANIC',
                                                    'DUTCH': 'WHITE',
                                                    'ENGLISH': 'WHITE',
                                                    'FRENCH': 'WHITE',
                                                    'GERMAN': 'WHITE',
                                                    'GREEK': 'WHITE',
                                                    'ITALIAN': 'WHITE',
                                                    'TURKISH': 'WHITE',
                                                    'HUNGARIAN': 'WHITE',
                                                    'NORDIC': 'WHITE',
                                                    'ROMANIAN': 'WHITE',
                                                    'SLAV': 'WHITE',
                                                    'AFRICAN': 'BLACK'})

# Inport wikipedia, ssa dataset
wiki = pd.read_csv('./data/wiki_name_race.csv')
ssa = pd.read_csv('./data/name_gender_ssa.csv')

# Change Label Names
ssa['gender'] = ssa['gender'].replace({
    'M': 'm',
    'F': 'f'
})

# Aggregate Categories
wiki['label'] = wiki['race'].replace({
    'Asian,GreaterEastAsian,EastAsian': 'ASIAN',
    'Asian,GreaterEastAsian,Japanese': 'ASIAN',
    'Asian,IndianSubContinent': 'ASIAN',
    'GreaterAfrican,Africans': 'BLACK',
    'GreaterAfrican,Muslim': 'BLACK',
    'GreaterEuropean,British': 'WHITE',
    'GreaterEuropean,EastEuropean': 'WHITE',
    'GreaterEuropean,Jewish': 'WHITE',
    'GreaterEuropean,WestEuropean,French': 'WHITE',
    'GreaterEuropean,WestEuropean,Germanic': 'WHITE',
    'GreaterEuropean,WestEuropean,Hispanic': 'HISPANIC',
    'GreaterEuropean,WestEuropean,Italian': 'WHITE',
    'GreaterEuropean,WestEuropean,Nordic': 'WHITE'
})

# Clean wiki and ssa
wiki['name'] = wiki['name_first'] + ' ' + wiki['name_last']
ssa['name'] = ssa['name'].apply(lambda x: str(x).lower())

# Clean gender, concat ssa with gender dataframe
gender_df['name'] = gender_df['name'].apply(lambda x: str(x).split(' ')[0].lower())
gender_df = gender_df.rename({'pred_gender_gt': 'gender'}, axis=1)
gender = pd.concat([gender_df, ssa[['name', 'gender']]])

# Sample balanced dataset
gender_sampled = pd.concat([gender[gender['gender'] == 'm'].sample(n=600000),
                            gender[gender['gender'] == 'f'].sample(n=600000),
                            gender[gender['gender'] == '-'].sample(
                                n=600000)]).reset_index(drop=True)

# Training Validation Split
gender_sampled_train, gender_sampled_val = train_test_split(gender_sampled, test_size=0.2,
                                                            stratify=gender_sampled[
                                                                'gender'])

gender_sampled_train.to_csv('./data/gender_mixed_train.csv', index=False)
gender_sampled_val.to_csv('./data/gender_mixed_val.csv', index=False)

# Concat ethinic dataframe and wiki dataframe
# Sample balanced dataset
ethnic = pd.concat(
    [ethnic[['name', 'label_Ethnea']].rename({'label_Ethnea': 'label'}, axis=1),
     wiki[['name', 'label']].dropna()])
ethnic = ethnic.drop_duplicates().reset_index(drop=True)
ethnic_sampled = pd.concat(
    [ethnic[ethnic['label'] == 'BLACK'].sample(n=180000, replace=True),
     ethnic[ethnic['label'] == 'ASIAN'].sample(n=180000),
     ethnic[ethnic['label'] == 'HISPANIC'].sample(n=180000),
     ethnic[ethnic['label'] == 'WHITE'].sample(n=180000)]).reset_index(drop=True)
ethnic_sampled_train, ethnic_sampled_val = train_test_split(ethnic_sampled, test_size=0.2,
                                                            stratify=ethnic_sampled[
                                                                'label'])

ethnic_sampled_train.to_csv('./data/ethnic_mixed_train.csv', index=False)
ethnic_sampled_val.to_csv('./data/ethnic_mixed_val.csv', index=False)



