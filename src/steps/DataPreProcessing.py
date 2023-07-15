import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreProcessing:
  """
  A class used to preprocessing label dataset and voice features dataset

  ...

  Attributes
  ----------
  label_dataset_path : str
      path of the label dataset

  voice_dataset_path :
      path of the voices features dataset

  Methods
  -------
  transform()
       preprocess label and voice datasets and merge them and build Data_final and save result to data/Data_final.csv
   """

  def __init__(self, label_dataset_path, voice_dataset_path):
    self.label_dataset_path = label_dataset_path
    self.voice_dataset_path = voice_dataset_path

  def transform(self):
    """preprocess label and voice datasets and merge them and return Data_final

    Parameters
    ----------
    none

    """
    label_dataset  = pd.read_csv(self.label_dataset_path)
    voice_dataset  = pd.read_csv(self.voice_dataset_path)

    #voice_dataset preprocessing
    for i in range (0, len(voice_dataset)):
      voice_dataset['voice_id'][i] = voice_dataset['voice_id'][i].replace('.wav', '')

    voice_dataset['voice_id']=voice_dataset['voice_id'].astype(str).astype(int)
    voice_dataset=voice_dataset.sort_values('voice_id')

    # scale data using StandardScaler
    Sc = StandardScaler()
    voice_dataset[voice_dataset.columns.drop('voice_id')] = Sc.fit_transform(voice_dataset.drop('voice_id' , axis=1))

    # remove Duplicate Rows based on voice id Column
    label_dataset.drop_duplicates(subset=['voice id'] , inplace=True)

    # rename voice id Column to voiceId & sort Rows based on voiceId
    label_dataset = label_dataset.rename({'voice id': 'voiceId'}, axis=1).sort_values('voiceId')

    # remove rows of label dataset and voice dataset that have voiceId that not exist in another dataset
    for j in voice_dataset['voice_id'] :
        if not ((label_dataset.voiceId == j).any()) :
            voice_dataset.drop(voice_dataset[voice_dataset['voice_id']== j].index, axis=0, inplace=True)

    for j in label_dataset['voiceId'] :
        if not ((voice_dataset.voice_id == j).any()) :
            label_dataset.drop(label_dataset[label_dataset['voiceId']== j].index, axis=0, inplace=True)

    # merge label dataset and voice dataset and build Data_final
    Data_final = pd.concat([voice_dataset.reset_index(), label_dataset.reset_index()], axis=1, join='inner')
    Data_final=Data_final.drop(['voiceId', 'index'],axis=1)

    # sex column must have only two element : f = female and m = male
    for i in range (0 , len(Data_final) ) :
      if (Data_final['sex'][i] == 'f') :
        Data_final['sex'][i] = 'f'
      if (Data_final['sex'][i] == 'f ') :
        Data_final['sex'][i] = 'f'
      if (Data_final['sex'][i] == 'F') :
        Data_final['sex'][i] = 'f'
      if (Data_final['sex'][i] == 'M') :
        Data_final['sex'][i] = 'm'
      if (Data_final['sex'][i] == 'm') :
        Data_final['sex'][i] = 'm'
      if (Data_final['sex'][i] == 'w') :
        Data_final['sex'][i] = 'm'

    # encode sex column with label encoder in sklearn library
    LE = LabelEncoder()
    Data_final['sex'] = LE.fit_transform(Data_final['sex'])

    # save dataset to .csv file
    with open('D:\Speech processing\Speech Classification and Clustering\src\dataset\Data_final.csv', 'w', encoding = 'utf-8-sig') as f:
      Data_final.to_csv(f)

    return Data_final