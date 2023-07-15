import librosa
import csv
import os
import numpy as np


class FeatureExtractionPipeLine:
   """
   A class used to extract custom features of voice data using librosa library

   ...

   Attributes
   ----------
   args : str
       names of features that will be extracted from voice
       The attribute name must be part of the list below
       [chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, chroma_cqt, chroma_cens, melspectrogram, spectral_contrast, spectral_flatness, poly_features, tonnetz, tempogram]

   Methods
   -------
   transform(voice_path : str)
       extract features of voice data and save result to data/voice_data.csv
   """

   attributes = dict(
    rmse = librosa.feature.rms,
    chroma_stft = librosa.feature.chroma_stft,
    spectral_centroid = librosa.feature.spectral_centroid,
    spectral_bandwidth = librosa.feature.spectral_bandwidth,
    spectral_rolloff = librosa.feature.spectral_rolloff,
    zero_crossing_rate = librosa.feature.zero_crossing_rate,
    mfcc = librosa.feature.mfcc,
    chroma_cqt = librosa.feature.chroma_cqt,
    chroma_cens= librosa.feature.chroma_cens,
    melspectrogram = librosa.feature.melspectrogram,
    spectral_contrast = librosa.feature.spectral_contrast,
    spectral_flatness = librosa.feature.spectral_flatness,
    poly_features = librosa.feature.poly_features,
    tonnetz = librosa.feature.tonnetz,
    tempogram = librosa.feature.tempogram
   )

   def __init__(self, *args):
    header = ['voice_id'] + list(args)
    if 'mfcc' in header:
      for i in range(1, 21):
        header.append(f' mfcc{i}')
      header.remove('mfcc')
    self.header = header
    self.transformers = args
    for item in args:
      if item not in list(self.attributes.keys()):
        raise Exception(f'No attribute : {item}')

   def __str__(self):
    return f'attributes : {self.attributes.keys()}'

   def transform(self, voice_path : str):
    """ extract features of voice data and save result to data/voice_data.csv

    Parameters
    ----------
    voice_path : str
        path of the voices file
    """

    with open(r'D:\Speech processing\Speech Classification and Clustering\src\dataset\voice_data.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(self.header)

    for filename in os.listdir(voice_path):
        songname = f'{voice_path}/{filename}'
        y0, sr0 = librosa.load(songname, mono=True)
        to_append = f'{filename}'

        for transform in self.transformers:
            if transform in ['rmse', 'zero_crossing_rate', 'spectral_flatness']:
              feature = self.attributes[transform](y=y0)

            else:
              feature = self.attributes[transform](y=y0, sr=sr0)

            if transform == 'mfcc':
              for e in feature:
                  to_append += f' {np.mean(e)}'
            else:
              to_append += f' {np.mean(feature)}'
        with open(r"D:\Speech processing\Speech Classification and Clustering\src\dataset\voice_data.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())