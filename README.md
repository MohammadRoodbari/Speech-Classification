# Speech-Classification

> This repository is made for the Machine Learning course project - 2022.

---
## Introduction

The goal of this project is to implement an audio classification system, which:
1. first reads in an audio,
2. and then recognizes the class(label) of this audio.


## Classes

The data is divided into four class based on emotion:
> anger, happiness, sadness, neutral

Also The data is divided into two class based on sex:
> male, female


## Method

Features: MFCCs (Mel-frequency cepstral coefficients), Spectral Centroid, Spectral Bandwidth, Rolloff, Melspectrogram, Spectral Contrast, Spectral Flatness are computed from the raw audio using [librosa](https://github.com/librosa/librosa) package.

Classifier: SVM (Support Vector Machine) is adopted for gender classificatioin, and CNN (Convolutional Neural Network) is adopted for emotion classificatioin

## Result

- Gender Classificatioin

| Train Acc.      |  Test Acc.  |  F1 Score.  |
| :-------------: | :---------: | :---------: |
|      0.999      |   0.95440   |    0.9543   |

- Emotion Classificatioin

| Train Acc.      |  Test Acc.  |
| :-------------: | :---------: |
|      0.601      |   0.4742    |


## Download Data

Voice data can be downloaded from [here](https://drive.google.com/drive/folders/1wnJ9eFlnJZsY1lNTOgLbnRrVgc_RDUNI?usp=drive_link).

## Install dependencies

In a Python3 virtual environment run:

```bash
pip install -r requirements.txt
```

