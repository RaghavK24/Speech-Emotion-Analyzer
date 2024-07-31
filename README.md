# Emotion Recognition from Audio

This project focuses on recognizing emotions from audio files using a Convolutional Neural Network (CNN). The code processes audio files, extracts relevant features, applies data augmentation, and trains a CNN model to classify emotions.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Augmentation](#data-augmentation)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to classify emotions from audio files. The dataset includes audio recordings labeled with emotions, which are processed and augmented to improve the model's robustness. The features are extracted from the audio data and used to train a CNN model for emotion classification.

## Data Augmentation
To improve the model's performance, data augmentation techniques are applied:
- **Noise Injection**: Adding random noise to the audio signal.
- **Time Stretching**: Stretching the audio signal in time.
- **Time Shifting**: Shifting the audio signal in time.

  
## Features

The following features are extracted from the audio files:

### MFCC (Mel Frequency Cepstral Coefficients)

- **MFCCs** are coefficients that represent the short-term power spectrum of a sound. They are derived from a type of cepstral representation of the audio clip.
- **Purpose**: MFCCs are used to represent the power spectrum of the sound, providing a compact representation of the audio signal.
- **Calculation**: 
  - The audio signal is divided into short overlapping windows.
  - For each window, the power spectrum is computed.
  - The power spectrum is then converted to the Mel scale using a filter bank.
  - The logarithm of the Mel spectrum is taken.
  - Finally, the discrete cosine transform (DCT) is applied to the log Mel spectrum to obtain the MFCCs.
- **Application**: MFCCs are commonly used in speech and audio processing, such as speech recognition, speaker identification, and audio classification, due to their ability to capture the timbral aspects of sound.
- **Implementation**: Typically, 13 to 40 MFCCs are extracted per frame, and these coefficients are used as features for further processing in the machine learning model.




## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Dense layer with softmax activation for classification


## Acknowledgements
References for feature extraction is taken from Akash Mallik's blog on audio signal feature extraction and clustering, available on Medium: [Audio signal feature extraction and clustering](https://medium.com/heuristics/audio-signal-feature-extraction-and-clustering-935319d2225).
