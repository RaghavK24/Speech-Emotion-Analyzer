# Emotion Recognition from Audio

This project focuses on recognizing emotions from audio files using a Convolutional Neural Network (CNN). The code processes audio files, extracts relevant features, applies data augmentation, and trains a CNN model to classify emotions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to classify emotions from audio files. The dataset includes audio recordings labeled with emotions, which are processed and augmented to improve the model's robustness. The features are extracted from the audio data and used to train a CNN model for emotion classification.

## Features
The following features are extracted from the audio files:
- Zero Crossing Rate
- Chroma_stft
- MFCC (Mel Frequency Cepstral Coefficients)
- RMS (Root Mean Square) value
- MelSpectrogram

## Data Augmentation
To improve the model's performance, data augmentation techniques are applied:
- **Noise Injection**: Adding random noise to the audio signal.
- **Time Stretching**: Stretching the audio signal in time.
- **Pitch Shifting**: Changing the pitch of the audio signal.
- **Time Shifting**: Shifting the audio signal in time.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Dense layer with softmax activation for classification


## Acknowledgements
References for feature extraction is taken from Akash Mallik's blog on audio signal feature extraction and clustering, available on Medium: [Audio signal feature extraction and clustering](https://medium.com/heuristics/audio-signal-feature-extraction-and-clustering-935319d2225).
