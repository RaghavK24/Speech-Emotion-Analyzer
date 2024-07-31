import joblib
from keras.models import load_model
import os
import librosa
import numpy as np


preprocessing_folder = 'preprocessing_objects'
model_folder = 'models'


encoder = joblib.load(os.path.join(preprocessing_folder, 'onehot_encoder.pkl'))
scaler = joblib.load(os.path.join(preprocessing_folder, 'robust_scaler.pkl'))
model = load_model(os.path.join(model_folder, 'Emotion_Voice_Detection_Model_final.h5'))

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def extract_features(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return np.array(mfcc)

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) # stacking vertically

    # data with shift
    shift_data = shift(data)
    res3 = extract_features(shift_data, sample_rate)
    result = np.vstack((result, res3)) # stacking vertically

    # data with stretching
    new_data = stretch(data)
    res4 = extract_features(new_data, sample_rate)
    result = np.vstack((result, res4)) # stacking vertically

    return result

def predict_label(file_path):
    # Load the audio file
    data, sample_rate = librosa.load(file_path)

    # Extract features from the new audio file
    features = extract_features(data, sample_rate)

    # Standardize the features
    features = scaler.transform([features])

    # Reshape to fit the model input
    features = np.expand_dims(features, axis=2)

    # Make prediction
    preds = model.predict(features, verbose=1)

    # Convert prediction to class index
    pred_label_index = np.argmax(preds, axis=1)

    # Convert class index to original label
    original_labels = encoder.categories_[0]

    # Map the class index to the original label
    predicted_label = original_labels[pred_label_index]

    return predicted_label

path = 'output10.wav'

predicted_label = predict_label(path)
print("Predicted label: ", predicted_label)

