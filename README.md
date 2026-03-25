# Human-Emotion-Detection-from-Voice-
Human emotion recognition from voice signals is a machine learning approach that analyzes speech patterns to detect feelings like happiness, anger, or sadness. It extracts features such as pitch, tone, and MFCCs, then uses models to classify emotions, enabling applications in healthcare, customer service, and human-computer interaction.
# Import Libraries
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Extract Features
# -------------------------------
def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    return mfccs

# -------------------------------
# STEP 2: Load Dataset
# -------------------------------
# Replace this path with your dataset folder
dataset_path = "ravdess_data/"

features = []
labels = []

# Emotion labels (basic)
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # Extract emotion from filename
            emotion_code = file.split("-")[2]
            emotion = emotion_dict.get(emotion_code)
            
            if emotion is not None:
                feature = extract_feature(file_path)
                features.append(feature)
                labels.append(emotion)

print("Dataset Loaded Successfully!")

# -------------------------------
# STEP 3: Train Test Split
# -------------------------------
X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# STEP 4: Train Model
# -------------------------------
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# -------------------------------
# STEP 5: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -------------------------------
# STEP 6: Predict New Audio
# -------------------------------
def predict_emotion(file_name):
    feature = extract_feature(file_name)
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# Test with your own audio file
test_file = "test.wav"  # replace with your file
print("Predicted Emotion:", predict_emotion(test_file))
