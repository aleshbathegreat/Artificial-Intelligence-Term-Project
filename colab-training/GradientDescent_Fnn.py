import os
import numpy as np
import pandas as pd
import librosa
import random
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from simpful import *

csv_path = '/kaggle/input/deap-data/Deap-data/static_annotations_averaged_songs_1_2000.csv'
dataset_path = '/kaggle/input/deap-data/Deap-data/MEMD_audio'

#clean and re-save CSV
df = pd.read_csv(csv_path)
df.columns = [col.strip() for col in df.columns]  #remove column name spaces difference in names was causing issues 
cleaned_csv_path = 'cleaned_file.csv'
df.to_csv(cleaned_csv_path, index=False)
""""
SUMMARY:
1. uploaded the files audio and csv, make sure names of columns are same
2. features extracted from audios to give input to cnn
3. split data into 80% training and 20% for testing
4. give 80% data as input to cnn
5. fuzzy logic to map valence & arousal to emotion
6. did the backward propogation seperately so EA and SA can also be added to the same code 
7. trained data 
8. tested and printed 10 
"""
# Feature Extraction 
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=50)  #had to cut down length of audio to 30
        return y,sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def features_standardize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-9
    return (features - mean)/ std

def features_extract(feature_matrix):
    features = np.concatenate([
        np.mean(feature_matrix, axis=0),
        np.std(feature_matrix, axis=0),
        skew(feature_matrix, axis=0),
        kurtosis(feature_matrix, axis=0)
    ])
    #replace NaNs and Infs with 0
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def chroma_mfcc(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    
    chroma_std = features_standardize(chroma)
    
    mfcc_std = features_standardize(mfcc)

    return np.concatenate([
        features_extract(chroma_std),
        features_extract(mfcc_std)
    ])

def process_dataset(audio_dir, csv_path, max_files=200):
    label_dict = {}
    import csv
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song_id = row['song_id'].strip()
            try:
                valence = float(row['valence_mean'])
                arousal = float(row['arousal_mean'])
                label_dict[song_id] = (valence, arousal)
            except ValueError:
                print(f"Skipping invalid row: {row}")

    X,y_valence,y_arousal = [], [], []
    files_processed = 0

    for file in os.listdir(audio_dir):
        if files_processed >= max_files:
            break
        if not file.lower().endswith(('.mp3')):
            continue

        song_id = os.path.splitext(file)[0]
        if song_id not in label_dict:
            continue

        file_path = os.path.join(audio_dir, file)
        audio, sr = load_audio(file_path)

        if audio is None or len(audio) == 0:
            print(f"Skipping empty or unreadable file: {file}")
            continue

        max_length = sr * 60
        if len(audio) > max_length:
            print(f"Trimming {file} from {len(audio)/sr:.2f}s to 60s")
            audio = audio[:max_length]

        try:
            features = chroma_mfcc(audio, sr)
            
            if not np.all(np.isfinite(features)):
                print(f"Skipping {file} due to NaNs or Infs in features.")
                continue

        except Exception as e:
            print(f"Feature extraction failed for {file}: {e}")
            continue

        X.append(features)
        val, ars = label_dict[song_id]
        y_valence.append(val)
        y_arousal.append(ars)
        files_processed += 1

    return np.array(X), np.array(y_valence), np.array(y_arousal)

#data Split 
def split_data(X, y_val, y_ars, test_ratio=0.2, seed=42):
    total_samples = len(X)
    indices = list(range(total_samples))
    random.seed(seed)
    random.shuffle(indices)

    split = int(total_samples * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]

    return (
        X[train_idx], X[test_idx],
        y_val[train_idx], y_val[test_idx],
        y_ars[train_idx], y_ars[test_idx]
    )

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.model(x)

# Fuzzy Logic System 
class EmotionLogic:
    def __init__(self):
        self.fuzz = FuzzySystem()
        
        self.fuzz.add_linguistic_variable("valence", AutoTriangle(3, terms=['negative', 'neutral', 'positive'], universe_of_discourse=[1, 9]))
        self.fuzz.add_linguistic_variable("arousal", AutoTriangle(3, terms=['calm', 'neutral', 'excited'], universe_of_discourse=[1, 9]))

        for i, emotion in enumerate(['sad', 'bored', 'angry', 'sleepy', 'calm', 'excited', 'relaxed', 'pleasant', 'happy'], 1):
            self.fuzz.set_crisp_output_value(emotion, i)

        self.fuzz.add_rules([
            "IF (valence IS negative) AND (arousal IS calm) THEN (emotion IS sad)",
            "IF (valence IS negative) AND (arousal IS neutral) THEN (emotion IS bored)",
            "IF (valence IS negative) AND (arousal IS excited) THEN (emotion IS angry)",
            "IF (valence IS neutral) AND (arousal IS calm) THEN (emotion IS bored)",
            "IF (valence IS neutral) AND (arousal IS neutral) THEN (emotion IS calm)",
            "IF (valence IS neutral) AND (arousal IS excited) THEN (emotion IS excited)",
            "IF (valence IS positive) AND (arousal IS calm) THEN (emotion IS relaxed)",
            "IF (valence IS positive) AND (arousal IS neutral) THEN (emotion IS pleasant)",
            "IF (valence IS positive) AND (arousal IS excited) THEN (emotion IS happy)"
        ])

        self.emotion_map = {
            1: 'sad', 2: 'bored', 3: 'angry', 4: 'sleepy', 5: 'calm',
            6: 'excited', 7: 'relaxed', 8: 'pleasant', 9: 'happy'
        }

    def fuzzy(self, ars, val):
        self.fuzz.set_variable("valence", val)
        self.fuzz.set_variable("arousal", ars)
        return self.emotion_map[int(self.fuzz.Sugeno_inference()["emotion"])]

X, y_valence, y_arousal = process_dataset(dataset_path, cleaned_csv_path, max_files=50)
X_train, X_test, y_val_train, y_val_test, y_ars_train, y_ars_test = split_data(X, y_valence, y_arousal)

model = EmotionClassifier(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.column_stack([y_val_train, y_ars_train]), dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

for epoch in range(30):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/30, Loss: {total_loss/len(train_loader):.4f}")

#test & Emotion Inference
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
predictions = model(X_test_tensor).detach().numpy()

fuzzy_logic = EmotionLogic()

print("\nSample Inference Results:")
for i in range(min(10, len(predictions))):
    val_pred, ars_pred = predictions[i]
    emotion = fuzzy_logic.fuzzy(ars_pred, val_pred)
    print(f"Valence: {val_pred:.2f}, Arousal: {ars_pred:.2f} -> Emotion: {emotion}")

print("\n Test Results (Sampled) ")
for i in range(min(10, len(predictions))):
    pred_val, pred_ars = predictions[i]
    actual_val, actual_ars = y_val_test[i], y_ars_test[i]
    predicted_emotion = fuzzy_logic.fuzzy(pred_ars, pred_val)
    actual_emotion = fuzzy_logic.fuzzy(actual_ars, actual_val)

    print(f"Sample {i+1}")
    print(f"  Predicted: Valence={pred_val:.2f}, Arousal={pred_ars:.2f}, Emotion={predicted_emotion}")
    print(f"  Actual   : Valence={actual_val:.2f}, Arousal={actual_ars:.2f}, Emotion ={actual_emotion}\n")
