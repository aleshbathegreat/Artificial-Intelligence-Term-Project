import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
import librosa 
import librosa.display
import IPython.display  as ipd 
# Audio files path
dataset_path = '/kaggle/input/deap-data/Deap-data/MEMD_audio'

arousal = pd.read_csv('/kaggle/input/deap-dataset/Deap-data/arousal.csv')
valence = pd.read_csv('/kaggle/input/deap-dataset/Deap-data/valence.csv')


arousal['mean_arousal'] = arousal.iloc[:, 1:].mean(axis=1)
valence['mean_valence'] = valence.iloc[:, 1:].mean(axis=1)

arousal = arousal[['song_id', 'mean_arousal']]
valence = valence[['song_id', 'mean_valence']]

merged = pd.merge(arousal, valence, on='song_id')



class Emotion_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Emotion_Classifier, self),__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  #input -> hidden
        self.relu = nn.ReLU()                             #ReLU is activation function
        self.layer2 = nn.Linear(hidden_size, num_classes) #hidden -> output

    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

#number of neurons in each layer
input_size = len(features)
hidden_size = 128
num_classes = 4
num_epochs = 30



