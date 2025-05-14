
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

#def load_audio(path):
def load_audio():
    #y, sr = librosa.load(path)
    y,sr = librosa.load("/Users/amnamh/Desktop/TermProject(AI)/MEMD_audio/5.mp3")
    ycut = y[int(14 * sr):] #cut audio from 15 - 45
    print("File Loaded")
    return ycut,sr

def features_extract(feature_matrix):
    mean = np.mean(feature_matrix, axis = 0)
    std = np.std(feature_matrix, axis= 0)
    skewness = skew(feature_matrix, axis = 0)
    kurt = kurtosis(feature_matrix, axis = 0)
    features = np.concatenate([mean, std, skewness, kurt])
    
    return features

def features_standardize(feature_matrix):
    mean = np.mean(feature_matrix, axis = 0)
    std = np.std(feature_matrix, axis = 0) + 1e-9 #avoid it being 0
    standard = (feature_matrix - mean)/std #z-score normalization
    return standard

def mfccmethod(y, sr):
    mfcc = librosa.feature.mfcc(y= y,sr = sr, n_mfcc = 13, hop_length = 1024).T
    #13 bec human audio and that's just the right amount
    stand_features = features_standardize(mfcc)
    features = features_extract(stand_features)
    return features #52

def chromamethod(y, sr):
    chroma = librosa.feature.chroma_stft(y= y,sr= sr,hop_length = 1024).T 
    stand_features = features_standardize(chroma)
    features = features_extract(stand_features)
    return features #48

def chroma_cqt(y, sr):
    cqt = librosa.feature.chroma_cqt(y=y,sr= sr, hop_length = 1024).T 
    stand_features = features_standardize(cqt)
    features = features_extract(stand_features)
    return features #48

def chroma_mfcc(y,sr): #a hybrid approach
    chroma = chromamethod(y,sr)
    mfcc = mfccmethod(y,sr)
    features = np.concatenate([mfcc, chroma])
    return features #100

if __name__ == "__main__":
    y, sr = load_audio()
    feat = chroma_cqt(y,sr)
    print(len(feat))
    print(feat)

#pick am audio gui
#pick a method gui
#funcs do the rest
#put features into fnn
#a seperate visualisation feature is better

