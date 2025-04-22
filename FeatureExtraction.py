
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def load_audio(path):
    y, sr = librosa.load(path)
    return y,sr

def features_extract(feature_matrix):
    mean = np.mean(feature_matrix, axis = 0)
    std = np.std(feature_matrix, axis= 0)
    skewness = skew(feature_matrix, axis = 0)
    kurt = kurtosis(feature_matrix, axis= 0)

    features = np.concactenate([mean,std,skewness,kurt])
    return features

def features_standardize(feature_matrix):
    mean = np.mean(feature_matrix, axis = 0)
    std = np.std(feature_matrix, axis = 0) + 1e-9 #avoid it being 0
    standard = (feature_matrix - mean)/std

    return standard

def mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y,sr,mfcc_n = 13 ).T #13 bec human audio and that's just the right amount
    stand_features = features_standardize(mfcc)
    features = features_extract(stand_features)
    return features

def chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y,sr).T 
    stand_features = features_standardize(chroma)
    features = features_extract(stand_features)
    return features

def chroma_cqt(y, sr):
    cqt = librosa.feature.chroma_cqt(y,sr).T #13 bec human audio and that's just the right amount
    stand_features = features_standardize(cqt)
    features = features_extract(stand_features)
    return features

def chroma_mfcc(y,sr): #a hybrid approach


#pick am audio gui
#pick a method gui
#funcs do the rest
#put features into fnn
#a seperate visualisation feature is better
