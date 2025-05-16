from FeatureExtraction import load_audio, mfccmethod, chromamethod, chroma_cqt, chroma_mfcc
import pandas as pd
import os

path = "/Users/amnamh/Desktop/TermProject(AI)/MEMD_audio"

data = []
metadata = pd.read_csv("/Users/amnamh/Desktop/TermProject(AI)/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")
metadata.columns = metadata.columns.str.strip()
lookup = metadata.set_index('song_id')[['valence_mean', 'arousal_mean']].to_dict(orient = 'index')

for file in os.listdir("/Users/amnamh/Desktop/TermProject(AI)/MEMD_audio"):
    if file.endswith(".mp3"):
        filepath = os.path.join(path,file)
        audioid = os.path.splitext(file)[0]
        audioid = int(audioid)
        if audioid in lookup:
            print("HI")
            valence = lookup[audioid]["valence_mean"]
            arousal = lookup[audioid]["arousal_mean"]
            y, sr = load_audio(filepath)
            features, mfcc, hop = chromamethod(y,sr)
            #get song id

            row = [audioid] + features.tolist() + [valence, arousal]

            print(f'{audioid} done')
            data.append(row)


column = ['songid']+ [f'chroma{i+1}_mean' for i in range(12)] + [f'chroma{i+1}_std' for i in range(12)] + [f'chroma{i+1}_skew' for i in range(12)] + [f'chroma{i+1}_kurtosis' for i in range(12)] + ['valence_mean'] + ['arousal_mean']

dataframe = pd.DataFrame(data, columns = column)
dataframe.to_csv("chroma.csv", index = False)


