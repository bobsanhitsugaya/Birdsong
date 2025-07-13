import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_metadata(json_path):
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'recordings' in data:
        recordings = data['recordings']
    else:
        recordings = data
    return pd.DataFrame(recordings)

def filter_quality(df, quality_keep=['A', 'B', 'C']):
    return df[df['q'].isin(quality_keep)].reset_index(drop=True)

def mmss_to_seconds(val):
    try:
        parts = str(val).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
        else:
            return None
    except:
        return None

def extract_mfcc_features(audio_path, sr=22050, n_mfcc=20):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = librosa.util.normalize(audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs, axis=1)
    return mfccs

def batch_extract_features(df, audio_dir, sr=22050, n_mfcc=20):
    features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_file = row['file'] if 'file' in row else f"{row['gen']}_{row['id']}.mp3"
        audio_path = os.path.join(audio_dir, audio_file)
        try:
            mfcc = extract_mfcc_features(audio_path, sr, n_mfcc)
            features.append(mfcc)
            labels.append(row['en'])
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return np.array(features), np.array(labels)

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
