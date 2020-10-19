import numpy as np
import glob
import os
import math
import librosa
from common import *

n_fold = 74

# load npz files including raw audio samples and annotations
def _load_file(npz_f):
    with np.load(npz_f) as f:
        tmp_data = f["x"]
        tmp_song_arousal_label = f["y_song_arousal"]
        tmp_song_valence_label = f["y_song_valence"]
        tmp_valence_label = f['y_valence']
        tmp_arousal_label = f["y_arousal"]

    # Reshape the data to match the input of the model
    tmp_data = tmp_data.reshape(60, 22050, 1)
    return tmp_data, tmp_song_arousal_label, tmp_song_valence_label, tmp_valence_label, tmp_arousal_label

# prepare the input and labels for model validation and test dataset
# here labels are arousal and valence values
def _load_data(files):
    data = []
    valence_labels = []
    arousal_labels = []

    for npz_f in files:
        tmp_data, _, _, tmp_valence_label, tmp_arousal_label = _load_file(npz_f)

        data.append(tmp_data)
        valence_labels.append(tmp_valence_label)
        arousal_labels.append(tmp_arousal_label)

    x = np.stack(data).reshape(-1, 22050, 1)
    y = np.stack((np.stack(arousal_labels), np.stack(valence_labels)), axis=2).reshape(-1, 2)
    return x, y

# prepare the input and labels for model training dataset with data augmentation
# here labels are arousal and valence values
def _load_data_augment(files):
    factor = 0.8
    sr = 44100
    data = []
    valence_labels = []
    arousal_labels = []

    for npz_f in files:
        tmp_data, _, _, tmp_valence_label, tmp_arousal_label = _load_file(npz_f)
        # print("tmp_data shape: {}".format(tmp_data.shape))

        data.append(tmp_data)
        valence_labels.append(tmp_valence_label)
        arousal_labels.append(tmp_arousal_label)
        # data augmentation:

        # ts_data = librosa.effects.time_stretch(tmp_data, factor)
        ts_data = np.flip(tmp_data, axis=1)
        data.append(ts_data)
        valence_labels.append(tmp_valence_label)
        arousal_labels.append(tmp_arousal_label)

        ps_data = librosa.effects.pitch_shift(tmp_data.reshape(60 * 22050), sr, n_steps=-1)
        data.append(ps_data.reshape(60, 22050, 1))
        valence_labels.append(tmp_valence_label)
        arousal_labels.append(tmp_arousal_label)

    x = np.stack(data).reshape(-1, 22050, 1)
    y = np.stack((np.stack(arousal_labels), np.stack(valence_labels)), axis=2).reshape(-1, 2)
    return x, y


def trans_angle_len(data):
    sin_labels = []
    cos_labels = []
    len_labels = []

    for idx in range(len(data)):
        x, y = data[idx]
        tmp_len = math.sqrt(x * x + y * y)
        tmp_sin = x / tmp_len
        tmp_cos = y / tmp_len

        sin_labels.append(tmp_sin)
        cos_labels.append(tmp_cos)
        len_labels.append(tmp_len)

    labels = np.stack((np.stack(sin_labels), np.stack(cos_labels), np.stack(len_labels)), axis=1)
    return labels

# partition data for k-fold cross validation
def load_split(fold_idx):
    assert 10 > fold_idx >= 0
    path = os.getenv('data_path', '../data/*.npz')
    files = sorted(glob.glob(path))
    test_f = files[fold_idx*n_fold:(fold_idx+1)*n_fold]
    if fold_idx < 9:
        valid_f = files[(fold_idx+1)*n_fold:(fold_idx+2)*n_fold]
    else:
        valid_f = files[:n_fold]

    train_f = set(files) - set(test_f + valid_f)

    print('Training files:')
    for f in sorted(train_f):
        print(f)

    print('Validation files:')
    for f in sorted(valid_f):
        print(f)

    print('Testing files:')
    for f in sorted(test_f):
        print(f)

    if my:
        print("my model use data augmentation")
        x_train, y_train = _load_data_augment(train_f)
    else:
        print("base model: no data augmentation")
        x_train, y_train = _load_data(train_f)
    x_test, y_test = _load_data(test_f)
    x_valid, y_valid = _load_data(valid_f)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def load_split_for_test(fold_idx):
    assert 10 > fold_idx >= 0
    path = os.getenv('data_path', '../data/*.npz')
    files = sorted(glob.glob(path))
    test_f = files[fold_idx*n_fold:(fold_idx+1)*n_fold]

    # print('Testing files:')
    # for f in sorted(test_f):
    #     print(f)

    x_test, y_test = _load_data(test_f)

    return x_test, y_test

