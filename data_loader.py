import numpy as np
import glob
import os

n_fold = 70

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

# partition data for k-fold cross validation
def load_split(fold_idx):
    assert 10 > fold_idx >= 0
    path = os.getenv('data_path', '../reg_data/*.npz')
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

    x_train, y_train = _load_data(train_f)
    x_test, y_test = _load_data(test_f)
    x_valid, y_valid = _load_data(valid_f)

    return x_train, x_valid, x_test, y_train, y_valid, y_test



