import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3


if os.getenv('model') == 'my':
    from my_model import model, ckpt_pattern
    print('using my model')
elif os.getenv('model') == 'base':
    from base_model import model, ckpt_pattern
    print('using base model')
else:
    raise ValueError('Need to specify model!')


def evaluate(y_pred, y_true):
    arousal_pred = np.squeeze(y_pred[:, :1])
    arousal_true = np.squeeze(y_true[:, :1])
    valence_pred = np.squeeze(y_pred[:, 1:])
    valence_true = np.squeeze(y_true[:, 1:])

    r2_arousal = r2_score(arousal_true, arousal_pred)
    r2_valence = r2_score(valence_true, valence_pred)
    print('r2_arousal: ', r2_arousal)
    print('r2_valence: ', r2_valence)

    r2 = r2_score(y_true, y_pred)
    print('r2_overall: ', r2)

    mse_arousal = mean_squared_error(arousal_true, arousal_pred)
    mse_valence = mean_squared_error(valence_true, valence_pred)
    print('mse_arousal: ', mse_arousal)
    print('mse_valence: ', mse_valence)

    mse = mean_squared_error(y_true, y_pred)
    print('mse_overall: ', mse)

    rmse_arousal = mean_squared_error(arousal_true, arousal_pred, squared=False)
    rmse_valence = mean_squared_error(valence_true, valence_pred, squared=False)
    print('rmse_arousal: ', rmse_arousal)
    print('rmse_valence: ', rmse_valence)

    # r, p_value = pearsonr(arousal_true, np.squeeze(arousal_pred))
    # print(r, p_value)

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print('rmse_overall: ', rmse)
