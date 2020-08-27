#update on 19/08
from data_loader import load_split_for_test
from common import *
import numpy as np

all_pred = []
all_true = []

for fold_idx in range(10):
    model.load_weights(ckpt_pattern.format(fold_idx))
    x_test, y_test = load_split_for_test(fold_idx)
    all_true.append(y_test)
    y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
    all_pred.append(y_pred)

all_true_arr = np.concatenate(all_true)
all_pred_arr = np.concatenate(all_pred)

evaluate(all_pred_arr, all_true_arr)
