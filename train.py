import keras
from keras.optimizers import Adam
from common import *
from data_loader import *

fold_idx = int(os.getenv('fold'))
print('running on fold', fold_idx)

ckpt_path = ckpt_pattern.format(fold_idx)

adam = Adam(lr=LR)

es = keras.callbacks.EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=10, restore_best_weights=True)
ckpt = keras.callbacks.ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True,
                                       monitor='val_mse', mode='min')

model.compile(loss='mse', optimizer=adam, metrics=['mse'])
model.summary()

x_train, x_valid, x_test, y_train, y_valid, y_test = load_split(fold_idx)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
                    validation_data=(x_valid, y_valid), callbacks=[es, ckpt])

prediction = model.predict(x_test, batch_size=BATCH_SIZE)

evaluate(prediction, y_test)
