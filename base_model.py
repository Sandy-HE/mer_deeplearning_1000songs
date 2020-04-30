from keras.layers import *
from keras.models import Model


def build_model():
    inputs = Input(shape=(22050, 1))
    model = Conv1D(filters=8, kernel_size=220, strides=110, activation='relu', input_shape=(22050, 1))(inputs)
    model = BatchNormalization()(model)
    model = TimeDistributed(Dense(16, activation='relu'))(model)
    model = Dropout(0.25)(model)
    model = Bidirectional(CuDNNGRU(8))(model)
    outputs = MaxoutDense(2)(model)
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()
ckpt_pattern = 'fold_{}_base.hdf5'
