from keras.layers import *
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras import regularizers


def build_model():
    inputLayer = Input(shape = (22050, 1))

    convFine = Conv1D(filters=8, kernel_size=32, strides=8, padding='same',
                      activity_regularizer=regularizers.l2(1e-4),
                      kernel_initializer='he_normal',
                      activation='relu', name='fConv1')(inputLayer)
    convFine = BatchNormalization()(convFine)
    convFine = MaxPooling1D(pool_size=8, padding='same', name='fMaxP1')(convFine)
    #convFine = Dropout(rate=0.2, name='fDrop1')(convFine)

    convCoarse = Conv1D(filters=8, kernel_size=128, strides=32, padding='same',
                        #activity_regularizer=regularizers.l2(1e-5),
                      kernel_initializer='he_normal',
                        activation='relu', name='cConv1')(inputLayer)
    convCoarse = BatchNormalization()(convCoarse)
    convCoarse = MaxPooling1D(pool_size=2, padding='same', name='cMaxP1')(convCoarse)
    #convCoarse = Dropout(rate=0.2, name='cDrop1')(convCoarse)

    model = Add()([convFine, convCoarse])
    model = Dropout(rate=0.2)(model)
    model = Bidirectional(CuDNNLSTM(32, return_sequences=True))(model)
    model = Bidirectional(CuDNNLSTM(32))(model)
    model = Dropout(0.2)(model)

    outputs = Dense(2)(model)
    model = Model(inputs=inputLayer, outputs=outputs)
    return model


model = build_model()
ckpt_pattern = 'fold_{}_my.hdf5'
