import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, ReLU
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import hashlib
    
def _create_model(input_size, learning_rate, epsilon):
    inputs = Input(shape=(input_size,))
    x = BatchNormalization()(inputs)
    x = ReLU()(x)
    x = Dense(128, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(64, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(32, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    output = Dense(5, kernel_initializer='he_uniform', activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    
    optimizer = Nadam(learning_rate=learning_rate, epsilon=epsilon)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary)
    return model

def _train_model(data, validation_split, batch_size, model, epochs, patience):  
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patience, restore_best_weights=True)
    history = model.fit(data[0], data[1],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es],
                        validation_split=validation_split,
                        verbose=1)
    return history

def _load_dataset():
    raw_traces = np.load('decode/train_traces.npy')
    raw_labels = np.load('decode/train_labels.npy')

    labels = []
    for label in raw_labels:
        O_bytes = [int(byte) for byte in hashlib.shake_256(bytearray([int(x) for x in label])).digest(16+232)[16:]]

        for byte in O_bytes[::4]:
            labels.append(int.bit_count(byte & 0xf))

    labels = to_categorical(labels, num_classes=5)

    traces_stacked = []
    for trace in raw_traces:
        for i in range(0, 232, 4):
            traces_stacked.append(np.concatenate((trace[930 + i * 44 : 985 + i * 44], trace[12930 + i * 40 : 12985 + i * 40])))

    traces = np.array(traces_stacked)
    del raw_traces
    del raw_labels

    traces = (traces - np.mean(traces, axis=0)) / np.std(traces, axis=0)

    shuffle_seed = np.random.default_rng().integers(2**32)

    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(traces)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(labels)

    return (traces, labels), traces.shape[1]

patience = 15
epochs = 100
batch_size = 1024
validation_split = 0.3
learning_rate = 0.01
epsilon = 1e-08

data, input_size = _load_dataset()

model = _create_model(input_size, learning_rate, epsilon)
_train_model(data, validation_split, batch_size, model, epochs, patience)
tf.keras.models.save_model(model, 'decode/new/model.keras')
