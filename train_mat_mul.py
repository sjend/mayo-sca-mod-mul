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

    # This should actually be 8 (2^3), but the provided models were accidentally trained with 9 classes
    # In practice, the redundant class is just ignored, so this doesn't matter
    output = Dense(9, kernel_initializer='he_uniform', activation='softmax')(x)

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
    traces = np.load('mat_mul/train_traces.npy') # Dataset is already cut+joined
    raw_labels = np.load('mat_mul/train_labels.npy')

    labels = []
    for label in raw_labels:
        O_bytes = [int(byte) for byte in hashlib.shake_256(bytearray([int(x) for x in label])).digest(16+232)[16:]] # skip seed_pk

        for byte in O_bytes[::4]:
            value = byte & 0xf
            labels.append((value & 0b11) | ((value >> 1) & 0b100)) # bit_2 doesn't leak, so can't be recovered like this

    labels = to_categorical(labels, num_classes=9) # Same as above, should be 8

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
tf.keras.models.save_model('mat_mul/new/model.keras')