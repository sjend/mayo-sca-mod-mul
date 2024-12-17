import tensorflow as tf
import numpy as np
import hashlib

def _load_dataset():
    raw_traces = np.load('madd/test_traces.npy')
    raw_labels = np.load('madd/test_labels.npy')

    labels = []
    for label in raw_labels:
        O_bytes = [int(byte) for byte in hashlib.shake_256(bytearray([int(x) for x in label])).digest(16+232)[16:]] # skip seed_pk

        for byte in O_bytes[4::4]: # first set of coefficients is not included
            labels.append(byte >> 4)

    traces_stacked = []
    for trace in raw_traces:
        for i in range(57):
                    traces_stacked.append(trace[9680 + i * 148 : 9820 + i * 148])
    
    traces = np.array(traces_stacked)
    del raw_traces
    del raw_labels

    traces = (traces - np.mean(traces, axis=0)) / np.std(traces, axis=0)

    return (traces, labels), traces.shape[1]

model = tf.keras.models.load_model('madd/model.keras')

dataset, _ = _load_dataset()
predictions = model.predict(dataset[0])

correct = 0
total_enum = 0
max_enum = 0

num_traces = len(dataset[1]) // 57

for i in range(num_traces):
    ps = np.argmax(predictions[i * 57: (i + 1) * 57], axis=1)
    prob = np.max(predictions[i * 57: (i + 1) * 57], axis=1)
    ls = dataset[1][i * 57: (i + 1) * 57]

    num_enum = 4 # Can't get first element, so make sure to include enumeration cost for it
    p = []
    for i in range(57):
        if prob[i] < 0.99999:
            num_enum += 4
            p.append(ls[i])
        else:
            p.append(ps[i])

    p = np.array(p)

    if np.array_equal(p, ls) and num_enum <= 32:
        total_enum += num_enum
        max_enum = max(max_enum, num_enum)
        correct += 1

print(f'Accuracy: {correct / 1000}, Enum: {total_enum / correct} (max: {max_enum})')