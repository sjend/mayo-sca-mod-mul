import tensorflow as tf
import numpy as np
import hashlib

def _load_dataset():
    raw_labels = np.load('mat_mul/test_labels.npy')
    raw_traces_decode = np.load('decode/test_traces.npy')
    raw_traces_mat_mul = np.load('mat_mul/test_traces.npy')

    labels = []
    for label in raw_labels:
        O_bytes = [int(byte) for byte in hashlib.shake_256(bytearray([int(x) for x in label])).digest(16+232)[16:]] # skip seed_pk

        for byte in O_bytes[::4]:
            value = byte & 0xf
            labels.append(value)

    traces_stacked_decode = []
    for trace in raw_traces_decode:
        for i in range(0, 232, 4):
            traces_stacked_decode.append(np.concatenate((trace[930 + i * 44 : 985 + i * 44], trace[12930 + i * 40 : 12985 + i * 40])))
    
    traces_decode = np.array(traces_stacked_decode)
    del raw_traces_decode

    traces_stacked_mat_mul = []
    for trace in raw_traces_mat_mul:
        for i in range(58):
            offsets = [0, 39756, 79584, 119336, 159160, 198912, 238736, 278488, 318312]
            traces_stacked_mat_mul.append(np.concatenate(tuple(trace[1140 + i * 672 + offsets[j] : 1250 + i * 672 + offsets[j]] for j in range(9))))

    traces_mat_mul = np.array(traces_stacked_mat_mul)
    del raw_traces_mat_mul
    del raw_labels

    traces_decode = (traces_decode - np.mean(traces_decode, axis=0)) / np.std(traces_decode, axis=0)
    traces_mat_mul = (traces_mat_mul - np.mean(traces_mat_mul, axis=0)) / np.std(traces_mat_mul, axis=0)

    return traces_decode, traces_mat_mul, labels

model_decode = tf.keras.models.load_model('decode/model.keras')
model_mat_mul = tf.keras.models.load_model('mat_mul/model.keras')

traces_decode, traces_mat_mul, labels = _load_dataset()

predictions_decode_p = model_decode.predict(traces_decode)
predictions_mat_mul_p = model_mat_mul.predict(traces_mat_mul)

correct = 0
correct_hw = 0
correct_bits = 0
total_enum = 0
max_enum = 0

num_traces = len(labels) // 58

for i in range(num_traces):
    p_hw = np.argmax(predictions_decode_p[i * 58:(i + 1) * 58], axis=1)
    prob_hw = np.max(predictions_decode_p[i * 58:(i + 1) * 58], axis=1)
    p_bits = np.argmax(predictions_mat_mul_p[i * 58:(i + 1) * 58], axis=1)
    prob_bits = np.max(predictions_mat_mul_p[i * 58:(i + 1) * 58], axis=1)

    l = labels[i * 58:(i + 1) * 58]

    p = []
    num_enum = 0
    for i in range(58):
        bits = int(p_bits[i])
        hw = p_hw[i]
        has_enum_bit_2 = False

        if (prob_hw[i] > 0.99999 and prob_bits[i] > 0.99999) or num_enum >= 32:
            if int.bit_count(bits) == hw:
                bit_2 = 0
            elif int.bit_count(bits) == hw - 1:
                bit_2 = 1
        else:
            num_enum += 1
            bit_2 = (l[i] & 0b100) >> 2
            has_enum_bit_2 = True

        if int.bit_count(l[i]) == hw:
            correct_hw += 1
        
        if (((l[i] & 0b1000) >> 1) | (l[i] & 0b11)) == bits:
            correct_bits += 1

        if int.bit_count(bits) not in [hw, hw - 1] or (prob_bits[i] < 0.99999 and num_enum < 28):
            if has_enum_bit_2:
                num_enum += 3
            else:
                num_enum += 4
            p.append(l[i])
            continue

        p.append(bits & 0b11 | (bit_2 << 2) | ((bits & 0b100) << 1))

    p = np.array(p)

    if np.array_equal(p, l) and num_enum <= 32:
        total_enum += num_enum
        max_enum = max(max_enum, num_enum)
        correct += 1

print(f'Accuracy: {correct / num_traces}, HW: {correct_hw / (num_traces * 58)}, Bits: {correct_bits / (num_traces * 58)}, Enum: {total_enum / correct if correct > 0 else 0} (max: {max_enum})')