# Single-trace side-channel attacks on MAYO exploiting leaky modular multiplication

This repository contains the training and test scripts for the paper "Single-trace side-channel attacks on MAYO exploiting leaky modular multiplication" (https://eprint.iacr.org/2024/1850).

The traces and trained models are available here: https://kth-my.sharepoint.com/:u:/g/personal/jendral_ug_kth_se/ESjXGUyOr2tAiRnP01vDxlEB8tlM1VBXhB-YJ4dM1W1LEQ?e=OLAdha

## First Attack: gf16_madd_bitsliced

This attack uses the ```train_madd.py``` and ```test_madd.py``` scripts and the ```madd/model.keras``` model.
The training dataset contains 10,000 traces, each containing 57 segments for each of the 57 entries of O that can be recovered.
The test dataset contains 1,000 traces with the same structure.

The ```test_madd.py``` script implements the enumeration method described in the paper based on the maximum predicted class probabilities.

## Second Attack: mul_f and decode

This attack uses the ```train_decode.py```, ```train_mat_mul.py``` and ```test_combined.py``` scripts and the ```decode/model.keras``` and ```mat_mul/model.keras``` models.
The training dataset for the decode operation contains 10,000 traces. The first 12,000 points of each trace cover the first invocation of the decode operation, the second 12,000 points cover the second invocation.
Each section covers the decoding of all 58 * 8 entries of O. For the attack, we extract only one column of entries (i.e. 58 segments) to train the model, though in practice, all entries could be used.
The test dataset contains 1,000 traces with the same structure.

The training dataset for the mat_mul operation is already split into the individual segments and contains 580,000 traces. Each trace covers 9 multiplications with a length of 110 points involving the same entry of O concatenated to each other.
The test dataset contains 1,000 traces, where each trace covers the multiplication of all 58 entries with all 9 values of x.

The ```test_combined.py``` script combines information from two models to implement the enumeration method described in the paper.
