# Config
## Model Config
[`limu_bert.json`](./limu_bert.json)
```
"base_v1":
{
    "feature_num" : 6, the input feature number, 6 represents accelerometer and gyroscope, S_{dim}
    "hidden": 72, the representation dimension, H_{dim}
    "hidden_ff": 144, feed forward dimension, S_{dim}
    "n_layers": 4, cross-layer parameter, R_{num}
    "n_heads": 4, number of attention head, A_{num}
    "seq_len": 120, window size or sequence length, L
}
```
[`classifier.json`](./classifier.json)
```
"gru_v1":
{
    "seq_len": 20, window size or sequence length, L
    "input": 6, the input feature number, 6 represents accelerometer and gyroscope, S_{dim}
    "num_rnn": 2, number of gru layers
    "num_layers": [2, 1], number of sub gru layers
    "rnn_io": [[6,20], [20, 10]], input and output or gru layers
    "num_linear": 1, number of fully-connected layers
    "linear_io": [[10, 3]], input and output of fully-connected layers
    "activ": false, use activation function or not
    "dropout": true, use drop out or not
}
```

## Training Config

[`pretrain.json`](./pretrain.json) and [`train.json`](./train.json) have the same structure. 
Note that they must have the same random seed.
```
{
    "seed": 3431, random seed
    "batch_size": 128, batch size
    "lr": 1e-3, learning rate
    "n_epochs": 2, number of epochs
    "warmup": 0.1, not applicable
    "save_steps": 1000,  not applicable
    "total_steps": 200000000, not applicable
}
```

[`mask.json`](./mask.json)
```
{
    "mask_ratio": 0.15, mask proportion of each sequence
    "mask_alpha": 6, not applicable
    "max_gram": 10, maximal number of masked subsequence
    "mask_prob": 0.8, the probablity of masking a sequence
}

```