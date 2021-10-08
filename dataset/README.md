# Dataset
Example:
```
"hhar_20_120": {
    "sr": 20, sampling rate
    "seq_len": 120, window size or sequence length
    "dimension": 6, input features, 6 represents accelerometer and gyroscope
    "activity_label_index": 2, activity label position in the label.npy
    "activity_label_size": 6, number of activities
    "activity_label": [
        "bike", "sit", "downstairs", "upstairs", "stand", "walk"
    ], names of activities
    "user_label_index": 0, user label position in the label.npy
    "user_label_size": 9, number of users
    "model_label_index": 1, model label position in the label.npy
    "model_label_size": 3, number of models
    "size": 9166, sample size
}
```