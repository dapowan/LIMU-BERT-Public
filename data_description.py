import numpy as np

data = np.load('dataset/hhar/data_20_40.npy')
print(data.shape)
print("mean of data", np.mean(data))