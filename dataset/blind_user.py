import os
import numpy as np
import pandas as pd


base_path = '/Users/prerna/Documents/gesture modeling/user_study/'
users = [user for user in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, user))]


target_sr = 20
seq_len = 120
curr_sr = 50



import numpy as np

def down_sample(data, target_sr, curr_sr, seq_len=120):
    factor = int(curr_sr / target_sr)
    data = data[::factor]
    
    total_pad = seq_len - len(data)
    if total_pad % 2 == 0:
        pad_start = pad_end = total_pad // 2
    else:
        pad_start = total_pad // 2
        pad_end = total_pad - pad_start
    data = np.pad(data, ((pad_start, pad_end), (0, 0)), 'constant')
    
    return data[:seq_len, :]



def preprocess(path, path_save, version, raw_sr=50, target_sr=20, seq_len=20):
    user_idx = 0
    labels = []
    data = []
    
    for user in users:
        user_path = os.path.join(base_path, user, 'Watch', 'cropped')
        if os.path.exists(user_path):
            files = os.listdir(user_path)
            for file in files:
                if(file.endswith(".csv")):
                    print("file", file)
                    trial_idx = file.split('_')[1]
                    gesture_idx = file.split('_')[0]
                    df = pd.read_csv(os.path.join(user_path, file))
                    if (len(df) < 50):
                        continue
                    print("old shape", df.shape)
                    try:
                        df = down_sample(df, target_sr, curr_sr)
                        print("new shape", df.shape)
                        print("-------------------")
                        data.append(df)
                        label = np.array([[int(gesture_idx), user_idx]])
                        label = np.tile(label, (seq_len, 1))
                        labels.append(label)
                    except:
                        print("error in file", file)
        user_idx += 1


    data = np.stack(data, axis=0)

    print("shape of data", np.shape(data))
    print("shape of labels", np.shape(labels)) 

    np.save("/Users/prerna/Documents/LIMU-BERT-blind-users/dataset/blind_user/data_20_120.npy", np.array(data))
    np.save("/Users/prerna/Documents/LIMU-BERT-blind-users/dataset/blind_user/label_20_120.npy", np.array(labels))
    return data, labels


# activity, user
path_save = r'blind_user'
version = r'20_120'
DATASET_PATH = r'/Users/prerna/Documents/gesture modeling/user_study/'

data, label = preprocess(DATASET_PATH, path_save, version, target_sr=20, seq_len=120)

