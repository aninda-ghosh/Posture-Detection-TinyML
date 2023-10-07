import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from tqdm import tqdm

class PostureDetectionDataset(tf.keras.utils.Sequence):
    def __init__(self, root, batch_size=1, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle

        
        files = glob.glob(root + '*.csv')

        print('Found {} files in Folder {}'.format(len(files), root))

        #use tqdm to read files
        self.data = []

        for file in tqdm(files):
            data = pd.read_csv(file)
            self.data.append(data)
    
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = []
        batch_labels = []
        batch_windows = []

        for i in batch_indexes:
            window = self.data[i]['window'].values[0]
            label = self.data[i][['supine', 'prone', 'side', 'sitting', 'unknown']].values[0]

            data = []
            data.append(self.data[i]['sens_x'].values)
            data.append(self.data[i]['sens_y'].values)
            data.append(self.data[i]['sens_z'].values)

            data = np.array(data, dtype=np.float32)
            label = np.array(label, dtype=np.float32)

            batch_data.append(data)
            batch_labels.append(label)
            batch_windows.append(window)


        batch_data = np.array(batch_data)
        batch_data = np.swapaxes(batch_data, 1, 2)

        batch_labels = np.array(batch_labels)

        return batch_data, batch_labels, batch_windows

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))