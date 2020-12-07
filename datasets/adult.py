import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

from sklearn import preprocessing


class DatasetAdult(Dataset):

    def __init__(self, file_path, train=True, val_split=0.2, seed=42, transform=None):
        self.fulldata = pd.read_csv(file_path)
        self.transform = transform
        self.train = train

        # normalize dataset
        x = self.fulldata.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.fulldata = pd.DataFrame(x_scaled, columns=self.fulldata.columns)

        train = self.fulldata.sample(frac=1 - val_split, random_state=seed)  # random state is a seed value
        test = self.fulldata.drop(train.index)

        if self.train:
            self.data = train
        else:
            self.data = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dataPoint = self.data.iloc[index, :-1].values.astype(np.float32)
        label = self.data.iloc[index, -1]

        if self.transform is not None:
            dataPoint = self.transform(dataPoint)

        return dataPoint, label
