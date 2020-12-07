import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch.utils.data import DataLoader, Dataset



class DatasetAdult_V2(Dataset):
    
    def __init__(self, file_path,val_split=0.2,train=True, seed=42,target=2):
        self.path = file_path
        self.fulldata = pd.read_csv(self.path)
        featureSet = ['age', 'education.num', 'hours.per.week', 'income', 'capital.total', 'Married',
                      'Never-married', 'Single', 'Husband', 'Not-in-family', 'Other-relative',
                      'Own-child', 'Unmarried', 'Wife', 'Blue_colloar', 'Military',
                      'Services', 'White_colloar']

        targetList = ['workclass_label', 'race_label', 'sex_label']

        trainset = self.fulldata.sample(frac=1 - val_split, random_state=seed)  # random state is a seed value
        testset = self.fulldata.drop(trainset.index)

        if train:
            self.X_set = trainset[featureSet]
            self.Y_set = trainset[targetList[target]]
        else:
            self.X_set = testset[featureSet]
            self.Y_set = testset[targetList[target]]
      
    def __len__(self):
        return len(self.X_set)

    def __getitem__(self, index):

        dataPoint = self.X_set.iloc[index,:].values.astype(np.float32)
        label = self.Y_set.iloc[index]

        return dataPoint, label
