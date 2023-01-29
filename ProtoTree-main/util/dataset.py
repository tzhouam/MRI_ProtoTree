import torch
import os
import pandas as pd
from joblib import delayed,Parallel
from util.hyper import Hyper
# def load_label(file,patient)
class Dataset(torch.utils.data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, root):
        'Initialization'
        # self.labels = labels
        list_IDs=sorted(os.listdir(root))
        self.labels=[]
        self.root=root
        self.list_IDs = []
        for i in list_IDs:
            if '_i' in i:
                continue
            self.list_IDs.append(i)
        a = pd.ExcelFile('./util/Clinical_and_Other_Features.xlsx')
        self.data = pd.read_excel(a, 'Data')
        f = self.data.iloc[0, :]
        self.data = self.data.iloc[2:, :]
        self.data.columns = f
        self.data = self.data.reset_index()
        for i in self.list_IDs:
            if '_i' in i:
                continue
            else:
                for j in range(922):
                    if self.data['Patient ID'][j] in i:
                        self.labels.append(int(self.data[Hyper().attribute][j]))
                        break
        # for i in self.list_IDs:
        #     if '_i' in i:
        #         self.labels.append(0)
        #     else:
        #         self.labels.append(1)
        assert len(self.labels)==len(self.list_IDs)
        # print(self.labels[:25])
        # print(self.list_IDs[:25])
        self.labels=torch.tensor(self.labels)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.root + ID)
        y = self.labels[index]

        return X, y