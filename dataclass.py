import torch.utils.data as data

class DataClass(data.Dataset):

    def __init__(self, X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = self.X[index]
        label = self.y[index]
        return sample, label