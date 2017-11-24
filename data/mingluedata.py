from torch.utils import data

class MingLueData(data.Dataset):
    
    def __init__(self, ids, X, y):
        self.len = X.shape[0]
        self.ids = ids
        self.x_data = X
        self.y_data = y
    
    def __getitem__(self, index):
        return self.ids[index], self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len


class MingLueTestData(data.Dataset):
    
    def __init__(self, X):
        self.len = X.shape[0]
        self.x_data = X
    
    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len



