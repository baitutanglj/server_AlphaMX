from torch_geometric.data import Dataset
import os
import torch


class mols_dataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, pre_filter=None):
        super(mols_dataset, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.root_dir = root_dir
        self.transform = transform
        self.pyg_data = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.pyg_data)

    def __getitem__(self, index):
        data_name = self.pyg_data[index]
        data_path = str(self.root_dir)+data_name
        pyg_data = torch.load(data_path)
        return pyg_data  
    
    def name(self, index):
        data_name = self.pyg_data[index]
        return data_name