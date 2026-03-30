import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class AgeDataset(Dataset):
    """PyTorch Dataset for age prediction from images"""
    
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns ['path', 'age']
            transform: torchvision transforms to apply to images
        """
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        
        return img, age
