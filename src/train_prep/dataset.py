
import numpy as np
import pandas as pd

import torch
from torch.utils.data import  Dataset
from ..utils.constants import PLPConstants

class TabularDataset(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame):
        
        self.target=data['outcomeCount'].astype(int)
        self.features=data.drop(columns=PLPConstants.basics_outcome,axis=1).values
    
    def __getitem__(self, idx):
        data = self.features[idx]
        target = self.target[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)