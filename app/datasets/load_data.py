import torch
from torch.utils.data import Dataset  
from modelNet import netTransform

class PTDataset(Dataset):
    """
    Custom Dataset loader for PyTorch .pt files.
    Expects the .pt file to contain a tuple (data, targets).
    """

    def __init__(self, pt_file):
        """
        Args:
            pt_file (str): Path to the .pt file containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data, self.targets = torch.load(pt_file)
        self.transform = netTransform

    def __len__(self):
        return len(self.data)   # ← ESSENCIAL

    def __getitem__(self, idx):
        x = self.data[idx].float()
        y = self.targets[idx]
        
        # se for tensor bruto 28x28 -> adiciona canal
        if isinstance(x, torch.Tensor) and x.ndim == 2:
            x = x.unsqueeze(0)

        # só aplica ToTensor() se não for tensor
        if self.transform:
            try:
                x = self.transform(x)
            except TypeError:
                pass

        return x, y