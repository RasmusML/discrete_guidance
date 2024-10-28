import numpy as np
import torch
import torchvision


class DictDataset(torch.utils.data.Dataset):
    """
    Define a custom dataset that returns a dictionary with dictionary-keys 
    'x' and 'y' if sliced where the dictionary-values will correspond to the 
    sliced x and y data values.
    """
    def __init__(self, x, y=None):
        """
        Args:
            x (torch.tensor): 2D torch tensor of shape 
                (#datapoints, #x-features).
            y (torch.tensor or None): Torch tensor of shape 
                (datapoints, #y-features) or 
                case y will be made a 1D torch tensor )
        
        """
        # Assign x and y to the corresponding class attributes
        self.x = x
        self.y = y

    def __len__(self):
        """ Return the number of datapoints. """
        # Remark: self.x should have shape (#datapoints, #x-features)
        return self.x.shape[0]

    def __getitem__(self, ix):
        """ Implement slicing. """
        # Cast ix to a list if it is a tensor
        if torch.is_tensor(ix):
            ix = ix.tolist()        

        # Return a dictionary of the data
        if self.y is None:
            return {'x': self.x[ix]}
        else:
            return {'x': self.x[ix], 'y': self.y[ix]}


class DiscreteCIFAR10(DictDataset):
    def __init__(self, cfg):
        """
        Adapted from https://github.com/andrew-cr/tauLDR/blob/main/lib/datasets/datasets.py
        Convert to type DictDataset        
        """
        super().__init__(x=None, y=None)  # Initialize with placeholders
        device = cfg.device

        self.cifar10 = torchvision.datasets.CIFAR10(
            root=cfg.data.root, 
            train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.cifar10.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.cifar10.targets))

        # If training, use random flip as data augmentation
        self.random_flips = cfg.data.train
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()

        self.x = self.data.to(device).view(-1, 3, 32, 32)
        self.y = self.targets.to(device)

    def __getitem__(self, index):
        # item is a Dict containing the single datapoint
        item = super().__getitem__(index)
        if self.random_flips:
            item['x'] = self.flip(item['x'])
        return item