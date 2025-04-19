import torch
import pickle
import copy
import numpy as np
import os


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, n_seqs=10):
        self.device = cfg.device

        all_data = pickle.load(
            open(
                os.path.join(
                    cfg.parent_dir,
                    'flybrain_data.pkl',
                ),
                "rb",
            )
        )
        seqs = torch.argmax(
            torch.from_numpy(copy.deepcopy(all_data[f"{split}_data"])), dim=-1
        )
        clss = torch.argmax(
            torch.from_numpy(copy.deepcopy(all_data[f"y_{split}"])), dim=-1
        )

        self.num_cls = all_data[f"y_{split}"].shape[-1]
        self.x = seqs if n_seqs == -1 else seqs[:n_seqs]
        self.y = clss if n_seqs == -1 else clss[:n_seqs]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.y[idx].to(self.device)


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, x, device, y=None):
        if isinstance(x, np.ndarray):
            self.x = torch.tensor(x, dtype=torch.long)
        elif isinstance(x, torch.Tensor):
            self.x = x
        else:
            raise NotImplementedError
        if y is not None:
            assert y.shape[0] == self.x.shape[0]
            if isinstance(y, np.ndarray):
                self.y = torch.tensor(y, dtype=torch.long)
            elif isinstance(y, torch.Tensor):
                self.y = y
            else:
                raise NotImplementedError
        else:
            self.y = torch.full_like(self.x, -1)
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.y[idx].to(self.device)
