import torch
import numpy as np
import math
from TrajectorBatch import TrajectorBatch

class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, batch_size, device, batch_len, rollout_steps):
        
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len 
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, rollout_steps)
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size
        
    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self
        
    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size) 
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False )
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            return TrajectorBatch(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)
            
