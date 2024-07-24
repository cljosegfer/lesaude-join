
import pandas as pd
import h5py

from torch.utils.data import Dataset

class CODEtest(Dataset):
    def __init__(self, hdf5_path = '/home/josegfer/datasets/code/data/codetest/data/ecg_tracings.hdf5', 
                 metadata_path = '/home/josegfer/datasets/code/data/codetest/data/annotations/gold_standard.csv',
                 tracing_col = 'tracings', output_col = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']):
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.metadata = pd.read_csv(metadata_path)

        self.tracing_col = tracing_col
        self.output_col = output_col
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return {'x': self.hdf5_file[self.tracing_col][idx],
                'y': self.metadata[self.output_col].loc[idx].values,}