
import h5py
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

class CODE():
    def __init__(self, hdf5_path = '/home/josegfer/datasets/code/output/code15.h5', 
                 metadata_path = '/home/josegfer/datasets/code/output/metadata.csv', 
                 texth5_path = '/home/josegfer/datasets/code/output/BioBERTpt_text_report.h5', 
                 val_size = 0.05, tst_size = 0.05):
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.metadata = pd.read_csv(metadata_path)
        self.texth5_file = h5py.File(texth5_path, 'r')

        self.val_size = val_size
        self.tst_size = tst_size

        self.trn_metadata, self.val_metadata, self.tst_metadata = self.split()

    def split(self, patient_id_col = 'patient_id'):
        patient_ids = self.metadata[patient_id_col].unique()

        num_trn = int(len(patient_ids) * (1 - self.tst_size - self.val_size))
        num_val = int(len(patient_ids) * self.val_size)

        trn_ids = set(patient_ids[:num_trn])
        val_ids = set(patient_ids[num_trn : num_trn + num_val])
        tst_ids = set(patient_ids[num_trn + num_val :])

        trn_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(trn_ids)].reset_index()
        val_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(val_ids)].reset_index()
        tst_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(tst_ids)].reset_index()
        self.check_dataleakage(trn_metadata, val_metadata, tst_metadata)

        return trn_metadata, val_metadata, tst_metadata

    def check_dataleakage(self, trn_metadata, val_metadata, tst_metadata, exam_id_col = 'exam_id'):
        trn_ids = set(trn_metadata[exam_id_col].unique())
        val_ids = set(val_metadata[exam_id_col].unique())
        tst_ids = set(tst_metadata[exam_id_col].unique())
        assert (len(trn_ids.intersection(val_ids)) == 0), "Some IDs are present in both train and validation sets."
        assert (len(trn_ids.intersection(tst_ids)) == 0), "Some IDs are present in both train and test sets."
        assert (len(val_ids.intersection(tst_ids)) == 0), "Some IDs are present in both validation and test sets."

class CODEsplit(Dataset):
    def __init__(self, database, metadata,
                 tracing_col = 'tracings', output_col = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], textfeatures_col = 'embeddings', 
                 exam_id_col = 'exam_id', h5_idx_col = 'h5_idx'):
        self.database = database
        self.metadata = metadata

        self.tracing_col = tracing_col
        self.output_col = output_col
        self.textfeatures_col = textfeatures_col

        self.exam_id_col = exam_id_col
        self.h5_idx_col = h5_idx_col
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return {'x': self.database.hdf5_file[self.tracing_col][self.metadata[self.h5_idx_col].loc[idx]],
                'y': self.metadata[self.output_col].loc[idx].values, 
                'exam_id': self.metadata[self.exam_id_col].loc[idx], 
                'h': self.database.texth5_file[self.textfeatures_col][self.metadata[self.h5_idx_col].loc[idx]]}