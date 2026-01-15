import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
import os
        
class AMP_Dataset(data.Dataset):
    def __init__(self, data_file, embeddings_fpath=None, des_fpath=None, stc_fpath=None):
        self.data_df = pd.read_csv(data_file)

        self.all_embeddings = h5py.File(embeddings_fpath,'r') if embeddings_fpath else None
        self.des_info = h5py.File(des_fpath,'r') if stc_fpath else None
        self.stc_info = h5py.File(stc_fpath,'r') if stc_fpath else None
        self.labels = torch.tensor(self.data_df['Labels'].tolist()) # for Distributed data sampler

        self.num_bac = len(self.data_df['bacterium_id'].unique())
        self.bacterium_id = torch.tensor(self.data_df['bacterium_id'].tolist())
        
    def __len__(self):
        return len(self.data_df)

    def _load_embeddings(self, protein_name):
        emb = self.all_embeddings[protein_name][:]
        return emb

    def _get_des_features(self, protein_name):
        features = self.des_info[protein_name][:]
        return features

    def _get_stc_features(self, protein_name):
        features = self.stc_info[protein_name][:]
        return features

    def __getitem__(self, idx):
        data_item = self.data_df.iloc[idx]
        mic = data_item['MIC']
        mic = torch.tensor(mic, dtype=torch.float)
        seq =  data_item['Sequence']
        label = data_item['Labels']
        bacterium_id = data_item['bacterium_id']
        bacterium = data_item['Bacterium']
        tensor_label = torch.tensor(label, dtype=torch.float)

        input_data = {'seq':seq,'label':tensor_label,'mic':mic, 'bacterium_id':bacterium_id, 'bacterium':bacterium}
        if self.all_embeddings:
            emb = self._load_embeddings(seq)
            input_data['emb'] = emb
        if self.des_info:
            des = self._get_des_features(seq)
            input_data['des'] = des
        if self.stc_info:
            stc = self._get_stc_features(seq)
            input_data['stc'] = stc

        return input_data

def build_dataset(cfg,type):
    """
    Build datasets according to cfg
    type: train, test, or val
    """
    data_file = cfg.data[type].datafile
    embeddings_fpath = cfg.data[type].embeddings_fpath
    des_fpath = cfg.data[type].des_fpath
    stc_fpath = cfg.data[type].stc_fpath

    dataset = AMP_Dataset(data_file,embeddings_fpath,des_fpath,stc_fpath)
    return dataset