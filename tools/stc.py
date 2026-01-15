import h5py
import pandas as pd
import os
from tqdm import tqdm

import torch

bac = 'bac11'
fname_list = ['train.csv', 'val.csv', 'test.csv']
base_path = ''

sequence_list = pd.DataFrame(columns=['Sequence', 'Structure'])
for fname in fname_list:
    fpath = os.path.join(base_path, fname)
    df = pd.read_csv(fpath)
    df = df[['Sequence', 'Structure']]
    print(len(df))
    sequence_list = pd.concat([sequence_list, df], ignore_index=True)

# padding
max_len = max(sequence_list['Sequence'].str.len())
print('max_len:', max_len)

outdir = ''
fname = f'{bac}.h5'
ss_code = {'C': 1,'E': 2, 'H': 3 }
os.makedirs(outdir,exist_ok=True)
with h5py.File(os.path.join(outdir,fname), 'w') as hf:
    for row in tqdm(sequence_list.iterrows()):
        sequence = row[1]['Sequence']
        secondary_struct = row[1]['Structure']
        structure_code = [ss_code[s] for s in secondary_struct]
        structure_code.extend([0 for i in range(max_len-len(structure_code))])
        # print(len(structure_code))
        structure_code = torch.tensor(structure_code)
        hf.create_dataset(sequence, data=structure_code)