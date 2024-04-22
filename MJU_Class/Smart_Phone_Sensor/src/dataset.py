# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# !!! 수정 금지 !!!
VAR_LIST = ['ACC_x', 'ACC_y', 'ACC_z', 
            'GRVT_x', 'GRVT_y', 'GRVT_z', 
            'MAG_x', 'MAG_y', 'MAG_z', 
            'ORT_a_cos', 'ORT_a_sin', 
            'ORT_p_cos', 'ORT_p_sin', 
            'ORT_r_cos', 'ORT_r_sin']  

# !!! 수정 금지 !!!
VAR_DICT = {var: i for i, var in enumerate(VAR_LIST)}



class SensorDataset(Dataset):
    def __init__(self, 
                 in_columns,
                 mode='train', 
                 data_dir='data',
                 data_fname='da23_sensor_data.npz'):
        
        data_fname = data_fname.replace('.npz', f'({mode}).npz')
        fname = os.path.join(data_dir, data_fname)
        if not os.path.exists(fname):
            raise FileNotFoundError(f'{fname} does not exist')
        
        var_idx = [VAR_DICT[var] for var in in_columns]

        data = np.load(fname)
        if mode in ['train', 'valid', 'test']:
            self.x = torch.from_numpy(data['X'][:, var_idx]).float()
            self.y = torch.from_numpy(data['Y']).float()
        else:
            raise ValueError(f'Invalid mode {mode}')

        del data


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



if __name__ == "__main__":

    in_columns = ['ACC_x', 'ACC_y', 'ACC_z', 
                  'ORT_a_cos', 'ORT_a_sin', 
                  'ORT_p_cos', 'ORT_p_sin', 
                  'ORT_r_cos', 'ORT_r_sin']
    mode = 'valid'
    data_dir = 'data'
    data_fname = 'da23_sensor_data.npz'

    dataset = SensorDataset(in_columns=in_columns,
                            mode=mode,
                            data_dir=data_dir,
                            data_fname=data_fname)
    
    print(f'dataset length : {len(dataset)}')
    print(f'x.shape : {dataset.x.shape}')
    print(f'y.shape : {dataset.y.shape}')

    import matplotlib.pyplot as plt
    sample_idx = 0
    sample_x = dataset.x[sample_idx].T.numpy()
    sample_x = (sample_x - sample_x.mean(axis=0)) / sample_x.std(axis=0)  # normalize
    sample_y = 'pos' if dataset.y[sample_idx].numpy() else 'neg'
    plt.plot(sample_x)
    plt.title(f'{mode} data : {sample_idx}-th sample : y={sample_y}')
    plt.legend(in_columns)
    plt.show()

