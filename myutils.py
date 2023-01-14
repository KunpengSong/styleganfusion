import os
import torch
from tqdm import tqdm
from shutil import copy
import pdb
st = pdb.set_trace


def auto_saveCode(save_folder):    
    try:
        code_path = os.path.join(save_folder, 'code/')
        os.makedirs(code_path,exist_ok=True)
        for f in os.listdir('.'):
            if '.py' in f:
                copy(f, os.path.join(code_path,f))
            if '.sh' in f:
                copy(f, os.path.join(code_path,f))
    except:
        pass
    folders = ['model']
    for i in folders:
        try:
            code_path = os.path.join(code_path, i)
            os.makedirs(code_path,exist_ok=True)
            for f in os.listdir(i):
                if '.py' in f:
                    copy(os.path.join(i,f), os.path.join(code_path,f))
        except:
            pass

def is_weight_module(m, n):
    return hasattr(m, 'weight') and isinstance(getattr(m, 'weight', None), torch.Tensor)



