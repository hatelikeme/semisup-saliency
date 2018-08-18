
# coding: utf-8

# In[1]:


from sal import rbd
from sal import mbd
import os

import numpy as np

from tqdm import tqdm

from PIL import Image

import multiprocessing


TRAIN_DATA_DIR = 'msra10k'

MASKS_DIR = 'mbd-masks-msra'

def gen_save_sal(path):
    try:
        smap = mbd.get_saliency_mbd(path).astype('uint8')
        smap = rbd.binarise_saliency_map(smap, method='adaptive').astype('uint8') * 255
        smap = Image.fromarray(smap, 'L')
        smap.save(os.path.join(MASKS_DIR ,os.path.basename(path).replace('.jpg', '')  + '.png'))
        return 0
    except:
        return 1

def loader():
    pool = multiprocessing.Pool(60)
    if not os.path.exists(MASKS_DIR):
        os.mkdir(MASKS_DIR)
    paths = [os.path.join(TRAIN_DATA_DIR, imname) for imname in os.listdir(TRAIN_DATA_DIR) if '.jpg' in imname]
    fs = sum(tqdm(pool.imap_unordered(gen_save_sal, paths), total=len(paths)))
    pool.close()
    pool.terminate()
    
if __name__ == '__main__':
    loader()
