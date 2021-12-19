import torch, os, sys, cv2, json, argparse, random, glob, struct, math, time
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import numpy as np 
import os.path as osp

import pyexr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./', help='')
    
    args = parser.parse_args()

    frames_list = os.listdir(args.data_dir+'train/frames/')
    random.shuffle(frames_list)
    num_test = int(0.1 * len(frames_list))

    for idx, name_og in enumerate(frames_list):
        if idx == num_test:
            break
        
        name = name_og.replace('.png', '').replace('.jpg', '').replace('.JPG', '').replace('image', '')
        
        os.system('mv %s/train/frames/%s %s/test/frames/' % (args.data_dir, name_og, args.data_dir))
        os.system('mv %s/train/uv/%s.npy %s/test/uv/' % (args.data_dir, name, args.data_dir))
        os.system('mv %s/train/uv_png/%s.png %s/test/uv_png/' % (args.data_dir, name, args.data_dir))
        os.system('mv %s/train/sh/%s.npy %s/test/sh/' % (args.data_dir, name, args.data_dir))
        os.system('mv %s/train/extrinsics/%s.npy %s/test/extrinsics/' % (args.data_dir, name, args.data_dir))