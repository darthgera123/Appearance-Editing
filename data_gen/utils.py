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

import sh

import pyexr

def concentric_sample_disk():
    r1 = random.random() * 2.0 - 1.0
    r2 = random.random() * 2.0 - 1.0

    if r1 == 0.0 and r2 == 0.0:
        return np.array([0.0, 0.0])

    theta = 0.0
    r = 0.0
    if(abs(r1) > abs(r2)):
        r = r1
        theta = math.pi * r2 / (4 * r1)
    else:
        r = r2
        theta = (math.pi / 2) - (math.pi * r1 / (4 * r2))

    return np.array([math.cos(theta), math.sin(theta)]) * r

def cosine_sample_hemisphere():	
    disk_point = concentric_sample_disk()
    z = math.sqrt(max(0.0, 1 - disk_point[0]**2 - disk_point[1]**2))

    wi = np.array([disk_point[0], z, disk_point[1]])
    return wi

def uniform_sample_hemisphere():
    r1 = random.random()
    r2 = random.random()
    sqrt = math.sqrt(max(0.0, 1-r1**2))
    phi = 2*math.pi*r2

    wi = np.array([math.cos(phi)*sqrt, r1, math.sin(phi)*sqrt])
    return wi

def load_image(img, size):
    image_ref = cv2.imread(img)

    image_ref = cv2.resize(image_ref, size)
    image_ref = cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    image_ref = image_ref**2.2

    return image_ref