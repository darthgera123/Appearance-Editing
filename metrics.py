import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from skimage import metrics
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pinecone_dr.xml')

    args = parser.parse_args()

    img_list = os.listdir(args.dir)

    global_psnr = 0.0
    global_ssim = 0.0
    idx = 0

    for img_name in img_list:
        if 'gt' in img_name:
            img_name_output = img_name.replace('gt', 'output')

            gt = cv2.imread('%s/%s' % (args.dir, img_name))
            output = cv2.imread('%s/%s' % (args.dir, img_name_output))

            psnr = metrics.peak_signal_noise_ratio(gt, output)
            ssim = metrics.structural_similarity(gt, output, multichannel=True)

            global_psnr += psnr
            global_ssim += ssim

            idx += 1
    
    global_psnr /= idx
    global_ssim /= idx

    print('PSNR: %f' % global_psnr)
    print('SSIM: %f' % global_ssim)