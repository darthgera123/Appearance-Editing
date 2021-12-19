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

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Color3f,Float
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, render_torch, write_bitmap, Adam
import time
import cv2, imageio
import numpy as np 

from utils import *

import tensorboardX

def save_opt(args, params, crop_size):
    write_bitmap('%s/optimized_textures/perfect_geometry/diffuse_opt.exr' % args.data_dir, params['obj_1.bsdf.reflectance.data'], (512, 512), write_async=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    parser.add_argument('--focal_length', type=str, default='28.0mm') # Default set to focal length of iPhone X, wide angle camera
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=2)

    args = parser.parse_args()

    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    init_tex = np.ones((512, 512, 3), dtype=np.float32) * 0.5
    imageio.imwrite('%s/optimized_textures/perfect_geometry/diffuse_opt.exr' % args.data_dir, init_tex)
    # cv2.imwrite('%s/optimized_textures/perfect_geometry/diffuse_opt.png' % args.data_dir, (init_tex*255.0).astype(np.uint8))

    writer = tensorboardX.SummaryWriter(logdir=args.data_dir+'/dr_tensorboard/perfect_geometry/')

    img_list = os.listdir('%s/video_frames' % args.data_dir)

    cnt = 0
    for epoch in range(0, 2):
        
        random.shuffle(img_list)
        for i, img_name in enumerate(img_list):
            identifier = img_name.replace('.png', '').replace('.jpg', '').replace('.JPG', '').replace('image', '')

            init_lr = 0.01

            gt = load_image('%s/video_frames/%s' % (args.data_dir, img_name), (args.img_width, args.img_height))
            mask = load_image('%s/video_frames_mask/%s.png' % (args.data_dir, identifier), (args.img_width, args.img_height))

            gt = gt * mask

            c = np.load('%s/video_frames_extrinsics/%s.npy' % (args.data_dir, identifier))

            scene = load_file(args.scene_file, integrator='direct', focal_length=str(args.focal_length)+'mm', tx="0", ty="0", tz="0", ox=c[0], oy=c[1], oz=c[2], \
                        ux="0", uy="1", uz="0", spp=4, width=args.img_width, height=args.img_height)
            
            params = traverse(scene)
            params.keep(['obj_1.bsdf.reflectance.data'])
            params.update()

            crop_size = scene.sensors()[0].film().crop_size()

            opt = Adam(params, lr=init_lr)

            final_ob_val = 0.0
            for si in range(args.epochs):
                # Perform a differentiable rendering of the scene
                image = render(scene, optimizer=opt, unbiased=True, spp=8)
                image = image * mask.flatten()

                if(i%10 == 0 and si == 0):
                    gt_ = (gt**(1.0/2.2) ) * 255.0
                    gt_ = gt_.astype(np.uint8)
                    gt_ = gt_.reshape(args.img_height, args.img_width, 3)
                    gt_ = cv2.cvtColor(gt_, cv2.COLOR_RGB2BGR)

                    write_bitmap('%s/dr_log/perfect_geometry/out_%05i_%05i_%05i_0.png' % (args.data_dir, epoch, i, si), image, crop_size)
                    cv2.imwrite('%s/dr_log/perfect_geometry/out_%05i_%05i_%05i_1.png' % (args.data_dir, epoch, i, si), gt_)

                ob_val = ek.hsum(ek.sqr(image - gt.flatten())) / len(image)
                ek.backward(ob_val)
                opt.step()

                final_ob_val += ob_val

                save_opt(args, params, crop_size)
            
            final_ob_val /= args.epochs

            writer.add_scalar('Loss', final_ob_val.numpy(), cnt)
            writer.add_scalar('Learning Rate', init_lr, cnt)
            cnt += 1