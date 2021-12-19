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
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
import cv2
import numpy as np 

from utils import *
from poses.read_model import camera_pose

import tensorboardX

def save_opt(args, params):
    diff_opt = np.array(params['obj_1.bsdf.bsdf_1.reflectance.data'])
    diff_opt = (diff_opt + np.array(params['obj_1.bsdf.bsdf_0.diffuse_reflectance.data']))/2.0
    # diff_opt = np.clip(diff_opt**(1.0/2.2), 0, 1) * 255.0
    diff_opt = np.clip(np.sign(diff_opt)*(np.abs(diff_opt))**(1.0/2.2), 0, 1) * 255.0
    diff_opt = diff_opt.astype(np.uint8)
    diff_opt = diff_opt.reshape(256, 256, 3)

    # weight_opt = np.clip(np.array(params['obj_1.bsdf.weight.data']), 0, 1)
    # weight_opt = (weight_opt**(1.0/2.2) ) * 255.0
    # weight_opt = weight_opt.astype(np.uint8)
    # weight_opt = weight_opt.reshape(512, 512, 3)
    
    alpha_opt = np.array(params['obj_1.bsdf.weight.data'])
    # alpha_opt = np.clip(alpha_opt**(1.0/2.2), 0, 1) * 255.0
    alpha_opt = np.clip(np.sign(alpha_opt)*(np.abs(alpha_opt))**(1.0/2.2), 0, 1) * 255.0
    alpha_opt = alpha_opt.astype(np.uint8)
    alpha_opt = alpha_opt.reshape(256, 256, 3)

    cv2.imwrite('%s/diffuse_opt.png' % args.data_dir, cv2.cvtColor(diff_opt, cv2.COLOR_RGB2BGR))
    cv2.imwrite('%s/alpha_opt.png' % args.data_dir, cv2.cvtColor(alpha_opt, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    parser.add_argument('--sensor_width', type=float, default=6.4) # Sensor width in mm
    parser.add_argument('--sensor_height', type=float, default=4.8) # Sensor height in mm
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    writer = tensorboardX.SummaryWriter(logdir=args.data_dir+'/dr_tensorboard')

    init_tex = np.ones((256, 256, 3), dtype=np.float) * 0.5
    cv2.imwrite('%s/diffuse_opt.png' % args.data_dir, (init_tex*255.0).astype(np.uint8))
    cv2.imwrite('%s/alpha_opt.png' % args.data_dir, (init_tex*255.0).astype(np.uint8))

    img_list = os.listdir('%s/colmap_output/dense/0/images/' % args.data_dir)

    init_lr = 0.2
    c = 0
    for epoch in range(0, args.epochs):
        print("Epoch :",epoch+1)

        random.shuffle(img_list)

        for i, img_name in enumerate(img_list):
            gt = load_image('%s/colmap_output/dense/0/images/%s' % (args.data_dir, img_name), (args.img_width, args.img_height))

            p, focal_length, og_width, og_height = camera_pose('%s/colmap_output/' % args.data_dir, img_name, 'sparse')
            pose = ' '.join([str(elem) for elem in p])

            # print(focal_length)
            # print(og_width)
            # print(og_height, args.sensor_width, sep='\n')
            estimated_f = math.sqrt( pow(args.sensor_width, 2) + pow(args.sensor_height, 2) ) * focal_length / math.sqrt( pow(og_width, 2) + pow(og_height, 2) )
            focal_length = estimated_f * 34.6 / 6.4
            # print(focal_length)
            # focal_length = focal_length * args.sensor_width / og_width
            # focal_length = focal_length * 35.0 / 26.0 # Correct for equivalent focal length of film camera

            scene = load_file(args.scene_file, integrator='direct', focal_length=str(focal_length)+'mm', poses=pose, envmap_pose=pose, \
                        spp=10, width=args.img_width, height=args.img_height)
            # scene = load_file(args.scene_file, integrator='direct', focal_length='27.0mm', poses=pose, envmap_pose=pose, \
            #             spp=10, width=args.img_width, height=args.img_height)
            
            params = traverse(scene)
            params.keep(['obj_1.bsdf.weight.data', 'obj_1.bsdf.bsdf_0.diffuse_reflectance.data', 'obj_1.bsdf.bsdf_1.reflectance.data'])
            # params.keep(['obj_1.bsdf.reflectance.data'])
            params.update()

            crop_size = scene.sensors()[0].film().crop_size()

            opt = Adam(params, lr=init_lr)

            sub_epochs = 1
            final_ob_val = 0.0
            for si in range(sub_epochs):
                # Perform a differentiable rendering of the scene
                image = render(scene, optimizer=opt, unbiased=True, spp=8)

                if(i%5 == 0 and si == 0):
                    gt_ = (gt**(1.0/2.2) ) * 255.0
                    gt_ = gt_.astype(np.uint8)
                    gt_ = gt_.reshape(args.img_height, args.img_width, 3)
                    gt_ = cv2.cvtColor(gt_, cv2.COLOR_RGB2BGR)

                    write_bitmap('%s/dr_log/out_%05i_%05i_0.png' % (args.data_dir, epoch, i), image, crop_size)
                    cv2.imwrite('%s/dr_log/out_%05i_%05i_1.png' % (args.data_dir, epoch, i), gt_)
                    # write_bitmap('%s/dr_log/%s.png' % (args.data_dir, img_name.replace('png', '')), image, crop_size)
                    # cv2.imwrite('%s/dr_log/%s_gt.png' % (args.data_dir, img_name.replace('png', '')), gt_)

                # Objective: MSE between 'image' and 'image_ref'
                ob_val = ek.hsum(ek.abs(image - gt.flatten())) / len(image)
                final_ob_val += ob_val
                # Back-propagate errors to input parameters
                ek.backward(ob_val)
                # Optimizer: take a gradient step
                opt.step()
            
            final_ob_val /= sub_epochs

            writer.add_scalar('Loss', final_ob_val.numpy(), c)
            writer.add_scalar('Learning Rate', init_lr, c)
                
            print('Iteration %03i' % (i),end="\r")

            save_opt(args, params)
            
            del scene
            del params
            del opt

            ek.cuda_malloc_trim()

            c += 1

        init_lr /= 2.0
