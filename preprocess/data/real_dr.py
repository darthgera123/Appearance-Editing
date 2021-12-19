import torch, os, sys, cv2, json, argparse, random, glob, struct, math, time, trimesh
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
from pathlib import Path

from utils import *
from poses.read_model import camera_pose

import tensorboardX

def save_opt(args, params, crop_size):
    write_bitmap('%s/0-DR-Dataset/optimized_textures/diffuse_opt.exr' % args.data_dir, params['obj_1.bsdf.reflectance.data'], (512, 512), write_async=False)

    # diff_opt = params['obj_1.bsdf.reflectance.data']
    # diff_opt = np.clip(np.sign(diff_opt)*(np.abs(diff_opt))**(1.0/2.2), 0, 1) * 255.0
    # diff_opt = diff_opt.astype(np.uint8)
    # diff_opt = diff_opt.reshape(512, 512, 3)

    # cv2.imwrite('%s/optimized_textures/diffuse_opt.png' % args.data_dir, cv2.cvtColor(diff_opt, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    parser.add_argument('--image_list_txt', type=str, default='./', help='')
    parser.add_argument('--sensor_width', type=float, default=6.4) # Sensor width in mm, default for ROG phone 2
    parser.add_argument('--sensor_height', type=float, default=4.8) # Sensor height in mm, default for ROG phone 2
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--alignment_x', type=float, default=0.0)
    parser.add_argument('--alignment_y', type=float, default=1.0)
    parser.add_argument('--alignment_z', type=float, default=0.0)

    args = parser.parse_args()
    
    _folders = 'dr_log dr_tensorboard optimized_textures test train'.split()
    for _fol in _folders:
        (Path(args.data_dir)/ '0-DR-Dataset' / _fol).mkdir(parents=True, exist_ok=True)


    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    init_tex = np.ones((512, 512, 3), dtype=np.float32) * 0.5
    imageio.imwrite('%s/0-DR-Dataset/optimized_textures/diffuse_opt.exr' % args.data_dir, init_tex)
    # cv2.imwrite('%s/optimized_textures/diffuse_opt.exr' % args.data_dir, (init_tex*255.0).astype(np.uint8))

    writer = tensorboardX.SummaryWriter(logdir=args.data_dir+'/0-DR-Dataset/dr_tensorboard/')
    
    img_list_file = open(args.image_list_txt, 'r')
    img_list = []
    for l in img_list_file:
        img_list.append(l)
    
    alignment_vec = np.array([args.alignment_x, args.alignment_y, args.alignment_z], dtype=np.float)
    from_vec = np.array([0.0, 1.0, 0.0])
    env_pose = trimesh.geometry.align_vectors(from_vec, alignment_vec)
    env_pose = env_pose.flatten()
    env_pose = ' '.join([str(elem) for elem in env_pose])
    print(env_pose)

    c = 0
    for epoch in range(0, 2):

        random.shuffle(img_list)
        for i, img_path in enumerate(img_list):            
            init_lr = 1e-2
            
            img_path = img_path.replace('\n', '')
            img_name = img_path.split('/')[-1]

            identifier = img_name.replace('.png', '').replace('.jpg', '').replace('.JPG', '').replace('image', '')
            
            print('########################################')
            print('%s/video_frames/%s.png' % (args.data_dir, identifier))
            print('########################################')
            if not os.path.exists('%s/video_frames/%s' % (args.data_dir, img_name)):
                print('%s/video_frames/%s' % (args.data_dir, img_name), 'does not exists')
                continue

            gt = load_image('%s/video_frames/%s' % (args.data_dir, img_name), (args.img_width, args.img_height))
            mask = load_image('%s/video_frames_mask/%s.png' % (args.data_dir, identifier), (args.img_width, args.img_height))

            gt = gt * mask
            # print('%s/colmap_output/' % args.data_dir, img_path, 'new_sparse')

            p, focal_length, og_width, og_height = camera_pose('%s/colmap_output/' % args.data_dir, img_path, 'new_sparse')
            pose = ' '.join([str(elem) for elem in p])
            eye = np.eye(4).flatten()
            eye = ' '.join([str(elem) for elem in eye])

            estimated_f = math.sqrt( pow(args.sensor_width, 2) + pow(args.sensor_height, 2) ) * focal_length / math.sqrt( pow(og_width, 2) + pow(og_height, 2) )
            focal_length = estimated_f * 34.6 / args.sensor_width

            scene = load_file(args.scene_file, integrator='path', focal_length=str(focal_length)+'mm', poses=pose, envmap_pose=env_pose, \
                        spp=4, width=args.img_width, height=args.img_height)
            # scene = load_file(args.scene_file, integrator='path', focal_length='35mm', poses=pose, envmap_pose=pose, \
            #             spp=8, width=args.img_width, height=args.img_height)
            
            params = traverse(scene)
            params.keep(['obj_1.bsdf.reflectance.data'])
            params.update()

            crop_size = scene.sensors()[0].film().crop_size()

            opt = Adam(params, lr=init_lr)

            final_ob_val = 0.0
            for si in range(args.epochs):
                # Perform a differentiable rendering of the scene
                image = render(scene, optimizer=opt, unbiased=True, spp=8)
                # image = image * mask.flatten()

                if(i%10 == 0 and si == 0):
                    gt_ = (gt**(1.0/2.2) ) * 255.0
                    gt_ = gt_.astype(np.uint8)
                    gt_ = gt_.reshape(args.img_height, args.img_width, 3)
                    gt_ = cv2.cvtColor(gt_, cv2.COLOR_RGB2BGR)

                    write_bitmap('%s/0-DR-Dataset/dr_log/out_%05i_%05i_%05i_0.png' % (args.data_dir, epoch, i, si), image, crop_size)
                    cv2.imwrite('%s/0-DR-Dataset/dr_log/out_%05i_%05i_%05i_1.png' % (args.data_dir, epoch, i, si), gt_)

                ob_val = ek.hsum(ek.sqr(image - gt.flatten())) / len(image)
                ek.backward(ob_val)
                opt.step()

                final_ob_val += ob_val

                save_opt(args, params, crop_size)
            
            final_ob_val /= args.epochs

            writer.add_scalar('Loss', final_ob_val.numpy(), c)
            writer.add_scalar('Learning Rate', init_lr, c)
            c += 1