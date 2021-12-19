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
from utils import *
from poses.read_model import camera_pose

import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')
from pathlib import Path

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

from mitsuba.python.autodiff import render

def process(args, i, img_path, frames_dir, output_dir, colmap_dir, env_pose):
    img_path = img_path.replace('\n', '')
    img_name = img_path.split('/')[-1]
    identifier = img_name.replace('.png', '').replace('.jpg', '').replace('.JPG', '').replace('image', '')

    if not os.path.exists('%s/%s/%s' % (args.data_dir, frames_dir, img_name)):
        print('%s/%s/%s' % (args.data_dir, frames_dir, img_name), 'does not exists')
        return

    gt = load_image('%s/%s/%s' % (args.data_dir, frames_dir, img_name), (args.img_width, args.img_height))
    mask = load_image('%s/%s_mask/%s.png' % (args.data_dir, frames_dir, identifier), (args.img_width, args.img_height))
    gt = gt * mask + (1-mask)

    p, focal_length, og_width, og_height = camera_pose('%s/%s/' % (args.data_dir, colmap_dir), img_path, 'new_sparse')
    pose = ' '.join([str(elem) for elem in p])

    estimated_f = math.sqrt( pow(args.sensor_width, 2) + pow(args.sensor_height, 2) ) * focal_length / math.sqrt( pow(og_width, 2) + pow(og_height, 2) )
    focal_length = estimated_f * 34.6 / 6.4

    tx = p[3]
    ty = p[7]
    tz = p[11]

    scene_forward = load_file(args.scene_file, integrator='path', focal_length=str(focal_length)+'mm', poses=pose, envmap_pose=env_pose, \
                        spp=8, width=args.img_width, height=args.img_height)
                        
    scene_forward.integrator().render(scene_forward, scene_forward.sensors()[0])
    film_forward = scene_forward.sensors()[0].film()
    img = film_forward.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    scene = load_file(args.scene_file, integrator='auxintegrator', focal_length=str(focal_length)+'mm', poses=pose, envmap_pose=env_pose, \
                        spp=8, width=args.img_width, height=args.img_height)

    rendered_op = render(scene, spp=1)
    rendered_op = rendered_op.numpy().reshape(IMG_HEIGHT, IMG_WIDTH, 11, 3)

    env = np.clip(rendered_op[:, :, 1, :], 0, 1) ** (1.0/2.2)
    env *= 255.0
    env = env.astype(np.uint8)
    env = cv2.cvtColor(env, cv2.COLOR_RGB2BGR)

    f_sh = rendered_op[:, :, 2:, :]
    f_sh[f_sh == 0] = 0.01
    f_sh = f_sh.reshape(IMG_HEIGHT, IMG_WIDTH, -1)
    
    uv = rendered_op[:, :, 0, :]

    uv_png = uv.copy()
    uv_png *= 255.0
    uv_png = uv_png.astype(np.uint8)

    gt = np.clip(gt, 0, 1)**(1.0/2.2)
    gt *= 255.0
    gt = gt.astype(np.uint8)

    mask = np.clip(mask, 0, 1)
    mask *= 255.0
    mask = mask.astype(np.uint8)

    cv2.imwrite('%s/forward/%s.png' % (output_dir, identifier), img)
    cv2.imwrite('%s/env/%s.png' % (output_dir, identifier), env)
    cv2.imwrite('%s/frames/%s.png' % (output_dir, identifier), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite('%s/mask/%s.png' % (output_dir, identifier), mask)
    np.save('%s/uv/%s.npy' % (output_dir, identifier), uv)
    cv2.imwrite('%s/uv_png/%s.png' % (output_dir, identifier), cv2.cvtColor(uv_png, cv2.COLOR_RGB2BGR))
    np.save('%s/extrinsics/%s.npy' % (output_dir, identifier), np.array([tx, ty, tz], dtype=np.float64))
    np.save('%s/sh/%s.npy' % (output_dir, identifier), f_sh)

    del rendered_op
    del scene

    ek.cuda_malloc_trim()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    # parser.add_argument('--output_dir', type=str, default='./', help='')
    parser.add_argument('--train_image_list_txt', type=str, default='./', help='')
    parser.add_argument('--test_image_list_txt', type=str, default='./', help='')
    parser.add_argument('--sensor_width', type=float, default=6.4) # Sensor width in mm, default for ROG phone 2
    parser.add_argument('--sensor_height', type=float, default=4.8) # Sensor height in mm, default for ROG phone 2
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--alignment_x', type=float, default=0.0)
    parser.add_argument('--alignment_y', type=float, default=1.0)
    parser.add_argument('--alignment_z', type=float, default=0.0)

    args = parser.parse_args()

    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    register_integrator('auxintegrator', lambda props: sh.AuxIntegrator(props))
    
    output_parent = Path(args.data_dir) / '0-DR-Dataset'
    sub_folders  = 'env  extrinsics  forward  frames  mask  sh  uv  uv_png'.split()

    for _fol in ['train', 'test']:
        for _sub_folder in sub_folders:
            _ = output_parent / _fol /_sub_folder
            _.mkdir(parents=True, exist_ok=True)

    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    alignment_vec = np.array([args.alignment_x, args.alignment_y, args.alignment_z], dtype=np.float32)
    from_vec = np.array([0.0, 1.0, 0.0])
    env_pose = trimesh.geometry.align_vectors(from_vec, alignment_vec)
    env_pose = env_pose.flatten()
    env_pose = ' '.join([str(elem) for elem in env_pose])

    frames_dir = 'video_frames'
    img_list_file = sorted(open(args.train_image_list_txt, 'r'))
    img_list = []
    for l in img_list_file:
        img_list.append(l)

    for i, img_path in enumerate(img_list):
        process(args, i, img_path, frames_dir, str(output_parent)+'/train/', 'colmap_output/', env_pose)
    
    #--Test--
    frames_dir = 'video_frames_test'
    img_list_file = sorted(open(args.test_image_list_txt, 'r'))
    img_list = []
    for l in img_list_file:
        img_list.append(l)

    for i, img_name in enumerate(img_list):
        process(args, i, img_name, frames_dir, str(output_parent)+'/test/', 'colmap_output/colmap_output_test/', env_pose)