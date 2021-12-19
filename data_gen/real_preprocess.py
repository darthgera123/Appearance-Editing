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
from utils import *
from poses.read_model import camera_pose

import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

def process(args, i, img_name):
    identifier = img_name.replace('.png', '').replace('.jpg', '').replace('.JPG', '')

    p = camera_pose('%s/colmap_output/' % args.data_dir, img_name, 'new_sparse')
    pose = ' '.join([str(elem) for elem in p])

    scene = load_file(args.scene_file, integrator='auxintegrator', focal_length=args.focal_length, poses=pose, envmap_pose=pose, \
                spp=1, width=args.img_width, height=args.img_height)

    scene.integrator().render(scene, scene.sensors()[0])
    film = scene.sensors()[0].film()
    film.set_destination_file('%s/render_tmp.exr' % (args.data_dir))
    film.develop()

    exrfile = pyexr.open('%s/render_tmp.exr' % (args.data_dir))

    sh_channels_list = []
    for i in range(0, 25):
        for c in ['r', 'g', 'b']:
            sh_channels_list.append('sh_%s_%d' % (c, i))

    f_sh = np.zeros((IMG_HEIGHT, IMG_WIDTH, 75), dtype=np.float)
    for i, channel in enumerate(sh_channels_list):
        ch = exrfile.get(channel)
        f_sh[:, :, i:i+1] += ch
    
    uv = np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float)
    uv[:, :, 0:1] = exrfile.get('R')
    uv[:, :, 1:2] = exrfile.get('G')

    uv_png = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float)
    uv_png[:, :, 0:2] = uv
    uv_png *= 255.0
    uv_png = uv_png.astype(np.uint8)

    tx = p[3]
    ty = p[7]
    tz = p[11]
    extrinsics = np.array([tx, ty, tz], dtype=np.float)

    os.system('%s/video_frames/%s %s/train/images/' % (args.data_dir, img_name, args.data_dir))
    np.save('%s/uv/%s.npy' % (args.output_dir, frame_name), uv)
    cv2.imwrite('%s/uv_png/%s.png' % (args.output_dir, frame_name), cv2.cvtColor(uv_png, cv2.COLOR_RGB2BGR))
    np.save('%s/sh/%s.npy' % (args.data_dir, identifier), f_sh)
    np.save('%s/extrinsics/%s.npy' % (args.output_dir, frame_name), extrinsics)

    ek.cuda_malloc_trim()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    parser.add_argument('--focal_length', type=str, default='28.0mm') # Default set to focal length of iPhone X, wide angle camera
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    args = parser.parse_args()

    register_integrator('auxintegrator', lambda props: sh.AuxIntegrator(props))
    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    img_list = os.listdir('%s/video_frames/' % args.data_dir)
    for i, img_name in enumerate(img_list):
        process(args, i, img_name)
    
    os.system('rm %s/render_tmp.exr' % args.data_dir)