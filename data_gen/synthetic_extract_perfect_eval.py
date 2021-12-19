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
import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

from mitsuba.python.autodiff import render

from U2Net.u2net_test import segment_object

def process(args, i, img_name, frames_dir, output_dir):
    identifier = img_name.replace('.png', '').replace('.jpg', '').replace('.JPG', '').replace('image', '')

    gt = load_image('%s/%s/%s' % (args.data_dir, frames_dir, img_name), (args.img_width, args.img_height))
    mask = load_image('%s/%s_mask/%s.png' % (args.data_dir, frames_dir, identifier), (args.img_width, args.img_height))
    c = np.load('%s/%s_extrinsics/%s.npy' % (args.data_dir, frames_dir, identifier))

    scene_forward = load_file(args.scene_file, integrator='direct', focal_length=args.focal_length, tx="0", ty="0", tz="0", ox=c[0], oy=c[1], oz=c[2], \
                    ux="0", uy="1", uz="0", spp="32", width=args.img_width, height=args.img_height)
    scene_forward.integrator().render(scene_forward, scene_forward.sensors()[0])
    film_forward = scene_forward.sensors()[0].film()
    img = film_forward.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Envmap SH
    scene = load_file(args.scene_file, integrator='envintegrator', focal_length=args.focal_length, tx="0", ty="0", tz="0", ox=c[0], oy=c[1], oz=c[2], \
                    ux="0", uy="1", uz="0", spp="1", width=args.img_width, height=args.img_height)

    rendered_env = render(scene, spp=1)
    rendered_env = rendered_env.numpy().reshape(IMG_HEIGHT, IMG_WIDTH, 26, 3)

    env_sh = rendered_env[:, :, 1:, :]
    env_sh = env_sh.reshape(IMG_HEIGHT, IMG_WIDTH, -1)

    # BRDF SH
    scene = load_file(args.scene_file, integrator='auxintegrator', focal_length=args.focal_length, tx="0", ty="0", tz="0", ox=c[0], oy=c[1], oz=c[2], \
                    ux="0", uy="1", uz="0", spp="1", width=args.img_width, height=args.img_height)

    rendered_op = render(scene, spp=1)
    rendered_op = rendered_op.numpy().reshape(IMG_HEIGHT, IMG_WIDTH, 26, 3)

    f_sh = rendered_op[:, :, 1:, :]
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
    cv2.imwrite('%s/frames/%s.png' % (output_dir, identifier), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite('%s/mask/%s.png' % (output_dir, identifier), mask)
    np.save('%s/uv/%s.npy' % (output_dir, identifier), uv)
    cv2.imwrite('%s/uv_png/%s.png' % (output_dir, identifier), cv2.cvtColor(uv_png, cv2.COLOR_RGB2BGR))
    np.save('%s/extrinsics/%s.npy' % (output_dir, identifier), c)
    np.save('%s/sh/%s.npy' % (output_dir, identifier), f_sh)
    np.save('%s/env_sh/%s.npy' % (output_dir, identifier), env_sh)

    del rendered_op
    del uv
    del scene

    ek.cuda_malloc_trim()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone_dr.xml')
    parser.add_argument('--data_dir', type=str, default='./', help='')
    parser.add_argument('--output_dir', type=str, default='./', help='')
    parser.add_argument('--focal_length', type=str, default='28.0mm') # Default set to focal length of iPhone X, wide angle camera
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--lb', type=int, default=180)
    parser.add_argument('--ub', type=int, default=230)

    args = parser.parse_args()

    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    register_integrator('auxintegrator', lambda props: sh.AuxIntegrator(props))
    register_integrator('envintegrator', lambda props: sh.EnvIntegrator(props))

    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    frames_dir = 'video_frames_test'
    img_list = sorted(os.listdir('%s/%s/' % (args.data_dir, frames_dir)))[args.lb:args.ub]
    # img_list = sorted(os.listdir('%s/%s/' % (args.data_dir, frames_dir)))[319:]
    for i, img_name in enumerate(img_list):
        process(args, i, img_name, frames_dir, args.output_dir)
