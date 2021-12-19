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
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, default='pinecone.xml')
    parser.add_argument('--vector_scale', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--focal_length', type=str, default='28.0mm') # Default set to focal length of iPhone X, wide angle camera
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--spp', type=int, default=32)

    args = parser.parse_args()

    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    register_integrator('auxintegrator', lambda props: sh.AuxIntegrator(props))

    Thread.thread().file_resolver().append(os.path.dirname(args.scene_file))

    theta_samples = np.linspace(0.2*np.pi/2.0, 0.7*np.pi/2.0, num=3)
    phi_samples = np.linspace(0, 2*np.pi, num=20)

    cam = []
    for theta in theta_samples:
        for phi in phi_samples:
            st = np.sin(theta)
            ct = np.cos(theta)
            sp = np.sin(phi)
            cp = np.cos(phi)

            v = np.array([cp*st, ct, sp*st], dtype=np.float)
            scale = args.vector_scale + random.random() * (args.vector_scale)/10.0
            cam.append(scale*v)
    
    for i, c in enumerate(cam):      
        frame_name = str(i).zfill(5)

        scene = load_file(args.scene_file, integrator='direct', focal_length=args.focal_length, tx="0", ty="0", tz="0", ox=c[0], oy=c[1], oz=c[2], \
                        ux="0", uy="1", uz="0", spp=args.spp, width=IMG_WIDTH, height=IMG_HEIGHT)

        scene.integrator().render(scene, scene.sensors()[0])
        film = scene.sensors()[0].film()
        img = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite('%s/colmap_capture/%s.JPG' % (args.output_dir, frame_name), img)

        del scene
        ek.cuda_malloc_trim()