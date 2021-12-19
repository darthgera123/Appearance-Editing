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
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

import config
from dataset.uv_dataset import UVDataset, UVDatasetEvalReal
from model.pipeline_new import PipeLine
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
    parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
    parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
    parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
    parser.add_argument('--view_direction', type=bool,
                        default=config.VIEW_DIRECTION)
    parser.add_argument('--data', type=str,
                        default=config.DATA_DIR, help='directory to data')
    parser.add_argument('--lif_checkpoint', type=str,
                        default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
    parser.add_argument('--mask_checkpoint', type=str,
                        default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
    parser.add_argument('--logdir', type=str, default=config.LOG_DIR,
                        help='directory to save checkpoint')
    parser.add_argument('--train', default=config.TRAIN_SET)
    parser.add_argument('--epoch', type=int, default=config.EPOCH)
    parser.add_argument('--cropw', type=int, default=config.CROP_W)
    parser.add_argument('--croph', type=int, default=config.CROP_H)
    parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--betas', type=str, default=config.BETAS)
    parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
    parser.add_argument('--eps', type=float, default=config.EPS)
    parser.add_argument('--load', type=str, default=config.LOAD)
    parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
    parser.add_argument('--epoch_per_checkpoint', type=int,
                        default=config.EPOCH_PER_CHECKPOINT)

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--material', type=str, default='')

    args = parser.parse_args()

    dataset = UVDatasetEvalReal(
        args.data, args.train, args.croph, args.cropw, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=False, num_workers=0)

    model_lif = torch.load(args.lif_checkpoint)
    model_lif = model_lif.to('cuda')
    model_lif.eval()

    l = model_lif.state_dict()
    k = []
    for key in l.keys():
        if 'albedo_tex' in key:
            k.append(key)

    img = Image.open((args.material), 'r')
    img = transforms.ToTensor()(img)
    img = img**(2.2)

    for i in range(3):
        model_lif.state_dict()[k[i]][0, 0] = img[i].cuda()
        model_lif.state_dict()[k[i+3]][0, 0] = img[i].cuda()

    model_mask = torch.load(args.mask_checkpoint)
    model_mask = model_mask.to('cuda')
    model_mask.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    iidx = 0
    for idx, samples in enumerate(tqdm(dataloader)):
        # print(idx)
        images, uv_maps, mask, extrinsics, wi, envmap = samples
        mask = mask.cuda()
        RGB_texture_masks, net_masks = model_mask(
            uv_maps.cuda(), extrinsics.cuda())

        mask_sigmoid = nn.Sigmoid()(net_masks)
        mask_sigmoid[mask_sigmoid >= 0.5] = 1
        mask_sigmoid[mask_sigmoid < 0.5] = 0
        images = images.cuda() * mask

        RGB_texture, preds, forward, albedo_uv = model_lif(
            wi.cuda(), envmap.cuda(), uv_maps.cuda(), extrinsics.cuda())
        preds *= mask_sigmoid
        forward *= mask

        mask = mask.cpu().numpy()

        for j in range(0, preds.shape[0]):

            output = np.clip(preds[j, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0 / 2.2)
            output = output * 255.0
            output = output.astype(np.uint8)
            output = np.transpose(output, (1, 2, 0))
            # print(output.shape)

            gt = np.clip(images[j, :, :, :].detach(
            ).cpu().numpy(), 0, 1) ** (1.0/2.2)
            gt *= mask[j, :, :, :]
            gt = gt * 255.0
            gt = gt.astype(np.uint8)
            gt = np.transpose(gt, (1, 2, 0))

            for_rend = np.clip(
                forward[j, :, :, :].cpu().numpy(), 0, 1) ** (1.0/2.2)
            for_rend *= mask[j, :, :, :]
            for_rend = for_rend * 255.0
            for_rend = for_rend.astype(np.uint8)
            for_rend = np.transpose(for_rend, (1, 2, 0))
            
            albedo = np.clip(RGB_texture[0, :, :, :].detach().cpu().numpy(), 0, 1) ** (1.0 / 2.2)
            albedo = albedo * 255.0
            albedo = albedo.astype(np.uint8)

            cv2.imwrite(args.output_dir+'/%s_output.png' %
                        str(iidx).zfill(5), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.output_dir+'/%s_gt.png' %
                        str(iidx).zfill(5), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.output_dir+'/%s_forward.png' %
                        str(iidx).zfill(5), cv2.cvtColor(for_rend, cv2.COLOR_RGB2BGR))

            iidx += 1
            albedo = np.transpose(albedo, (1, 2, 0))
            albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
            cv2.imwrite('texture.png', albedo)
