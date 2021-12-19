import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F

import os, sys, cv2, json, argparse, random, glob, struct, math, time
import torchvision.transforms as transforms

def load_image(img_path, new_size):
    image_ref = cv2.imread(img_path)

    if not os.path.isfile(img_path):
        return None, False
    
    if image_ref.shape[0] > image_ref.shape[1]:
        image_ref = cv2.resize(image_ref, (new_size[1], new_size[0]))
    else:
        image_ref = cv2.resize(image_ref, (new_size[0], new_size[1]))

    return image_ref, True

def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = image_transforms(image)
    return image


def map_transform(map):
    map = torch.from_numpy(map)
    return map

def augment_eval(img, mask, map, sh, env_sh, crop_size,forward):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)
    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    forward = forward.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    sh = sh[:, h1:h1 + crop_h, w1:w1 + crop_w]
    env_sh = env_sh[:, h1:h1 + crop_h, w1:w1 + crop_w]

    # final transform
    img, mask, forward, map, sh, env_sh = img_transform(img), img_transform(mask), img_transform(forward),\
                                    map_transform(map), torch.from_numpy(sh), torch.from_numpy(env_sh)
    
    # mask for valid uv positions
    # mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
    # mask = mask.repeat((3,1,1))

    return img, map, sh, env_sh, mask, forward

def augment_new(img, map, mask, transform, crop_size):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)
    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    transform = transform[h1:h1 + crop_h, w1:w1 + crop_w, :]

    # final transform
    img, mask, map, transform = img_transform(img), img_transform(mask), map_transform(map), torch.from_numpy(transform)

    return img, map, mask, transform

def augment_new_eval(img, map, mask, transform, crop_size):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)
    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    transform = transform[h1:h1 + crop_h, w1:w1 + crop_w, :]

    # final transform
    img, mask, map, transform = img_transform(img), img_transform(mask), map_transform(map), torch.from_numpy(transform)

    return img, map, mask, transform


def augment(img, mask, forward, env, map, sh, crop_size):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)

    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    forward = forward.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    env = env.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    sh = sh[:, h1:h1 + crop_h, w1:w1 + crop_w]

    # final transform
    img, mask, forward, env, map, sh = img_transform(img), img_transform(mask), img_transform(forward), \
                                    img_transform(env), map_transform(map), torch.from_numpy(sh)
    
    # mask for valid uv positions
    # mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
    # mask = mask.repeat((3,1,1))

    return img, mask, forward, env, map, sh

def augment_center_crop(img, mask, map, sh, crop_size,forward):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size

    w1 = w/2.0
    w1 = int(w1 - crop_w/2.0)
    h1 = h/2.0
    h1 = int(h1 - crop_h/2.0)

    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    forward = forward.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    sh = sh[:, h1:h1 + crop_h, w1:w1 + crop_w]

    # final transform
    img, mask, forward, map, sh = img_transform(img), img_transform(mask), img_transform(forward),\
                                    map_transform(map), torch.from_numpy(sh)
    
    # mask for valid uv positions
    # mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
    # mask = mask.repeat((3,1,1))

    return img, map,sh,mask,forward

def augment_center_crop_mask(img, mask, map,crop_size):
    '''
    :param img:  PIL input image
    :param mask:  PIL input mask
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size

    w1 = w/2.0
    w1 = int(w1 - crop_w/2.0)
    h1 = h/2.0
    h1 = int(h1 - crop_h/2.0)

    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    mask = mask.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
    # forward = forward.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    # sh = sh[:, h1:h1 + crop_h, w1:w1 + crop_w]

    # final transform
    img, mask, map,  = img_transform(img), img_transform(mask),\
                                    map_transform(map)
    
    

    return img, mask, map

def augment_og(img, map, crop_size):
    '''
    :param img:  PIL input image
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)
    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]

    # final transform
    img, map = img_transform(img), map_transform(map)
    
    # mask for valid uv positions
    mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
    mask = mask.repeat((3,1,1))

    return img, map, mask