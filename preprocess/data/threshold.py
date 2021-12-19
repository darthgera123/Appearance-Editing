import sys, os
import numpy as np
import argparse
import skimage.color
import skimage.filters
import skimage.io


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--th', type=float, default=.8)

    args = parser.parse_args()

    img_list = sorted( os.listdir(args.input_dir) )

    for file in img_list:
        image = skimage.io.imread(fname=('%s/%s'%(args.input_dir, file)))
        _temp = skimage.color.rgb2gray(image)
        mask = _temp < args.th
        skimage.io.imsave('%s/%s'%(args.output_dir, file), (mask*255).astype('uint8'))