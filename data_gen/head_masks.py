import numpy as np
import cv2, os, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    #AS WE NEED ALL THE SCENE WE JUST USE DUMMY WHITE MASK
    args = parser.parse_args()
    img_list = sorted(os.listdir(args.input_dir))

    for img_name in img_list:
        mask = np.ones(shape=(args.img_width, args.img_height)) * 255
        cv2.imwrite('%s/%s'%(args.output_dir, img_name), mask)