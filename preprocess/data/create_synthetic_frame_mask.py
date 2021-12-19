import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='')
parser.add_argument('--output_dir', type=str, default='video_frames_mask', help='')
parser.add_argument('--frames_dir',type=str,default='video_frames')

args = parser.parse_args()

images_dir = f'{args.data_dir}/{args.frames_dir}/'
prediction_dir = f'{args.data_dir}/{args.output_dir}/'

img_list = os.listdir(images_dir)

for item in img_list:
	img = cv2.imread('%s/%s' % (images_dir, item))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img[img>10] = 255.0

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.imwrite('%s/%s' % (prediction_dir, item.replace('.JPG', '.png')), img)