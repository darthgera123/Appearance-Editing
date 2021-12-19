import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--img_1', type=str, default='img1.png')
parser.add_argument('--img_2', type=str, default='img2.png')
parser.add_argument('--output', type=str, default='cmp_img1_img2.png')

args = parser.parse_args()

img1 = cv2.imread(args.img_1)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


img2 = cv2.imread(args.img_2)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

error = np.abs(img2-img1)
error = (error * 255.0).astype(np.uint8)
error = cv2.cvtColor(error,cv2.COLOR_RGB2GRAY)

fig, ax = plt.subplots()
im = ax.imshow(error)

fig.colorbar(im, orientation='vertical')
fig.tight_layout()
plt.axis('off')
plt.savefig(args.output, dpi=300, pad_inches=0)
# plt.imsave(args.output,error)