import cv2
import os
import argparse


parser= argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='video_frames')
parser.add_argument('--output_video', type=str, default='train_video.mp4')

args = parser.parse_args()
image_folder = args.input_dir
video_name = args.output_video

images = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()