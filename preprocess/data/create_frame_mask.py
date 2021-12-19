import os
import argparse
from U2Net.u2net_test import segment_object

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='')
parser.add_argument('--output_dir', type=str, default='video_frames_mask', help='')
parser.add_argument('--model_name',type = str, default='u2net', help='Select the model')
parser.add_argument('--frames_dir',type=str,default='video_frames')
parser.add_argument('--model_path', type=str, default='U2Net/saved_models')

args = parser.parse_args()

images_dir = f'{args.data_dir}/{args.frames_dir}/'
prediction_dir = f'{args.data_dir}/{args.output_dir}/'
model_dir = os.path.join(os.getcwd(),f'{args.model_path}/{args.model_name}/{args.model_name}.pth')
segment_object(args.model_name,images_dir,prediction_dir,model_dir)