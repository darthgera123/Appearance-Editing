import os
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./', type=str, help='Path to Data')
parser.add_argument('--width', type=int, default=960, help='Width')
parser.add_argument('--height', type=int, default=540, help='Height')

args = parser.parse_args()

try:
    '''
    Making Dirs for the task
    '''

    data_dir = pathlib.Path(args.data_dir).resolve(strict=True)

    for _dir in ['video_frames', 'video_frames_test', 'video_frames_mask',
                'video_frames_test_mask', 'video_frames_extrinsics', 
                'video_frames_test_extrinsics', 'dr_log', 'dr_tensorboard', 'optimized_textures', 
                'dr_tensorboard/perfect_geometry/', 'dr_log/perfect_geometry', 'optimized_textures/perfect_geometry']:
        
        dir = data_dir / _dir
        dir.mkdir(parents=True, exist_ok=True)


    '''
    DATA GEN
    '''
    os.chdir('..')
    synthetic_gen = 'python3 data/synthetic_generate.py '+ \
                    '--scene_file ' + str(data_dir/'scene_gt.xml') + \
                    ' --vector_scale 3.5' + \
                    ' --output_dir '+ str(data_dir) + \
                    ' --img_width '+ str(args.width) + \
                    ' --img_height ' + str(args.height) + ' --spp 32'
    os.system(synthetic_gen)
    exit()
    os.chdir('./preprocess')
    '''
    MASK GEN
    '''
    mask_gen_train = 'python3 create_synthetic_frame_mask.py --data_dir ' + str(data_dir) +\
                     ' --frames_dir video_frames' + \
                     ' --output_dir video_frames_mask'
    mask_gen_test = 'python3 create_synthetic_frame_mask.py --data_dir ' + str(data_dir) +\
                     ' --frames_dir video_frames_test' + \
                     ' --output_dir video_frames_test_mask'

    os.system(mask_gen_train)
    os.system(mask_gen_test)

except:

    print('Error Occured')