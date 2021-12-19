import os
import argparse
import pathlib
from extract_sample_frames import extract_and_sample_frames

parser = argparse.ArgumentParser()
parser.add_argument('--train_video', type=str, default='./train.mp4', help='Provide Train Video File')
parser.add_argument('--test_video', type=str, default='./test.mp4', help='Provide Test Video File')
parser.add_argument('--train_skip', type=int, default=6, help='Frames to skip in train video')
parser.add_argument('--test_skip', type=int, default=6, help='Frames to skip in test video')
parser.add_argument('--colmap_skip', type=int, default=4, help='Frames to skip in train video for colmap')
parser.add_argument('--center_crop', type=str, default='no', help='Center Crop or Resize maintaining Aspect Ratio')
parser.add_argument('--width', type=int, default=960, help='Width')
parser.add_argument('--height', type=int, default=540, help='height')

args = parser.parse_args()
try:
    """
        Making Directories and extracting frames and sampling for colmap
    """

    extract_and_sample_frames(args)

    dataset_path = pathlib.Path(args.train_video).resolve(strict=True).parent


    """
        Creating Masks
    """

    os.system('python3 create_frame_mask.py --data_dir ' + str(dataset_path) +
              ' --frames_dir video_frames --output_dir video_frames_mask --model_path '
              'U2Net/saved_models --model_name u2net')

    os.system('python3 create_frame_mask.py --data_dir ' + str(dataset_path) +
              ' --frames_dir video_frames_test --output_dir video_frames_test_mask --model_path '
              'U2Net/saved_models --model_name u2net')


    """
        colmap dense reconstruction
    """

    feature_extraction = 'colmap feature_extractor --database_path ' + str(dataset_path / 'colmap_output/database.db') + \
                         ' --image_path ' + str(dataset_path / 'colmap_capture/') + ' --ImageReader.single_camera true'
    print(feature_extraction)
    os.system(feature_extraction)
    print("Feature Extraction Successful")

    exhaustive_mapper = 'colmap exhaustive_matcher --database_path ' + str(dataset_path / 'colmap_output/database.db')
    print(exhaustive_mapper)
    os.system(exhaustive_mapper)
    print("Exhaustive Mapper Successful")

    sparse_dir = 'mkdir ' + str(dataset_path/'colmap_output/sparse')
    os.system(sparse_dir)

    mapper = 'colmap mapper --image_path ' + str(dataset_path/'colmap_capture/') + ' --database_path ' + str(dataset_path/'colmap_output/database.db') + ' --output_path ' + str(dataset_path/'colmap_output/sparse/')
    os.system(mapper)
    print("Mapper Successful")

    img_undistort = 'colmap image_undistorter --image_path ' + str(dataset_path / 'colmap_capture') + ' --input_path ' + \
                    str(dataset_path / 'colmap_output/sparse/0/') + ' --output_path ' + \
                    str(dataset_path / 'colmap_output/dense/0/') + ' --output_type COLMAP --max_image_size 2000'
    os.system(img_undistort)
    print('Undistort Successful')

    patch_match_stereo = 'colmap patch_match_stereo --workspace_path ' + str(dataset_path/'colmap_output/dense/0') + ' --workspace_format COLMAP --PatchMatchStereo.geom_consistency true'
    os.system(patch_match_stereo)
    print('Path Match Stereo Successful')

    stereo_fusion = 'colmap stereo_fusion --workspace_path ' + str(dataset_path/'colmap_output/dense/0/') + \
                   ' --workspace_format COLMAP --input_type geometric --output_path ' + \
                   str(dataset_path/'colmap_output/dense/0/fused.ply')
    os.system(stereo_fusion)
    print('Stereo Fusion Successful')

    original_run_create = 'mkdir ' + str(dataset_path/'colmap_output/original_run')
    original_run_copy1 = 'cp -r ' + str(dataset_path/'colmap_output/dense/') + ' ' + str(dataset_path/'colmap_output/original_run/')
    original_run_copy2 = 'cp -r ' + str(dataset_path/'colmap_output/database.db') + ' ' + str(dataset_path/'colmap_output/original_run/')
    os.system(original_run_create)
    os.system(original_run_copy1)
    os.system(original_run_copy2)

    test_copy = 'cp -r ' + str(dataset_path/'colmap_output/original_run/*') + ' ' + str(dataset_path/'colmap_output/colmap_output_test/')
    os.system(test_copy)


    """
    	Registration
    """
    register_train = './register_images.sh NEW_IMAGES="video_frames" IMAGE_LIST="dense/0/image-list.txt" RELATIVE_PATH="../../../../" ' + \
                     'BASE_PATH=' + str(dataset_path.parent) + ' DATASET=' + str(dataset_path.name) + ' COLMAP_PATH=' + str(dataset_path/'colmap_output')
    os.system(register_train)
    print('Train Registration Successful')

    register_test = './register_images.sh NEW_IMAGES="video_frames_test" IMAGE_LIST="dense/0/image-list.txt" ' \
                    'RELATIVE_PATH="../../../../../" BASE_PATH=' + str(dataset_path.parent) + ' DATASET=' + str(dataset_path.name)\
                    + ' COLMAP_PATH=' + str(dataset_path/'colmap_output/colmap_output_test/')
    os.system(register_test)
    print('Test Registration Successful')

except FileNotFoundError:
    parser.print_help()
