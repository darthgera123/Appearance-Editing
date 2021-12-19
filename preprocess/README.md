# Appearance Editing of Captured Objects
Our method can edit the appearance of captured objects. This is a guide through different scripts and how to run them

## Basic Installation
+ [mitsuba2](https://mitsuba2.readthedocs.io/en/latest/index.html) : Differentiable renderer for material optimization
+ [colmap](https://colmap.github.io/) : Sfm software to recover geometry and poses
+ [redner](https://github.com/BachiLi/redner) : UV maps are retrieved through redner. Although you can also use [Blender](https://www.blender.org/) smart uv unwrap to unwrap the geometry
+ [meshlab](https://www.meshlab.net/) : Used to remove the surface of the object as well as other extra points

## Geometry Recovery
### Synthetic: 
In this case we have the geometry and need to unwrap the geometry
### Real: 
We only have the images as input. To retrive the geometry we use COLMAP Sfm software. We first take a subset of images and retrieve geometry from it. We register the remaining images onto the retrieved geometry and get their poses.  
After that we need to clean it in meshlab and unwrap it.  
### Preprocessing
- Create directories 
    - `video_frames`
    - `video_frames_test` 
    - `video_frames_mask`
    - `video_frames_test_mask`
    - `video_frames_test_extrinsics`
    - `video_frames_extrinsics`
    - `colmap_capture`
    - `colmap_output`
- Save Environment Map as `envmap.exr`
- Create a mitsuba scene file,`scene.xml` with diffuse material model (only needed for Independent Optimization)
- Generate train and test video frames and extrinsics
    - Synthetic: `python data/synthetic_generate.py --scene_file <scene_file.xml> --vector_scale <scale> --output_dir <output_dir> --img_width <width --img_height <height> --spp <samples>`
    - Real: Sample frames by `python data/extract_frames.py --video_file <video_file> --output_dir <output_dir> --skip <sample_rate> --img_width <width> --img_height <height>`


### [Colmap](https://colmap.github.io/)
- Run the following script to retrieve geometry of the object as well as poses of the scene.
`python preprocess_real.py --train_video --test_video --width --height`
- To register images run:
`data/colmap_reconstruction/register_images.sh NEW_IMAGES=<remaining_frames> IMAGE_LIST=<path_to_list_of_new_images> RELATIVE_PATH=<relative_to_images_dir> BASE_PATH=<base_dataset_dir> DATASET=<dataset_name> COLMAP_PATH='<colmap_output_dir>'`
- More instructions present in [`image_registration.txt`](./image_registration.txt)
- If you are using GUI:
    - Follow [this tutorial](https://journal.missiondata.com/lab-notes-a-recipe-for-using-photogrammetry-to-create-3d-model-7c256e38b1fb)
    - Set shared intrinsics
    - (For manual as well) Set feature matching to sequential if frames are from a video  

### Meshlab
- Quadratic edge collapse decimation
- Remove non-manifold edges/vertices
- Close holes
- Export as obj (FISH.obj)

### UV unwrapping
You can either use Blender's [smart uv unwrap](https://www.youtube.com/watch?v=illIxYKb-1k) to recover the uv mapping. We also provide a script to unwrap scenes using `redner` library.
`python data/uv_redner.py --input_file /media/aakash/wd1/DATASETS/FISH/cleaned.obj --output_file /media/aakash/wd1/DATASETS/FISH/unwrapped.obj`

### Export
Export object in blender as .ply with Y up and -Z forward

## Mask Frames
We use [U2Net](https://github.com/xuebinqin/U-2-Net) for generating ground truth masks in case of real scenes. 
- Synthetic: `python data/create_synthetic_frame_mask.py --data_dir <dataset_dir> --frames_dir <data_dir/input_dir> --output_dir <data_dir/output_dir>`
- Real : `python create_frame_mask.py --data_dir <dataset_dir> --frames_dir <data_dir/input_dir> --output_dir <data_dir/output_dir> --model_path U2Net/saved_models --model_name u2net`

## Material Optimization
Skip this if you want to perform joint optimizations.
We now have the geometry,poses, environment map and input images. Using this we can do material optimization. Steps for both synthetic and real are same.
- MAKE 'optimized_textures' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

- MAKE 'dr_tensorboard' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

- MAKE 'dr_log' DIRECTORY
  - Make a subdirectory 'colmap_geometry'
  - Make another subdirectory 'perfect_geometry'

- Recover material from synthetic scene
`python data/synthetic_dr_perfect.py --scene_file <scene_file.xml>  --data_dir <output_dir> --epochs 20`

- Recover material from real scene
`python data/real_dr.py --scene_file <scene_file.xml> --data_dir <input_dir> --image_list_txt <img_list.txt> --epochs 20 --img_width 480 --img_height 270`

- Generate training data for optimized texture and geometry
`python data/synthetic_extract_perfect.py --scene_file <scene_file> --data_dir <input_dir> --output_dir <output_dir>`

## Joint Optimization data
- MAKE `Parent` DIRECTORY
  - Make `train` and `test` subdirectories
  - In both 'train' and 'test', create 7 directories
    - extrinsics
    - frames
    - forward
    - uv
    - uv_png
    - mask
    - sh
    - env
- GET ALIGNMENT VECTOR
    - Open the .ply model in blender
    - Copy paste the code from 'data/blender_alignment_vec.py' inside blender python texgt editor
    - Run the script
    - The output will be displayed on the command line, which is the normal vector to be aligned
    - If you dont have command line output, start blender from the command line to see it
- Generate Real scenes data  
`python data/real_extract_joint.py --scene_file <scene.xml> --data_dir <input_dir> --output_dir <output_dir> --train_image_list_txt <image_list> --test_image_list_txt <test_image_list> --img_width 960 --img_height 540 --alignment_x 0.043 --alignment_y -0.7631 --alignment_z -0.644`  
The alignment_vec_* params should be replaced with the vector obtained in the "GET ALIGNMENT VECTOR" step




