# Appearance Editing of Captured Objects
Our method can edit the appearance of captured objects. This is a guide through different scripts and how to run them

## Basic Installation
+ mitsuba2 : Differentiable renderer for material optimization
+ colmap : Sfm software to recover geometry and poses
+ redner : UV maps are retrieved through redner. Although you can also use Blender smart uv unwrap to unwrap the geometry
+ meshlab : Used to remove the surface of the object as well as other extra points

## Geometry Recovery
### Synthetic: 
In this case we have the geometry and need to unwrap the geometry
### Real: 
We only have the images as input. To retrive the geometry we use COLMAP Sfm software. We first take a subset of images and retrieve geometry from it. We register the remaining images onto the retrieved geometry and get their poses as well.  
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
- Create a mitsuba scene file,`scene.xml` with diffuse material model
- Generate train and test video frames and extrinsics
    - Synthetic: `python data/synthetic_generate.py --scene_file <scene_file.xml> --vector_scale <scale> --output_dir <output_dir> --img_width <width --img_height <height> --spp <samples>`
    - Real: Sample frames by `python data/extract_frames.py --video_file <video_file> --output_dir <output_dir> --skip <sample_rate> --img_width <width> --img_height <height>`


### Colmap
- To generate geometry run:   
`data/colmap_reconstruction/reconstruct.sh <dataset>` 
- To register images run:
`data/colmap_reconstruction/register_images.sh NEW_IMAGES=<remaining_frames> IMAGE_LIST=<path_to_list_of_new_images> RELATIVE_PATH=<relative_to_images_dir> BASE_PATH=<base_dataset_dir> DATASET=<dataset_name> COLMAP_PATH='<colmap_output_dir>'`   
- If you are using GUI:
    - Follow https://journal.missiondata.com/lab-notes-a-recipe-for-using-photogrammetry-to-create-3d-model-7c256e38b1fb
    - Set shared intrinsics
    - (For manual as well) Set feature matching to sequential if frames are from a video  

### Meshlab
- Quadratic edge collapse decimation
- Remove non-manifold edges/vertices
- Close holes
- Export as obj (FISH.obj)

### UV unwrapping
`python data/uv_redner.py --input_file /media/aakash/wd1/DATASETS/FISH/cleaned.obj --output_file /media/aakash/wd1/DATASETS/FISH/unwrapped.obj`

### Export
Export as .ply with Y up and -Z forward

## Material Optimization
We now have the geometry,poses, environment map and input images. Using this we can do material optimization. Steps for both synthetic and real are same.

### Preprocessing
- Make `scene_dr.xml` mitsuba scene description file
- Create directories `optimized_textures`,`dr_tensorboard`,`dr_log`

### Mask Frames
We use U2Net for generating ground truth masks in case of real scenes. 
- Synthetic: `python data/create_synthetic_frame_mask.py --data_dir <dataset_dir> --frames_dir <data_dir/input_dir> --output_dir <data_dir/output_dir>`
- Real : `python create_frame_mask.py --data_dir <dataset_dir> --frames_dir <data_dir/input_dir> --output_dir <data_dir/output_dir> --model_path U2Net/saved_models --model_name u2net`




