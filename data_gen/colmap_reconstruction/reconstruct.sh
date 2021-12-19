#!/bin/bash

DATASET_PATH=$1

echo $1

colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images

colmap exhaustive_matcher --database_path $DATASET_PATH/database.db --ExhaustiveMatching.block_size 5

#mkdir $DATSET_PATH/sparse

colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse

#mkdir $DATASET_PATH/dense

colmap image_undistorter --image_path $DATASET_PATH/images --input_path $DATASET_PATH/sparse/0 --output_path $DATASET_PATH/dense --output_type COLMAP --max_image_size 700

colmap patch_match_stereo --workspace_path $DATASET_PATH/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 700

colmap stereo_fusion --workspace_path $DATASET_PATH/dense --workspace_format COLMAP --input_type geometric --output_path $DATASET_PATH/dense/fused.ply  

colmap poisson_mesher --input_path $DATASET_PATH/dense/fused.ply --output_path $DATASET_PATH/dense/meshed-poisson.ply
