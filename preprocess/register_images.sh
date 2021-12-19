#!/bin/bash
#bash my_scripts.sh  STEPS="ABC" REPOSITORY_NAME="stackexchange"
BASE_PATH=/media/aakash/wd1/DATASETS
DATASET=MONKEY_SYNTHETIC
NEW_IMAGES=dense/0/training_frames
IMAGE_LIST=dense/0/image-list.txt
DATABASE_PATH=database.db
IMAGES=dense/0/images
INPUT_PATH=dense/0/sparse/
OUTPUT_PATH=dense/0/new_sparse
VOCAB_BIN=vocab.bin
RELATIVE_PATH=../
COLMAP_PATH=$BASE_PATH/$DATASET/"colmap_output"
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            
	    NEW_IMAGES)         NEW_IMAGES=${VALUE} ;;
	    IMAGE_LIST)         IMAGE_LIST=${VALUE} ;;
	    DATABASE_PATH)      DATABASE_PATH=${VALUE} ;;
	    IMAGES)             IMAGES=${VALUE} ;;
	    INPUT_PATH)         INPUT_PATH=${VALUE} ;;
	    OUTPUT_PATH)        OUTPUT_PATH=${VALUE} ;;
	    VOCAB_BIN)          VOCAB_BIN=${VALUE} ;;
	    CREATE_VOCAB_BIN)   CREATE_VOCAB_BIN=${VALUE} ;;
		RELATIVE_PATH)      RELATIVE_PATH=${VALUE} ;;
		BASE_PATH)      	BASE_PATH=${VALUE} ;;
		DATASET)      		DATASET=${VALUE} ;;
		COLMAP_PATH)      	COLMAP_PATH=${VALUE} ;;
            *)
    esac


done


DATASET_PATH=$BASE_PATH/$DATASET

# Image-list and frames dir are relative to dataset_path
# the rest are relative to colmap path
if [ ! -d "$COLMAP_PATH/$OUTPUT_PATH" ]; then
  echo "Creating new_sparse"

  mkdir $COLMAP_PATH/$OUTPUT_PATH
fi

python3 create_list.py \
			--dataset_dir $DATASET_PATH \
			--frames_dir $NEW_IMAGES \
			--output_file $COLMAP_PATH/$IMAGE_LIST \
			--relative_path $RELATIVE_PATH
echo "image-list created"


colmap feature_extractor \
			--database_path $COLMAP_PATH/$DATABASE_PATH \
			--image_path $COLMAP_PATH/$IMAGES \
			--image_list_path $COLMAP_PATH/$IMAGE_LIST \
			--ImageReader.existing_camera_id 1 \
			--ImageReader.single_camera 1
echo "Feature done"

colmap exhaustive_matcher --database_path $COLMAP_PATH/$DATABASE_PATH
echo "Matching done"

colmap image_registrator \
			--database_path $COLMAP_PATH/$DATABASE_PATH \
			--input_path $COLMAP_PATH/$INPUT_PATH \
			--output_path $COLMAP_PATH/$OUTPUT_PATH
echo "Registration done"

