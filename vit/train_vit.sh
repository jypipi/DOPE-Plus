#!/bin/bash

# ############### Block
# SINGLE_DIR=~/Deep_Object_Pose/train/training_data/Single_object
# MULTI_OBJ_DIR=~/Deep_Object_Pose/train/training_data/Multi_object

# DATA_ARGS=""
# for i in {5..48}; do
#   DATA_ARGS+=" $SINGLE_DIR/$i"
# done
# for i in {51..79}; do
#   DATA_ARGS+=" $SINGLE_DIR/$i"
# done
# # for i in {0..6}; do
# #   DATA_ARGS+=" $MULTI_OBJ_DIR/$i"
# # done

# VAL_DATA_DIR=~/Deep_Object_Pose/train/training_data/eval_data
# VAL_DATA_ARGS=""
# for i in {80..82}; do
#   VAL_DATA_ARGS+=" $VAL_DATA_DIR/$i"
# done
# for i in {49..50}; do
#   VAL_DATA_ARGS+=" $VAL_DATA_DIR/$i"
# done

############### Cookies
TRAINING_SYNC_DIR=/home/jeff/Deep_Object_Pose/train/training_data/cookies_imgs
TRAINING_REAL_DIR=~/Deep_Object_Pose/train/training_data/hope-dataset/hope_video

DATA_ARGS=""
for i in {1..38}; do
  DATA_ARGS+=" $TRAINING_SYNC_DIR/$i"
done
for i in {0..5}; do
  DATA_ARGS+=" $TRAINING_REAL_DIR/scene_000$i"
done
DATA_ARGS+=" $TRAINING_REAL_DIR/scene_0009"


VAL_DATA_DIR=~/Deep_Object_Pose/train/training_data/hope-dataset/hope_image/valid
VAL_DATA_ARGS=""
for i in 1 3 6 7; do
  VAL_DATA_ARGS+=" $VAL_DATA_DIR/scene_000$i"
done
for i in {46..47} {50..51}; do
  VAL_DATA_ARGS+=" $TRAINING_SYNC_DIR/$i"
done

############## Training Command
CMD="python -m torch.distributed.launch --nproc_per_node=1 train_vit.py --data $DATA_ARGS --val_data $VAL_DATA_ARGS"

# Append other arguments
#  --pretrained \
#  --net_path ~/Deep_Object_Pose/train/output/weights/Block_190.pth \
#  --pretrained \
#  --net_path ~/Deep_Object_Pose/train/pretrained_models/best_500.pth \

# ####### Block
# ARGS="--object Block_w_sandpaper \
#  --namefile Block \
#  --epoch 400 \
#  --batchsize 64 \
#  --lr 0.00005 \
#  --outf "output_vit/weights"
#  --loginterval 10"
 
# # loginterval: print log every XX batches

####### Cookies
ARGS="--object Cookies \
 --namefile Cookies \
 --epoch 200 \
 --batchsize 64 \
 --lr 0.00005 \
 --outf "output_cookies_vit/weights_train3" \
 --obj_model_path "/home/jeff/Deep_Object_Pose/data_generation/models/Cookies/textured.obj" \
 --loginterval 10"

CMD+=" $ARGS"

# Run
echo "Running DOPE training with command:"
echo $CMD
eval $CMD
