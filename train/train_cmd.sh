#!/bin/bash

SINGLE_DIR=~/ROB590_WS/Deep_Object_Pose/train/training_data/Single_object
MULTI_OBJ_DIR=~/ROB590_WS/Deep_Object_Pose/train/training_data/Multi_object

DATA_ARGS=""
for i in {5..10}; do
  DATA_ARGS+=" $SINGLE_DIR/$i"
done
for i in {5..6}; do
  DATA_ARGS+=" $MULTI_OBJ_DIR/$i"
done

CMD="python -m torch.distributed.launch --nproc_per_node=1 train.py --data $DATA_ARGS"

# Append other arguments
# CMD+=" --object Block_w_sandpaper --namefile Block_w_sandpaper_weight --epoch 120 --batchsize 16 --pretrained --net_path ~/ROB590_WS/Deep_Object_Pose/train/pretrained_models/sugar_60.pth --loginterval 0.25"
# --net_path ~/ROB590_WS/Deep_Object_Pose/train/pretrained_models/sugar_60.pth
#  --pretrained \
#  --net_path ~/ROB590_WS/Deep_Object_Pose/train/output/train_1_weights/Block_110.pth \
ARGS="--object Block_w_sandpaper \
 --namefile Block \
 --epoch 190 \
 --batchsize 16 \
 --lr 0.0003 \
 --pretrained \
 --net_path ~/ROB590_WS/Deep_Object_Pose/train/output/weights/Block_180.pth \
 --loginterval 30"
 
# loginterval: print log every XX batches

CMD+=" $ARGS"

# Print and run
echo "Running DOPE training with command:"
echo $CMD
eval $CMD
