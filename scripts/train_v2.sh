#!/bin/bash

# 训练参数
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
SCHEDULER="cosine"
WARMUP_EPOCHS=5

# 模型参数
PATCH_SIZE=32
GRID_SIZE=4
SEQUENCE_LENGTH=16

# 数据增强参数
USE_AUGMENTATION=true
RANDOM_ROTATION=15.0
RANDOM_HORIZONTAL_FLIP=0.5
COLOR_JITTER=0.4
RANDOM_ERASING=0.2

# 路径设置
DATA_ROOT="../datasets"
RESNET_PATH="../pretrained/resnet50.pth"
CONVNEXT_PATH="../pretrained/convnext_base.pth"
CHECKPOINT_DIR="../checkpoints"
LOG_DIR="../logs"

# 创建必要的目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# 启动训练
python main.py \
    --mode train \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --scheduler $SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    --patch_size $PATCH_SIZE \
    --grid_size $GRID_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --resnet_path $RESNET_PATH \
    --convnext_path $CONVNEXT_PATH \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --use_augmentation $USE_AUGMENTATION \
    --random_rotation $RANDOM_ROTATION \
    --random_horizontal_flip $RANDOM_HORIZONTAL_FLIP \
    --color_jitter $COLOR_JITTER \
    --random_erasing $RANDOM_ERASING 