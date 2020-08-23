# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:45:26 2019

@author: Chaobo
"""

import os
import numpy as np

# set up path for dataset and weight
MODEL_PATH   = os.path.join('*** FILE PATH ***','DIS-YOLO')
DATASET      = os.path.join(MODEL_PATH, 'data')
OUTPUT_DIR   = os.path.join(MODEL_PATH, 'output')
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'pretrained_weights', 'yolov3_3class_coco.ckpt')
#WEIGHTS_FILE = os.path.join(MODEL_PATH, 'pretrained_weights', 'model.ckpt-10000')

GPU = '0'

# classes, and anchors from dimension clustering on image size of 576
CLASSES = ['crack', 'spall', 'rebar']
ANCHORS = np.array([[31,23], [62,58], [143,91], [213,186], [61,337], [194,432], [474,248], [551,93], [478,454]], dtype=np.float32)

# data augmentation
FLIPPED          = True
BLUR_NOISE_LIGHT = True


'''training and testing settings'''
# change for different training stages
MAX_ITER       = 10000

# summarrize and save steps
SUMMARY_ITER   = 50
SAVE_ITER      = 500

# alpha for leaky_relu function
ALPHA          = 0.1 

# train batch size, train image size, number of position-sensitive score maps
BATCH_SIZE     = 2
IMAGE_SIZE     = 576
K_MAP          = 3

# number of grid cells on lowest-resolution output map of YOLOv3
BASE_GRID      = int(IMAGE_SIZE / 32)

# train loss parameters
OBJECT_SCALE   = 2.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE    = 1.0
COORD_SCALE    = 1.0
MASK_SCALE     = 5.0
SCORE_SCALE    = 2.0

# ignore confidence loss for good predicted boxes (iou > 0.5)
IGNORE_THRESH = 0.5 

# threshold for class-specific confidence score
OBJ_THRESHOLD = 0.25

# iou threshold for non-maximum supression
IOU_THRESHOLD = 0.3

# test image size
TEST_SIZE = 576

# maximum number of ground-truth objects per image for training
MAX_BOX_PER_IMAGE = 20

# maximum number of proposed boxes for training mask subnet and output final predictions
MAX_DETECTION = 30
