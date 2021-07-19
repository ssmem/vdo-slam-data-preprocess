#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:28:21 2019

@author: lxh
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
# Root directory of the project
ROOT_DIR = os.getcwd();#os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import glob as gb

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "./")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights("./mask_rcnn_coco.h5", by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


img_path = gb.glob("./images/*.png")
for path in img_path:
    image = cv2.imread(path)
    results = model.detect([image], verbose=1)
    r = results[0]
    masks = r['masks'].transpose(2,0,1)
    with open(path+".mask", 'w') as f:    # 打开test.txt   如果文件不存在，创建该文件。
        f.write(str(masks.shape[1])+" "+str(masks.shape[2])+" "+str(masks.shape[0])+"\n")
        for i in range(len(r['class_ids'])):
            x = class_names[r['class_ids'][i]]
            f.write(str(x)+"\n")
        maskresult = np.zeros([masks.shape[1],masks.shape[2]], dtype=np.int8)
        for i in range(len(r['class_ids'])):
            masks[i] = r['class_ids'][i] * masks[i]
            maskresult = maskresult + masks[i]
        maskresult[maskresult==0] = -1
        np.savetxt(path+".mask", maskresult, fmt='%d',delimiter=' ', newline='\r\n')






    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    #cv2.imwrite(path[:-4]+"_mask.jpg", res_img)


 














 
