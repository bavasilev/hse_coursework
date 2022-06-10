import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf

import torch
import gc

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

class CustomConfig(coco.CocoConfig):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """

    GPU_COUNT = 1
    BATCH_SIZE = 3
    IMAGES_PER_GPU = 3
    NUM_CLASSES = 3  # Background + categories
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

config = CustomConfig()
config.display()

torch.cuda.empty_cache()
gc.collect()

dataset_selected_train = coco.CocoDataset()
dataset_selected_val = coco.CocoDataset()

catIds = [27,31]
dataset_selected_train.load_coco(dataset_dir = "./coco_train", subset = "train", class_ids = catIds, year="2017", auto_download=False)
dataset_selected_val.load_coco(dataset_dir = "./coco_val", subset = "val", class_ids = catIds, year="2017", auto_download=False)

dataset_selected_train.prepare()
dataset_selected_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config_upd,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_selected_train, dataset_selected_val, 
            learning_rate=config_upd.LEARNING_RATE,
            epochs=40,
            layers='heads')

model_path = os.path.join(MODEL_DIR, "mask_rcnn_experiment_coco.h5")
model.keras_model.save_weights(model_path)

