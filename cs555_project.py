# -*- coding: utf-8 -*-
"""CS555_Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fGO5rzy7yuVTpamMOrY-w24Wp39mouij
"""

# install dependencies: 
!pip install pyyaml==5.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab

# install detectron2: (Colab has CUDA 10.1 + torch 1.7)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
assert torch.__version__.startswith("1.7")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from google.colab import drive
drive.mount('/content/gdrive')

!ls "/content/gdrive/My Drive/CS555 Project"

!cp "/content/gdrive/My Drive/CS555 Project/detectron2-master/configs/DLA_mask_rcnn_R_101_FPN_3x.yaml" "DLA_mask_rcnn_R_101_FPN_3x.yaml"

!cp "/content/gdrive/My Drive/CS555 Project/model_final_trimmed.pth" "model_final_trimmed.pth"

!cp -r "/content/gdrive/My Drive/CS555 Project/detectron2-master" "detectron2-master"

# Commented out IPython magic to ensure Python compatibility.
!ls
# %cd detectron2-master
!ls
# %cd demo
!ls
# %cd detectron2-master

# Commented out IPython magic to ensure Python compatibility.

# %cd "/content/detectron2-master"

!cp "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images/00001276.tif" "00001276.tif"

!python demo/demo.py --config-file configs/DLA_mask_rcnn_R_101_FPN_3x.yaml --input "/content/detectron2-master/00001276.tif"  --output output_predicted5 --confidence-threshold 0.5 --opts MODEL.WEIGHTS "/content/model_final_trimmed.pth" MODEL.DEVICE cpu

img = cv2.imread("output_predicted5.png")
cv2_imshow(img)

from detectron2.data.datasets import register_coco_instances
register_coco_instances("dla_train", {}, "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/annotations.json", "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images")
register_coco_instances("dla_test", {}, "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/annotations.json", "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images")

from detectron2.data.datasets import register_coco_instances
register_coco_instances("dla_train", {}, "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/annotations.json", "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images")
register_coco_instances("dla_test", {}, "/content/gdrive/My Drive/CS555 Project/publaynet/val.json", "/content/gdrive/My Drive/CS555 Project/publaynet/val")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/detectron2-master

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file("configs/DLA_mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TEST = ("dla_test",)
cfg.DATASETS.TRAIN = ("dla_train",)
cfg.MODEL.WEIGHTS = "/content/model_final_trimmed.pth"
#cfg.DATALOADER.NUM_WORKERS = 4
#cfg.SOLVER.IMS_PER_BATCH = 4
#cfg.SOLVER.BASE_LR = 0.001

#cfg.SOLVER.WARMUP_ITERS = 1000
#cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
#cfg.SOLVER.STEPS = (1000, 1500)
#cfg.SOLVER.GAMMA = 0.05

#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

#cfg.TEST.EVAL_PERIOD = 500

from detectron2.data import MetadataCatalog
MetadataCatalog.get("dla_val").thing_classes = ['Text','Title','List','Table','Figure']

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/detectron2-master/

!mkdir output

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
#predictor = DefaultPredictor(cfg)
#predictor.resume_or_load(resume=True)
evaluator = COCOEvaluator("dla_test", cfg, False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "dla_test")

inference_on_dataset(trainer.model, test_loader, evaluator)

# Commented out IPython magic to ensure Python compatibility.
# %cd output

!cp coco_instances_results.json "/content/gdrive/My Drive/CS555 Project/coco_instances_results.json"

!cp instances_predictions.pth "/content/gdrive/My Drive/CS555 Project/instances_predictions.pth"

#!python demo/demo.py --config-file configs/DLA_mask_rcnn_R_101_FPN_3x.yaml --input "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images/00000398.tif"  --output output_predicted5 --confidence-threshold 0.5 --opts MODEL.WEIGHTS "/content/model_final_trimmed.pth" MODEL.DEVICE cpu
!python demo/demo.py --config-file configs/DLA_mask_rcnn_R_101_FPN_3x.yaml --input "/content/gdrive/My Drive/CS555 Project/PRImA Layout Analysis Dataset/Images/00000398.tif"  --output output_predicted5 --confidence-threshold 0.5 --opts MODEL.WEIGHTS "/content/model_final_trimmed.pth" MODEL.DEVICE cpu

img = cv2.imread("output_predicted5.png")
cv2_imshow(img)

!cp -r output "/content/gdrive/My Drive/CS555 Project/output"