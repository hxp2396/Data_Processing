#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time: 2024/10/24 0:02
# @File: calculate_indicator_2D.py
# @annotation:
# load libraries
import os
import cv2
import numpy as np
from miseval import evaluate
from glob import glob
pred_path='../CENet'
label_path="../Masks"

files=os.listdir(pred_path)
dices=[]
for file in files:
    pred_file_path=os.path.join(pred_path,file)
    pred_arr=cv2.imread(pred_file_path,0)/255.
    label_file_path=os.path.join(label_path,file)
    label_arr=cv2.imread(label_file_path,0)/255.
    label_arr=cv2.resize(label_arr,(224,224),interpolation=cv2.INTER_NEAREST)
    dice = evaluate(label_arr, pred_arr, metric="DSC")
    dices.append(dice)
print(np.mean(dices))
# # Get some ground truth /conannotated segmentations
# np.random.seed(1)
# real_bi = np.random.randint(2, size=(64,64))  # binary (2 classes)
# real_mc = np.random.randint(5, size=(64,64))  # multi-class (5 classes)
# # Get some predicted segmentations
# np.random.seed(2)
# pred_bi = np.random.randint(2, size=(64,64))  # binary (2 classes)
# pred_mc = np.random.randint(5, size=(64,64))  # multi-class (5 classes)
#
# # Run binary evaluation
# dice = evaluate(real_bi, pred_bi, metric="DSC")
#   # returns single np.float64 e.g. 0.75
#
# # Run multi-class evaluation
# dice_list = evaluate(real_mc, pred_mc, metric="DSC", multi_class=True,
#                      n_classes=5)
#   # returns array of np.float64 e.g. [0.9, 0.2, 0.6, 0.0, 0.4]
#   # for each class, one score