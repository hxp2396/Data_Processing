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
def calculate_indicator_2D(pred_path, label_path,indicators=None,num_classes=2,save_file=None):
    if save_file is not None and os.path.exists(save_file):
            os.remove(save_file)
    pred_files=os.listdir(pred_path)
    label_files=os.listdir(label_path)
    assert (pred_files==label_files),'标签与预测文件不一致'
    if num_classes<=2:
        for indicator in indicators:
            results=0.0
            for file in pred_files:
                pred_file_path=os.path.join(pred_path,file)
                pred_arr=cv2.imread(pred_file_path,0)/255.
                label_file_path=os.path.join(label_path,file)
                label_arr=cv2.imread(label_file_path,0)/255.
                if label_arr.shape!=pred_arr.shape:
                    label_arr=cv2.resize(label_arr,pred_arr.shape,interpolation=cv2.INTER_NEAREST)
                result= evaluate(label_arr, pred_arr, metric=indicator)
                results+=result
            result_average=results/len(pred_files)
            print(indicator,result_average)
            if save_file is not None:
                with open(save_file, 'a') as f:  # 设置文件对象
                    f.write('{}:{}\n'.format(indicator,result_average))  # 将字符串写入文件中
    else:
        for indicator in indicators:
            results=0.0
            for file in pred_files:
                pred_file_path=os.path.join(pred_path,file)
                pred_arr=cv2.imread(pred_file_path,0)
                label_file_path=os.path.join(label_path,file)
                label_arr=cv2.imread(label_file_path,0)
                if label_arr.shape!=pred_arr.shape:
                    label_arr=cv2.resize(label_arr,pred_arr.shape,interpolation=cv2.INTER_NEAREST)
                result= evaluate(label_arr, pred_arr, metric=indicator, multi_class=True,n_classes=num_classes)
                results+=result
            result_average=results/len(pred_files)
            print(indicator,result_average)
            if save_file is not None:
                with open(save_file, 'a') as f:  # 设置文件对象
                    f.write('{}:{}\n'.format(indicator,result_average))  # 将字符串写入文件中
if __name__=='__main__':
    pred_path='../Desktop/CEnet'
    label_path="../Desktop/test"
    num_classes=2
    indicators=['DSC','Jaccard',"Accuracy",'Precision','Sensitivity','Specificity','AUC','HD']###BoundaryDistance这个指标计算非常慢
    calculate_indicator_2D(pred_path,label_path,indicators,num_classes=num_classes,save_file='test_result.txt')