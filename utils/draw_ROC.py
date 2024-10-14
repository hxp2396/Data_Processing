#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  :
# @annotation :根据生成的npy文件画roc曲线图
import numpy as np
from sklearn import metrics
import pylab as plt
import os
import warnings;
warnings.filterwarnings('ignore')
from glob import glob

def draw_ROC_curve(filelist):
    # print(filelist)
    for file in filelist:
        modelname = os.path.split(file)[-1].split('_label')[0]
        y_test = np.load(file)
        y_test = y_test.flatten()
        pred = np.load(file.replace('label', 'pred'))
        pred = pred.flatten()
        fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label='%s AUC = %0.5f' % (modelname, roc_auc))
        # print(modelname)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == '__main__':
    filelist = glob('Auc/*_label.npy')
    draw_ROC_curve(filelist)
