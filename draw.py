# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="ticks", palette="pastel")
#
# # Load the example tips dataset
# flights = sns.load_dataset("normalize")
#
# flights_wide = flights.pivot(index="database", columns="set_type", values="mean")
#
# sns.boxplot(x="database", y="mean", hue="set_type",
#                  data=flights, palette="Set2")
#
#
#
# plt.show()


# ------------------------------------

# import sys
# import math
# import numpy as np
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# def opendata(file,dataset,softname):
#     true_list = []
#     pred_list = []
#     ii = 0
#     soft_fu = ['SIFT_score','PROVEAN_score','LRT_score','FATHMM_score','SIFT_RAW','SIFT']
#
#     soft_id = get_soft_id(softname)
#     for line in open(file,'r'):
#         line = line.strip('\n')
#         if "pos" in line:
#             continue
#         tmp = line.split('\t')
#
#         if tmp[soft_id] == "" or tmp[soft_id] == "." :
#             continue
#         else:
#             pb = tmp[-2]
#
#         if tmp[soft_id] == "" or tmp[soft_id] == ".":
#             continue
#         elif softname == "SIFT":
#             s_pred = 0-float(tmp[soft_id])
#         else:
#             s_pred = float(tmp[soft_id])
#
#         if dataset in tmp[-3]:
#             true_list.append(pb)
#             pred_list.append(s_pred)
#             ii +=1
#     return true_list,pred_list
#
# def get_soft_id(name):
#     head = ["#chr","pos",".","ref","alt","CADD","ClinPred","MCAP","MetaLR","Polyphen2","PrimateAI","REVEL","SIFT","data","P_B"]
#     i = 0
#     hash={}
#     for id in head:
# #       print(id)
#         hash[id] = i
#         i+=1
#     return hash[name]
#
# (file,dataset,i)=(sys.argv[1],sys.argv[2],sys.argv[3])
# auc=[]
# name={}
#
# for softname in ["CADD","ClinPred","MCAP","MetaLR","Polyphen2","PrimateAI","REVEL","SIFT"]:
#     true_list,pred_list = opendata(file,dataset,softname)
#     fpr, tpr, thresholds = metrics.roc_curve(true_list,pred_list,pos_label='Pathogenic')
#     roc_auc = metrics.auc(fpr, tpr)
#     name[roc_auc]=softname
#     auc.append(roc_auc)
#
#
# for roc_auc in sorted(list(set(auc)),reverse=True):
#     softname = name[roc_auc]
#     print (softname)
#     true_list,pred_list = opendata(file,dataset,softname)
#     fpr, tpr, thresholds = metrics.roc_curve(true_list,pred_list,pos_label='Pathogenic')
#     plt.plot(fpr, tpr, lw=1,label='%s=%0.4f' % (softname, roc_auc))
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig(sys.argv[2] + sys.argv[3] + '.png')
# plt.show()
# --------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import os
import warnings;warnings.filterwarnings('ignore')

from glob import glob
filelist=glob('Auc/*_label.npy')
# print(filelist)
for file in filelist:
    modelname=os.path.split(file)[-1].split('_label')[0]
    y_test=np.load(file)
    y_test=y_test.flatten()
    pred=np.load(file.replace('label','pred'))
    pred=pred.flatten()
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5,label = '%s AUC = %0.5f' %(modelname,roc_auc) )

    # print(modelname)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# y_test=np.load('label.npy')
# y_test=y_test.flatten()
# pred=np.load('pre.npy')
# pred=pred.flatten()
# #############画图部分
# y_test1=np.load('label1.npy')
# y_test1=y_test1.flatten()
# pred1=np.load('pre1.npy')
# pred1=pred1.flatten()
#
# y_test2=np.load('DDUnet_label.npy')
# y_test2=y_test2.flatten()
# pred2=np.load('DDUnet_pred.npy')
# pred2=pred2.flatten()
#
# fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
#
# fpr1, tpr1, threshold1 = metrics.roc_curve(y_test1, pred1)
# roc_auc1 = metrics.auc(fpr1, tpr1)
#
# fpr2, tpr2, threshold2 = metrics.roc_curve(y_test2, pred2)
# roc_auc2 = metrics.auc(fpr2, tpr2)
#
# plt.figure(figsize=(6,6))
#
#
# plt.title('Test ROC curve')
# plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
# plt.plot(fpr1, tpr1, 'r', label = 'test AUC = %0.3f' % roc_auc1)
# plt.plot(fpr2, tpr2, 'g', label = 'AA AUC = %0.3f' % roc_auc2)
#
#
#
#
#
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
