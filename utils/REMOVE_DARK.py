#coding:utf-8
import os
import hashlib  #hash函数
import shutil
import cv2
import numpy as np
from glob import glob
import os
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))
LABEL_PATH='./valid/Masks'
files=glob(LABEL_PATH+'/*.png')
dark_img_path='./valid/Dark'
os.makedirs(dark_img_path,exist_ok=True)
# print(files)
for file in files:
    # os.path.dirname(s)  # 输出为 './Masks'
    # filename = os.path.splitext(file)  输出为('./Masks\\study_0304.nii.gz_9', '.png')
    filename=os.path.basename(file)  # 输出为 study_0304.nii.gz_9.png
    array=cv2.imread(file,0)
    print(np.unique(array))
    leng=len(np.unique(array))
    print(leng)
    if leng==1:
        mymovefile(file,os.path.join(dark_img_path,filename))
        print("sasas")