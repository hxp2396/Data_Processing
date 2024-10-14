#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  :
# @annotation :这个文件的目的是为了匹配groundtruth（真实标签）和detection-results（预测得到的标签）一一对应
import sys
import os
import glob
# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
def backup(src_folder, backup_files, backup_folder):
    # non-intersection files (txt format) will be moved to a backup folder不同时存在两个文件夹的文件将会被移动到backup文件夹
    if not backup_files:
        print('No backup required for', src_folder)
        return
    os.chdir(src_folder)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + '/' + file)
def intersect_gt(GT_PATH,DR_PATH,backup_folder):
    os.chdir(GT_PATH)#将当前工作目录设置到ground truth
    gt_files = glob.glob('*.tif')#获得当前文件夹下的所有.txt文件
    if len(gt_files) == 0:
        print("Error: no .txt files found in", GT_PATH)
        sys.exit()
    os.chdir(DR_PATH)#将当前工作目录设置到detection-reult
    dr_files = glob.glob('*.tif')
    if len(dr_files) == 0:
        print("Error: no .txt files found in", DR_PATH)
        sys.exit()
    gt_files = set(gt_files)#将得到的文件名列表用set集合包裹，利用set的不重复性，进行删除不同时在两个集合存在的文件
    dr_files = set(dr_files)
    print('total ground-truth files:', len(gt_files))
    print('total detection-results files:', len(dr_files))
    print()
    #去除两个集合中重复的文件
    gt_backup = gt_files - dr_files
    dr_backup = dr_files - gt_files
    backup(GT_PATH, gt_backup, backup_folder)
    backup(DR_PATH, dr_backup, backup_folder)
    if gt_backup:
        print('total ground-truth backup files:', len(gt_backup))
    if dr_backup:
        print('total detection-results backup files:', len(dr_backup))
    intersection = gt_files & dr_files  # 求交集
    print('total intersected files:', len(intersection))
    print("Intersection completed!")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    GT_PATH = r'E:\DataCollection\XXX\cut\GroundTruth'
    DR_PATH = r'E:\DataCollection\XXX\cut\TissueImages'
    backup_folder = 'backup_no_matches_found' # must end without slash