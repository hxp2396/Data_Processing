#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 10:14
# @File :
# @annotation :用于比较两个文件夹内的同名文件，若是不能一一对应则将落单的文件另外存放
import hashlib  # hash函数
import shutil
import os, sys
def makesDir(filepath):  # 判断如果文件不存在,则创建
    if not os.path.exists(filepath):
        os.makedirs(filepath)
# ------------Python对比两个文件夹中文件并筛选出差异文件----------------------------------
def Compare_folder_1(A,B,C,D):
    md5dict = {}
    md5dict2 = {}
    for filename in os.listdir(A):  # 返回的是一个一个的文件名
        hashvalue = hashlib.md5(filename.encode('utf-8')).hexdigest()
        md5dict[hashvalue] = os.path.join(A, filename)
    for filename in os.listdir(B):
        hashvalue = hashlib.md5(filename.encode('utf-8')).hexdigest()
        if hashvalue not in md5dict:
            shutil.copy(os.path.join(B, filename), os.path.join(C, filename))
            os.remove(os.path.join(B, filename))
    for filename in os.listdir(B):  # 返回的是一个一个的文件名
        hashvalue = hashlib.md5(filename.encode('utf-8')).hexdigest()
        md5dict2[hashvalue] = os.path.join(B, filename)
    for filename in os.listdir(A):
        hashvalue = hashlib.md5(filename.encode('utf-8')).hexdigest()
        if hashvalue not in md5dict2:
            shutil.copy(os.path.join(A, filename), os.path.join(D, filename))
            os.remove(os.path.join(A, filename))
    shutil.move(C, B + '/' + C)
    shutil.move(D, A + '/' + D)

def Compare_folder_2(path1, path2,diff_path):
    path = diff_path
    fileName1 = set([_ for _ in os.listdir(path1)])
    fileName2 = set([_ for _ in os.listdir(path2)])
    diffs = fileName1.difference(
        fileName2)  # fileName1对比fileName2，fileName1中多出来的文件；注意，如果fileName2里有fileName1中没有的文件，也不会筛选出来
    filePath = [os.path.join(path, i) for i in diffs]
    if not os.path.exists(path):
        os.mkdir(path)
    for file in filePath:
        fileName = file.split('/')[-1]
        shutil.copy(os.path.join(path1, fileName), '/'.join(file.split('/')[:-1]))
        print('复制文件--', fileName)
# --------------------------------------------------------------------------------------------------------
def reportdiff(unique1, unique2, dir1, dir2): #比较两个目录里文件是否一致？1.要列出2个文件夹中所有的文件；2.相互做比较；3.把比较的结果以报告的形式呈现
    '''生成目录差异化报告'''
    if not (unique1 or unique2):
        print("Directory lists are identical")
    else:
        if unique1:
            print('Files unique to：', dir1)
            for file in unique1:
                print('....', file)
        if unique2:
            print('Files unique to：', dir2)
            for file in unique2:
                print('.........', file)
def difference(seq1, seq2):
    '''仅返回seq1中的所有项'''
    return [item for item in seq1 if item not in seq2]
def Compare_folder_3(dir1, dir2, files1=None, files2=None):
    '''比较文件的名字'''
    print('Comparing...', dir1, 'to....', dir2)
    files1 = os.listdir(dir1) if files1 is None else files1
    files2 = os.listdir(dir2) if files2 is None else files2
    unique1 = difference(files1, files2)
    unique2 = difference(files2, files1)
    reportdiff(unique1, unique2, dir1, dir2)
    return not (unique1, unique2)

if __name__ == '__main__':
    A = './Images'  # 代表是目录A
    B = './Ground-truths'  # 代表是目录B
    C = 'B_not_in_A'  # 代表是目录A和目录B的差集放在目录C
    D = 'A_not_in_B'
    makesDir(C)
    makesDir(D)
    Compare_folder_1(A, B, C, D)
# ------
    dir1, dir2 = '',''
    Compare_folder_3(dir1, dir2)

