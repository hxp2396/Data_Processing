#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 16:36
# @File : list_subdir_and_files.py
# @annotation:# -*----------------------------------------------分别遍历一个目录中的所有子目录和所有文件
import os
file = []
dir = []
def list_dir(start_dir):
    dir_res = os.listdir(start_dir)
    for path in dir_res:
        temp_path = start_dir + '/' + path
        if os.path.isfile(temp_path):
            file.append(temp_path)
        if os.path.isdir(temp_path):
            dir.append(temp_path)
            list_dir(temp_path)
def list_subdir_and_files(start_dir):
    list_dir(start_dir)
    return file,dir

if __name__ == '__main__':
    startdir=''
    file,dir = list_subdir_and_files(startdir)
