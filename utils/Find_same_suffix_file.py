#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:40
# @File : Find_same_suffix_file.py
# @annotation:查找某目录下具有相同后缀名的文件
import os
def findAllFilesWithSpecifiedSuffix(target_dir, target_suffix="dcm"):
    find_res = []
    target_suffix_dot = "." + target_suffix
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == target_suffix_dot:
                find_res.append(os.path.join(root_path, file))
    return find_res

if __name__ == "__main__":
    a = findAllFilesWithSpecifiedSuffix("E:/DataCollection/NSCLC/NSCLC/volume", "gz")
