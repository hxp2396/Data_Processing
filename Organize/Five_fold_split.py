#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/30 12:24
# @File : Five_fold_split.py
# @annotation:
import os
import shutil
import random
def create_subsets(src_dir, dest_dir, num_subsets=5):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 文件夹名称
    folders = ['ADC', 'ADC_ss', 'LABEL']

    # 收集所有文件路径
    all_files = {folder: [] for folder in folders}

    for folder in folders:
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            all_files[folder] = [os.path.join(folder_path, f) for f in files]
    print(all_files)

    # 确保每个文件夹中的文件数量相同
    num_files = len(all_files['ADC'])
    if not all(len(all_files[folder]) == num_files for folder in folders):
        raise ValueError("All folders must contain the same number of files.")

    # 创建文件索引并打乱顺序
    file_indices = list(range(num_files))
    random.shuffle(file_indices)

    # 计算每个子集的大小
    subset_size = num_files // num_subsets
    print(subset_size)
    # 创建子集并复制文件
    for i in range(num_subsets):
        subset_dir = os.path.join(dest_dir, f'folder{i + 1}')
        os.makedirs(subset_dir, exist_ok=True)

        # 确定当前子集的文件索引
        start_index = i * subset_size
        end_index = start_index + subset_size if i < num_subsets - 1 else num_files
        subset_indices = file_indices[start_index:end_index]
        print(subset_indices)

        # 复制文件到当前子集目录
        for index in subset_indices:
            for folder in folders:
                print(all_files[folder][index])
                print(os.path.join(subset_dir, folder))
                os.makedirs(os.path.join(subset_dir, folder), exist_ok=True)
                shutil.copy(all_files[folder][index], os.path.join(subset_dir, folder))


# 示例用法
src_directory = '../dataset/BONBID2023'  # 源目录路径
dest_directory = '../dataset/predata'  # 目标目录路径
create_subsets(src_directory, dest_directory)
