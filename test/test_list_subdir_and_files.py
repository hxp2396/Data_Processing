#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 16:49
# @File : test_list_subdir_and_files.py
# @annotation:
import os
from utils.list_subdir_and_files import *

def test_list_subdir_and_files():
    start_dir='D:\code\Data_Processing'
    fiel,dir=list_subdir_and_files(start_dir)
    print(dir)
    print(file)
if __name__ == '__main__':
    test_list_subdir_and_files()