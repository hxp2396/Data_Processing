#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 16:15
# @File : test_read_dicom_meta.py
# @annotation:
from utils.read_dicom_meta import *
def test_read_dicom_meta():
    dcmpath="i0012,0000b.dcm"
    output_json_path='dcm.json'
    Generate_MetaData_Json(dcmpath,output_json_path)
if __name__ == '__main__':
    test_read_dicom_meta()