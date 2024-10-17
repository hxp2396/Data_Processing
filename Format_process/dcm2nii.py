#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 17:25
# @File : dcm2nii.py
# @annotation:
import pydicom
import dicom2nifti
def dicom2nii(in_path_dicom_nifti,out_path):
    dicom2nifti.dicom_series_to_nifti(in_path_dicom_nifti, out_path + '.nii.gz')

