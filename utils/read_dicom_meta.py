#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:35
# @File : read_dicom_meta.py
# @annotation:读取dicom文件的头信息
import pydicom
import os
import json
def get_files():
    for filepath,dirnames,filenames in os.walk(r'D:地址'):
        # print(len(filenames))
        for dirname in dirnames:
            print(os.path.join(filepath,dirname))
            filename2=os.path.join(filepath,dirname)
            #print(filename2)
            qutag(filename2)
def qutag(filename2):
    print(filename2)
    # dcm_file_List = list(filter(lambda x: x.find('xls') <= 0, filename2))  # 拿出文件中不含xsl的dcm文件
    file_list = os.listdir(filename2)
    dcm_file_List = list(filter(lambda x: x.find('xls') <= 0, file_list))
    print(dcm_file_List)
    file = dcm_file_List[1]
    print(file)
    fileN = os.path.join(filename2, file)
    print(fileN)
    dcm_file = pydicom.read_file(fileN)
    PatientID = dcm_file.PatientID
    filename = filename2 + '.txt'
    data = open(filename, 'w', encoding="utf-8")
    print(dcm_file, file=data)  # 输出txt文档
    json_file = PatientID + '.json'
    print(json.dumps(dcm_file.to_json_dict(), indent=1), file=open(json_file, 'w'))  # 输出json文档
def main():
    get_files()

if __name__ == '__main__':
    main()

