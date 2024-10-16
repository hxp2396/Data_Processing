#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:53
# @File : write_csv.py
# @annotation:
import csv
def writecsv(file_name,csv_head=["姓名","年龄","性别"],data_item=["l",'18','男']):
    # 1. 创建文件对象
    f = open(file_name,'a',encoding='utf-8-sig')
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    csv_writer.writerow(csv_head)
    # 4. 写入csv文件内容
    csv_writer.writerow(data_item)
    # 5. 关闭文件
    f.close()

if __name__=="__main__":
    writecsv("test.csv")
