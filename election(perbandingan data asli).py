# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:16 2019

@author: Asus
"""

import xlrd

def LoadDataset(FileLoc):#load dataset
    label = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    count = 0
    for i in range(0,sheet.nrows):
        label.append(int(sheet.cell_value(i,1)))
        count += 1
        print(count, "data inserted")
    return label

label = LoadDataset('april7rb_acak.xlsx')

jkw = 0
pbw = 0
count = 0
for l in label:
    if l == 0:
        jkw += 1
    else:
        pbw += 1
    count += 1
    print(count)
        
print('Jokowi:',jkw / len(label))
print('Parbowo:',pbw / len(label))