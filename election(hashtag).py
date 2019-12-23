# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:12:23 2019

@author: HP
"""

import xlrd
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import math
from sklearn import svm
from sklearn.model_selection import cross_val_predict

def LoadDataset(FileLoc):#load dataset
    data = []
    label = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    count = 0
    for i in range(0,sheet.nrows):
        data.append(sheet.cell_value(i,0))
        label.append(int(sheet.cell_value(i,1)))
        count += 1
        print(count, "data inserted")
    return data,label

def ConstructHashtagFeature(data):
    hashtag = []
    tempData = []
    for d in data:
        d = str(d)
        tweet = d.split()
        tempData.append(tweet)
        for word in tweet:
            if word[0] == '#':
                if word not in hashtag:
                    hashtag.append(word)
    features = []
    for d in tempData:
        tempFeature = []
        for h in hashtag:
            if h in d:
                tempFeature.append(1)
            else:
                tempFeature.append(0)
        features.append(tempFeature)
    return hashtag, features

def Accuracy(label,predict):
    acc = 0
    for i in range(len(label)):
        if label[i] == predict[i]:
            acc += 1
    return acc/len(label)

def Compare(hashtag,feature,label):
    comparison = []
    for i in range(len(feature)):
        tempComparison = []
        for j in range(len(feature[i])):
            if feature[i][j] == 1:
                if label[i] == 1:
                    tempComparison.append([hashtag[j],1])
                else:
                    tempComparison.append([hashtag[j],0])
        comparison.append(tempComparison)
    c = []
    for i in range(len(hashtag)):
        jkw = 0
        pbw = 0
        for j in range(len(comparison)):
            tempJkw = 0
            tempPbw = 0
            for k in range(len(comparison[j])):
                if hashtag[i] == comparison[j][k][0]:
                    if comparison[j][k][1] == 0:
                        tempJkw += 1
                    elif comparison[j][k][1] == 1:
                        tempPbw += 1
            jkw += tempJkw
            pbw += tempPbw
        c.append([hashtag[i],round(jkw/(jkw+pbw),4),round(pbw/(jkw+pbw),4)])
    return c

def Percentage(label):
    jkw = 0
    pbw = 0
    for l in label:
        if l == 0:
            jkw += 1
        elif l == 1:
            pbw +=1
    return round((jkw/len(label)),4),round((pbw/len(label)),4)

rawData,label = LoadDataset("DATA FIX(50).xlsx")
hashtag,hashtagFeature = ConstructHashtagFeature(rawData)

clf = svm.SVC(kernel='linear')

jkw,pbw = Percentage(label)

predict = cross_val_predict(clf,hashtagFeature,label,cv=10)
predict = predict.tolist()
jkwH,pbwH = Percentage(predict)

print("Actual percentage:")
print("Jokowi:",jkw,"Prabowo:",pbw)

print(Accuracy(label,predict))
print("Jokowi:",jkwH,"Prabowo:",pbwH)

compareActual = compareU = Compare(hashtag,hashtagFeature,label)
compare = Compare(hashtag,hashtagFeature,predict)

        
        






















