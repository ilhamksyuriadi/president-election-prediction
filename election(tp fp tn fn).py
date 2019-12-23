# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:12:23 2019

@author: ACER-E15
"""

import xlrd
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import math
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

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
        print(count, "")
    return data,label

def Preprocessing(data):#Preprocessing
    cleanData = []
    tokenizer = RegexpTokenizer(r'\w+')
    factory_stopwords = StopWordRemoverFactory()
    stopwords = factory_stopwords.get_stop_words()
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    count = 0
    for i in range(len(data)):
        lowerText = data[i].lower()#Case folding
        tokenizedText = tokenizer.tokenize(lowerText)#Punctual removal and tokenization
        swRemovedText = []#Stopwords removal
        for j in range(len(tokenizedText)):
            if tokenizedText[j] not in stopwords:
                swRemovedText.append(tokenizedText[j])
        stemmedText = []
        for k in range(len(swRemovedText)):#Stemming
            stemmedText.append(stemmer.stem(swRemovedText[k]))
        cleanData.append(stemmedText)
        count += 1
        print(count, "cleaned")
    return cleanData

#def ConstructHashtagFeature(data):
#    hashtag = []
#    tempData = []
#    for d in data:
#        tweet = d.split()
#        tempData.append(tweet)
#        for word in tweet:
#            if word[0] == '#':
#                if word not in hashtag:
#                    hashtag.append(word)
#    features = []
#    for d in tempData:
#        tempFeature = []
#        for h in hashtag:
#            if h in d:
#                tempFeature.append(1)
#            else:
#                tempFeature.append(0)
#        features.append(tempFeature)
#    return hashtag, features

def CreateUnigram(data):
    unigram = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in unigram:
                unigram.append(data[i][j])
                count += 1
                print(count, "unigram")
    return unigram

def CreateBigram(data):
    bigram = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            tempBigram = data[i][j] + " " + data[i][j+1]
            if tempBigram not in bigram:
                bigram.append(tempBigram)
                count += 1
                print(count, "bigram")
    return bigram

def CreateTrigram(data):
    trigram = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])-2):
            tempTrigram = data[i][j] + " " + data[i][j+1] + " " + data[i][j+2]
            if tempTrigram not in trigram:
                trigram.append(tempTrigram)
                count += 1
                print(count, "trigram")
    return trigram

def TransformToBigram(data):
    bigramData = []
    count = 0
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-1):
            temp = data[i][j] + " " + data[i][j+1]
            bigramPerData.append(temp)
        bigramData.append(bigramPerData)
        count += 1
        print(count, "Data's feature transformed to bigram")
    return bigramData

def TransformToTrigram(data):
    trigramData = []
    count = 0
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-2):
            temp = data[i][j] + " " + data[i][j+1] + " " + data[i][j+2]
            bigramPerData.append(temp)
        trigramData.append(bigramPerData)
        count += 1
        print(count, "Data's feature transformed to bigram")
    return trigramData

def CreateDF(term,doc):
    df = {}
    deletedDf = []
    count = 0
    for i in range(len(term)):
        for j in range(len(doc)):
            if term[i] in doc[j]:
                if term[i] in df:
                    df[term[i]] += 1
                else:
                    df[term[i]] = 1
        count += 1
        print(count, "df created")
    countTreshold = 0
    for i in term:
        if df[i] <= 0: 
            deletedDf.append(i)
            del df[i]
            countTreshold += 1
            print(countTreshold, "threshold applied")
    return df, deletedDf

def CreateTFIDF(data,df,term,deletedDf):
    tfidf = []
    count = 0
    for i in range(len(data)):
        tempTfidf = []
        for j in range(len(term)):
            if term[j] in data[i] and term[j] not in deletedDf:
                tf = 0
                for k in range(len(data[i])):
                    if data[i][k] == term[j]:
                        tf += 1
                idf = math.log10(len(data)/df[term[j]])
                tempTfidf.append(idf*tf)
            else:
                tempTfidf.append(0)
        count += 1
        print(count, "tf-idf created")
        tfidf.append(tempTfidf)
    return tfidf

def FeatureMerger(data1,data2,data3):
    mergedData = []
    if len(data3) == 0:
        for i in range(len(data1)):
            value = data1[i] + data2[i]
            mergedData.append(value)
            print(i+1,'Feature merged')
    else:
        for i in range(len(data1)):
            value = data1[i] + data2[i] + data3[i]
            mergedData.append(value)
            print(i+1,'Feature merged')
        
    return mergedData

#def AddHashtagFeature(data,hashtag):
#    features = []
#    for i in range(len(data)):
#        feature = data[i]+hashtag[i]
#        features.append(feature)
#    return features

def Accuracy(label,predict):
    tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    return acc

rawData,label = LoadDataset("april7rb_1000.xlsx")
#hashtag,hashtagFeature = ConstructHashtagFeature(rawData)
cleanData = Preprocessing(rawData)

def Percentage(label):
    jkw = 0
    pbw = 0
    for l in label:
        if l == 0:
            jkw += 1
        elif l == 1:
            pbw +=1
    return round((jkw/len(label)),4),round((pbw/len(label)),4)

unigram = CreateUnigram(cleanData)
bigram = CreateBigram(cleanData)
trigram = CreateTrigram(cleanData)

bigramData = TransformToBigram(cleanData)
trigramData = TransformToTrigram(cleanData)

uniDF,deletedUniDf = CreateDF(unigram,cleanData)
biDF,deletedBiDf = CreateDF(bigram,bigramData)
triDF,deletedTriDf = CreateDF(trigram,trigramData)

#withot hashtag
uniTFIDF = CreateTFIDF(cleanData,uniDF,unigram,deletedUniDf)
biTFIDF = CreateTFIDF(bigramData,biDF,bigram,deletedBiDf)
triTFIDF = CreateTFIDF(trigramData,triDF,trigram,deletedTriDf)
unibiTFIDF = FeatureMerger(uniTFIDF,biTFIDF,[])
unitriTFIDF = FeatureMerger(uniTFIDF,triTFIDF,[])
bitriTFIDF = FeatureMerger(biTFIDF,triTFIDF,[])
unibitriTFIDF = FeatureMerger(uniTFIDF,biTFIDF,triTFIDF)

#with hashtag
#uniTFIDF = AddHashtagFeature(uTFIDF,hashtagFeature)
#biTFIDF = AddHashtagFeature(bTFIDF,hashtagFeature)
#triTFIDF = AddHashtagFeature(tTFIDF,hashtagFeature)
#unibiTFIDF = AddHashtagFeature(ubTFIDF,hashtagFeature)
#unitriTFIDF = AddHashtagFeature(utTFIDF,hashtagFeature)
#bitriTFIDF = AddHashtagFeature(btTFIDF,hashtagFeature)
#unibitriTFIDF = AddHashtagFeature(ubtTFIDF,hashtagFeature)

clf = svm.SVC(kernel='linear')

predictU = cross_val_predict(clf,uniTFIDF,label,cv=10)
predictU = predictU.tolist()

predictB = cross_val_predict(clf,biTFIDF,label,cv=10)
predictB = predictB.tolist()

predictT = cross_val_predict(clf,triTFIDF,label,cv=10)
predictT = predictT.tolist()

predictUB = cross_val_predict(clf,unibiTFIDF,label,cv=10)
predictUB = predictUB.tolist()

predictUT = cross_val_predict(clf,unitriTFIDF,label,cv=10)
predictUT = predictUT.tolist()

predictBT = cross_val_predict(clf,bitriTFIDF,label,cv=10)
predictBT = predictBT.tolist()

predictUBT = cross_val_predict(clf,unibitriTFIDF,label,cv=10)
predictUBT = predictUBT.tolist()

print("unigram : ", Accuracy(label,predictU))
print("bigram : ", Accuracy(label,predictB))
print("trigram : ", Accuracy(label,predictT))
print("unigram-bigram : ", Accuracy(label,predictUB))
print("unigram-trigram : ", Accuracy(label,predictUT))
print("bigram-trigram : ",Accuracy(label,predictBT))
print("unigram-bigram-trigram : ",Accuracy(label,predictUBT))
print("persentase : ", Percentage(predictUB))


        
        






















