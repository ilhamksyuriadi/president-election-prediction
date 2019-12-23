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

def LoadUnformalWord(FileLoc):#Load corpus for unformal word
    unformal = []
    formal = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    count = 0
    for i in range(0,sheet.nrows):
        if sheet.cell_value(i,1) != '':
            unformal.append(sheet.cell_value(i,0))
            formal.append(sheet.cell_value(i,1))
            count += 1
            print(count, "unformal word inserted")
    return unformal,formal

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
        print(count, "data cleaned")
    return cleanData

def UnformalToFormal(data,unformal,formal):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in unformal:
                data[i][j] = formal[unformal.index(data[i][j])]

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

def CreateUnigram(data):
    unigram = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in unigram:
                unigram.append(data[i][j])
                count += 1
                print(count, "unigram created")
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
                print(count, "bigram created")
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
                print(count, "trigram created")
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
            print(countTreshold, "treshold applied")
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

def AddHashtagFeature(data,hashtag):
    features = []
    for i in range(len(data)):
        feature = data[i]+hashtag[i]
        features.append(feature)
    return features

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
cleanData = Preprocessing(rawData)
unformal,formal = LoadUnformalWord('Unformal Word.xlsx')
UnformalToFormal(cleanData,unformal,formal)

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
#resultUni = cross_val_score(clf,uniTFIDF,label,cv=10)
#resultBi = cross_val_score(clf,biTFIDF,label,cv=10)
#resultTri = cross_val_score(clf,triTFIDF,label,cv=10)
#resultUnibi = cross_val_score(clf,unibiTFIDF,label,cv=10)
#resultUnitri = cross_val_score(clf,unitriTFIDF,label,cv=10)
#resultBitri = cross_val_score(clf,bitriTFIDF,label,cv=10)
#resultUnibitri = cross_val_score(clf,unibitriTFIDF,label,cv=10)
#
#print("Unigram :", sum(resultUni)/len(resultUni))
#print("Bigram :", sum(resultBi)/len(resultBi))
#print("Trigram :", sum(resultTri)/len(resultTri))
#print("Unigram Bigram :", sum(resultUnibi)/len(resultUnibi))
#print("Unigram Trigram :", sum(resultUnitri)/len(resultUnitri))
#print("Bigram Trigram :", sum(resultBitri)/len(resultBitri))
#print("Unigram Bigram Trigram:", sum(resultUnibitri)/len(resultUnibitri))

jkw,pbw = Percentage(label)

predictU = cross_val_predict(clf,uniTFIDF,label,cv=10)
predictU = predictU.tolist()
jkwU,pbwU = Percentage(predictU)

predictB = cross_val_predict(clf,biTFIDF,label,cv=10)
predictB = predictB.tolist()
jkwB,pbwB = Percentage(predictB)

predictT = cross_val_predict(clf,triTFIDF,label,cv=10)
predictT = predictT.tolist()
jkwT,pbwT = Percentage(predictT)

predictUB = cross_val_predict(clf,unibiTFIDF,label,cv=10)
predictUB = predictUB.tolist()
jkwUB,pbwUB = Percentage(predictUB)

predictUT = cross_val_predict(clf,unitriTFIDF,label,cv=10)
predictUT = predictUT.tolist()
jkwUT,pbwUT = Percentage(predictUT)

predictBT = cross_val_predict(clf,bitriTFIDF,label,cv=10)
predictBT = predictBT.tolist()
jkwBT,pbwBT = Percentage(predictBT)

predictUBT = cross_val_predict(clf,unibitriTFIDF,label,cv=10)
predictUBT = predictUBT.tolist()
jkwUBT,pbwUBT = Percentage(predictUBT)

print("Actual percentage:")
print("Jokowi:",jkw,"Prabowo:",pbw)

print("Unigram")
print(Accuracy(label,predictU))
print("Jokowi:",jkwU,"Prabowo:",pbwU)

print("Bigram")
print(Accuracy(label,predictB))
print("Jokowi:",jkwB,"Prabowo:",pbwB)

print("Trigram")
print(Accuracy(label,predictT))
print("Jokowi:",jkwT,"Prabowo:",pbwT)

print("Unigram Bigram")
print(Accuracy(label,predictUB))
print("Jokowi:",jkwUB,"Prabowo:",pbwUB)

print("Unigram Trigram")
print(Accuracy(label,predictUT))
print("Jokowi:",jkwUT,"Prabowo:",pbwUT)

print("Bigram Trigram")
print(Accuracy(label,predictBT))
print("Jokowi:",jkwBT,"Prabowo:",pbwBT)

print("Unigram Bigram Trigram")
print(Accuracy(label,predictUBT))
print("Jokowi:",jkwUBT,"Prabowo:",pbwUBT)

compareActual = compareU = Compare(hashtag,hashtagFeature,label)
compareU = Compare(hashtag,hashtagFeature,predictU)
compareB = Compare(hashtag,hashtagFeature,predictB)
compareT = Compare(hashtag,hashtagFeature,predictT)
compareUB = Compare(hashtag,hashtagFeature,predictUB)
compareUT = Compare(hashtag,hashtagFeature,predictUT)
compareBT = Compare(hashtag,hashtagFeature,predictBT)
compareUBT = Compare(hashtag,hashtagFeature,predictUBT)

        
        






















