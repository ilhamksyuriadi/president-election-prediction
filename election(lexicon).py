# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:24:17 2018

@author: ilhamksyuriadi
"""

import xlrd
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tag import CRFTagger
from googletrans import Translator
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import math
from sklearn.naive_bayes import MultinomialNB

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

def CreateBow(data):
    token = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            #if data[i][j] not in token:
            token.append(data[i][j])
            count += 1
            print(count, "token created")
    return token

def UnformalToFormal(data,unformal,formal):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in unformal:
                data[i][j] = formal[unformal.index(data[i][j])]
                
def Postagging(data):
    postaggedData = []
    postagOnly = []
    ct = CRFTagger()
    ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    postaggedData = ct.tag_sents(data)
    for i in range(len(postaggedData)):
        for j in range(len(postaggedData[i])):    
            postagOnly.append(postaggedData[i][j][1])
    return postagOnly

def RemoveDuplicate(bow,postag):
    uniqBow = []
    uniqPostag = []
    for i in range(len(bow)):
        if bow[i] not in uniqBow:
            uniqBow.append(bow[i])
            uniqPostag.append(postag[i])
    return uniqBow,uniqPostag

def LexiconScore(bow,postag):
    lexicalPostag = ['NN','NNP','NND','MD','JJ','RB']
    lexiconScore = {}
    translator = Translator()
    allWord = set(wn.all_lemma_names())
    for i in range(len(bow)):
        if bow[i] not in lexiconScore:
            englishWord = translator.translate(bow[i], dest='en')
            word = englishWord.text
            score = 0
            if word in allWord:
                if postag[i] in lexicalPostag:
                    if (postag[i] == 'NN' or postag[i] == 'NNP'or postag[i] == 'NND'):
                        allsyn = list(swn.senti_synsets(word, 'n'))
                        if len(allsyn) != 0:
                            lexicon = swn.senti_synset(word+'.n.01')
                            score = lexicon.pos_score() - lexicon.neg_score()
                    elif postag[i] == 'VB' or postag[i] == 'MD':
                        allsyn = list(swn.senti_synsets(word, 'v'))
                        if len(allsyn) != 0:
                            lexicon = swn.senti_synset(word+'.v.01')
                            score = lexicon.pos_score() - lexicon.neg_score()
                    elif postag[i] == 'JJ':
                        allsyn = list(swn.senti_synsets(word, 'a'))
                        if len(allsyn) != 0:
                            lexicon = swn.senti_synset(word+'.a.01')
                            score = lexicon.pos_score() - lexicon.neg_score()
                    elif postag[i] == 'RB':
                        allsyn = list(swn.senti_synsets(word, 'r'))
                        if len(allsyn) != 0:
                            lexicon = swn.senti_synset(word+'.r.01')
                            score = lexicon.pos_score() - lexicon.neg_score()
                lexiconScore[bow[i]] = score
            else:
                lexiconScore[bow[i]] = 0
    return lexiconScore
        
def LexiconMatrix(data,bow,lexicon):
    matrixX = []
    for i in range(len(data)):
        matrixY = []
        for j in range(len(bow)):
            if bow[j] in lexicon:
                matrixY.append(lexicon[bow[j]])
            else:
                matrixY.append(0)
        matrixX.append(matrixY)
    return matrixX

#def CreateDF(data,doc):
#    df = {}
#    count = 0
#    for i in range(len(data)):
#        for j in range(len(doc)):
#            if data[i] in doc[j]:
#                if data[i] in df:
#                    df[data[i]] += 1
#                else:
#                    df[data[i]] = 1
#        count += 1
#        print(count, "df created")
#    return df

#def CreateTFIDF(data,df,token):
#    tfidf = []
#    count = 0
#    for i in range(len(data)):
#        tempTfidf = []
#        for j in range(len(token)):
#            if token[j] in data[i]:
#                tf = 0
#                for k in range(len(data[i])):
#                    if data[i][k] == token[j]:
#                        tf += 1
#                idf = math.log10(len(data)/df[token[j]])
#                tempTfidf.append(idf*tf)
#            else:
#                tempTfidf.append(0)
#        count += 1
#        print(count, "tf-idf created")
#        tfidf.append(tempTfidf)
#    return tfidf


#token = CreateToken(cleanData)
#df = CreateDF(token,cleanData)
#tfidfData = CreateTFIDF(cleanData,df,token)
#trainData,testData = tfidfData[0:40],tfidfData[40:48]
#trainLabel,testLabel = label[0:40],label[40:48]
#clf = MultinomialNB()
#clf.fit(trainData,trainLabel)
#print(clf.score(testData,testLabel))

rawData,label = LoadDataset("DATA FIX(50).xlsx")
cleanData = Preprocessing(rawData)
unformal,formal = LoadUnformalWord('Unformal Word1.xlsx')
UnformalToFormal(cleanData,unformal,formal)
bow = CreateBow(cleanData)
postag = Postagging(cleanData)
uniqBow,uniqPostag = RemoveDuplicate(bow,postag)
#lexiconScore = LexiconScore(uniqBow,uniqPostag)
#matrixLexicon = LexiconMatrix(cleanData,uniqBow,lexiconScore)
#
#dataTrain = matrixLexicon[0:40]
#dataTest = matrixLexicon[40:50]
#labelTrain = label[0:40]
#labelTest = label[40:50]
#
#from sklearn import svm
#clf = svm.SVC()
#clf.fit(dataTrain,labelTrain)
#print(clf.score(dataTest,labelTest))




















