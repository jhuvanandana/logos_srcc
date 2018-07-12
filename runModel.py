#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Kaggle, Home Credit project
Author: Jacqueline Huvanandana
Created: 20/06/2018
"""
import os
import numpy as np
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from difflib import SequenceMatcher

from datetime import datetime
from sklearn.metrics import accuracy_score

def unbracket(vector):
    lst  = [val for sublist in vector for val in sublist]
    return np.array(lst)

def mapToCV(word):
    vowelList = 'aeiouy'
    return list(map(lambda k: 'v' if k in vowelList else 'c', word))
    
# https://www.howmanysyllables.com/howtocountsyllables
def countSyllables(word):
    nSyl = 0
    nLetters = len(word)
    vowelList = 'aeiouy'
    lastWasVowel = False
    
    cvMap = mapToCV(word)
    if nLetters>3 and word[-2:]=='ed' and cvMap[-4:]=='vcvc':
        ## redefine word
        nLetters = len(word)
    
    for letter in word:
        foundVowel = False
        if letter in vowelList and not lastWasVowel:
            nSyl+=1
            foundVowel = lastWasVowel = True
        if not foundVowel:
            lastWasVowel = False
    if nLetters > 2 and word[-2:] == 'es':
        nSyl-=1
    elif nLetters > 1 and word[-1:] == 'e':
        nSyl-=1
    return nSyl

if __name__ == '__main__':
    
    ## initialise
    start = datetime.now()
    os.chdir('C:/Users/Jacqueline/Documents/DataScience/Projects/logos')
    
    ## training data
    trainDir = 'data/simulated/train'
    trainFlist    = os.listdir(trainDir)
    trainTargList = list(filter(lambda k: 'targets.txt' in k, trainFlist))
    
    ## testing data
    testDir  = 'data/simulated/test'
    testFlist = os.listdir(testDir)
    testTargList = list(filter(lambda k: 'targets.txt' in k, testFlist))
    
    ## logos training data
    logTrainDir = 'data/LOGOS_exemplar/train'
    logFlist    = os.listdir(logTrainDir)
    logTargList = list(filter(lambda k: 'targets.txt' in k, logFlist))
    
    ## collate sounds
    i_do_sim_train = 1
    i_do_sim_pred  = 0
    i_do_log_train = 0
    nWords = 15
    r = sr.Recognizer()
    
    i_run = False
    if i_do_sim_train:
        i_run = True
        fnameList = trainTargList
        fnameDir  = trainDir        
    elif i_do_sim_pred:
        i_run = True
        fnameList = trainTargList
        fnameDir  = trainDir        
    elif i_do_log_train:
        i_run = True
    
    if i_run:
        outVec = []
        predList = []
        targList = []
        for txtFname in trainTargList:
            print('Completing for: %s'%txtFname)
            file = open('%s/%s'%(trainDir,txtFname),'r')
            fname, wordList = file.read().splitlines()
            fname = fname.replace('# ','')
            rID = int(fname.split('-')[1].split('_')[0])

            gtfile = open('%s/%s_ground_truth.txt'%(trainDir,fname))
            gtList = gtfile.read()
            gtList = list(filter(lambda k: len(k), list(gtList.split(' '))))
    
            wordList = list(filter(lambda k: len(k), wordList.split(' ')))
                
            wavFname = '%s/%s.wav'%(trainDir,fname)
            speechFile = sr.AudioFile(wavFname)
            with speechFile as source:
                audio = r.record(source)
            guessList = r.recognize_google(audio,language="en-AU")
            guessList = list(map(lambda k: k.lower(), guessList.split(' ')))
            
            itemList = [0]*15 # prepare storage
            exclList = []
            for guess in guessList:
                idxList = list(filter(lambda k: wordList[k]==guess, range(nWords)))
                if len(idxList):
                    itemList[idxList[0]]=1
                else:
                    exclList.append(guess)

            filtIdx = list(filter(lambda k: not itemList[k], range(nWords)))
            filtWordList = list(map(lambda k: wordList[k], filtIdx))
                    
            if len(exclList):
                ratMat = []
                for word in filtWordList:
                    nSyl = countSyllables(word)
                    ratVec = []
                    for guess in exclList:
                        s = SequenceMatcher(lambda x: x == ' ',word,guess)
                        sameFirst = word[0]==guess[0]
                        sameLast = word[-1]==guess[-1]
                        if nSyl==countSyllables(guess) and (sameFirst or sameLast):
                            rat = s.ratio()
                        else:
                            rat = 0
                        ratVec.append(rat)
                    ratMat.append(ratVec)
                ratMat = np.array(ratMat).T # flip on its side
    
                mxVec = list(map(lambda k: np.argmax(k) if max(k)>0.5 else np.nan, ratMat))
                for i_arg in mxVec:
                    if not np.isnan(i_arg):
                        word = filtWordList[i_arg]
                        idx = list(filter(lambda k: wordList[k]==word, range(nWords)))[0]
                        itemList[idx]=1

            target = list(map(lambda k: int(k in gtList), wordList))
            predList.append(itemList)
            targList.append(target)
            outVec.append([rID]+itemList)
        
        ## what is the accuracy score?
        y_true = unbracket(targList)
        y_pred = unbracket(predList)

        print('Accuracy score from Logos Training: %0.3f'%accuracy_score(y_true,y_pred))

        outDf = pd.DataFrame(outVec,columns=['Id']+list(map(lambda k: 'R%d'%k,range(1,16))))
        outDf.to_csv('output/submission.csv',index=False)
    
print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')