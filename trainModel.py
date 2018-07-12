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
    i_do_sim_train = 0
    i_do_log_train = 1
    nWords = 15
    r = sr.Recognizer()
    
    i_run = False
    if i_do_sim_train:
        i_run = True
        fnameList = trainTargList
        fnameDir  = trainDir        
    elif i_do_log_train:
        i_run = True
        fnameList = logTargList
        fnameDir  = logTrainDir
    
    if i_run:
        outVec = []
        predList = []
        targList = []
        featList = []
        for txtFname in fnameList:
            fullTxtName = '%s/%s'%(fnameDir,txtFname)
            sndFileExists = os.path.exists(fullTxtName.replace('_targets.txt','.mp3')) or os.path.exists(fullTxtName.replace('_targets.txt','.wav'))
            if sndFileExists:
                print('Completing for: %s'%txtFname)
                file = open(fullTxtName,'r')
                fname, wordList = file.read().splitlines()
                fname = fname.replace('# ','')
                rID = int(fname.split('-')[1].split('_')[0])
    
                gtfile = open('%s/%s_ground_truth.txt'%(fnameDir,fname))
                gtList = gtfile.read()
                gtList = list(filter(lambda k: len(k), list(gtList.split(' '))))
        
                wordList = list(filter(lambda k: len(k), wordList.split(' ')))
                    
                wavFname = '%s/%s.wav'%(fnameDir,fname)
                iGenFile = False
                if not os.path.exists(wavFname):
                    iGenFile = True
                    fullFname = '%s/%s.mp3'%(fnameDir,fname)
                    sound = AudioSegment.from_mp3(fullFname)
                    fullFname = fullFname.replace('.mp3','.wav')
                    sound.export(fullFname, format='wav')
                
                speechFile = sr.AudioFile(wavFname)
                with speechFile as source:
                    audio = r.record(source)
                guessList = r.recognize_google(audio,language="en-AU")
                guessList = list(map(lambda k: k.lower(), guessList.split(' ')))

                if iGenFile:
                    os.remove(wavFname)
                
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

                target = list(map(lambda k: int(k in gtList), wordList))
                for targ in wordList:
                    sVec = []
                    for guess in guessList:
                        s = SequenceMatcher(lambda x: x== ' ', targ, guess)
                        sVec.append(s.ratio())
                    arg = np.argmax(sVec)
                    strList = str(guessList)[1:-1]
                    strList = strList.replace(' ','')
                    strList = strList.replace('\'','')
                    strList = strList.replace('[','')
                    strList = strList.replace(']','')
                    row = [rID
                           ,targ
                           ,int(targ in gtList)
                           ,int(targ in guessList)
                           ,guessList[arg]
                           ,max(sVec)
                           ,strList
                           ]
                    outVec.append(row)
        
        outCol = ['ID','target','y_true','y_pred','mx_word','mx_ratio','guess_list']

        outDf = pd.DataFrame(outVec,columns=outCol)
        outDf.to_csv('output/features_log_train.csv',index=False)
    
print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')