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
import tensorflow as tf
import speech_recognition as sr
from io import BytesIO
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher
from scipy.io import wavfile
from scipy.signal import resample, coherence, spectrogram
from matplotlib import pyplot as plt
from pygame import mixer

from datetime import datetime
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import ensemble

from keras import backend as K
from keras import optimizers
from keras import callbacks
from keras import layers
from keras import models
from pydub import AudioSegment

## https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word
def countSyllables(word):
    vowels = "aeiouy"
    numVowels = 0
    lastWasVowel = False
    for wc in word:
        foundVowel = False
        for v in vowels:
            if v == wc:
                if not lastWasVowel: numVowels+=1   #don't count diphthongs
                foundVowel = lastWasVowel = True
                break
        if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
            lastWasVowel = False
    if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
        numVowels-=1
    elif len(word) > 1 and word[-1:] == "e":    #remove silent e
        numVowels-=1
    return numVowels

if __name__ == '__main__':
    
    ## initialise
    start = datetime.now()
    os.chdir('C:/Users/Jacqueline/Documents/DataScience/Projects/logos')
    trainDir = 'data/simulated/train'
    testDir  = 'data/simulated/test'
    
    ## training data
    trainFlist    = os.listdir(trainDir)
    trainTargList = list(filter(lambda k: 'targets.txt' in k, trainFlist))
    
    ## testing data
    testFlist = os.listdir(testDir)
    testTargList = list(filter(lambda k: 'targets.txt' in k, testFlist))
    
    ## collate sounds
    i_do_train = 0
    i_do_pred = 1
    nWords = 15
    r = sr.Recognizer()
    
    if i_do_train:
        for txtFname in trainTargList:
            print('Completing for: %s'%txtFname)
            file = open('%s/%s'%(trainDir,txtFname),'r')
            fname, wordList = file.read().splitlines()
            fname = fname.replace('# ','')
                                  
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
                    
            for word in filtWordList:
                nSyl = countSyllables(word)
                ratVec = []
                for guess in exclList:
                    s = SequenceMatcher(lambda x: x == ' ',word,guess)
                    if nSyl==countSyllables(guess):
                        rat = s.ratio()
                    else:
                        rat = 0
                    ratVec.append(rat)
                if max(ratVec)>0.6: # try threshold
                    idx = list(filter(lambda k: wordList[k]==word, range(nWords)))[0]
                    itemList[idx]=1
#
#            spectList = []
#            for word in filtWordList:
#                soundFname = 'library/%s.wav'%word
#                if not os.path.isfile(soundFname):
#                    tts = gTTS(text=word, lang='en')
#                    tts.save('temp.mp3')    
#                    sound = AudioSegment.from_mp3('temp.mp3')
#                    sound.export(soundFname, format='wav')
#                
#                targ_sampr, targData = wavfile.read(soundFname,'r')
#                f, t, spect = np.fft(targData, fs=targ_sampr)
#                spectList.append(spect)
#                
#            for word in exclList:
#                tts = gTTS(text=word, lang='en')
#                tts.save('temp.mp3')
#    
#                sound = AudioSegment.from_mp3('temp.mp3')
#                sound.export('library/%s.wav'%word, format='wav')
#                samp_rate, guessData = wavfile.read('library/%s.wav'%word,'r')
#                
#                # mean squared coherence
#                for sound in soundList:
#                    coherence(guessData,sound,fs=samp_rate)
                
            
    
    if i_do_pred:
        outVec = []
        
        for txtFname in testTargList:
            print('Completing for: %s'%txtFname)
            file = open('%s/%s'%(testDir,txtFname),'r')
            fname, wordList = file.read().splitlines()
            fname = fname.replace('# ','')
            rID = int(fname.split('-')[1].split('_')[0])
    
            wordList = list(filter(lambda k: len(k), wordList.split(' ')))
                
            wavFname = '%s/%s.wav'%(testDir,fname)
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
                for word in filtWordList:
                    nSyl = countSyllables(word)
                    ratVec = []
                    for guess in exclList:
                        s = SequenceMatcher(lambda x: x == ' ',word,guess)
                        if nSyl==countSyllables(guess):
                            rat = s.ratio()
                        else:
                            rat = 0
                        ratVec.append(rat)
                    if max(ratVec)>0.6: # try threshold
                        idx = list(filter(lambda k: wordList[k]==word, range(nWords)))[0]
                        itemList[idx]=1                
            outVec.append([rID]+itemList)
            
        outDf = pd.DataFrame(outVec,columns=['Id']+list(map(lambda k: 'R%d'%k,range(1,16))))
        outDf.to_csv('output/submission-seqmatch.csv',index=False)
    
print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')