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
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import split_on_silence

from scipy.io import wavfile
from scipy.signal import resample
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
    
    ## (test) collate sounds
    outVec = []
    r = sr.Recognizer()
    for txtFname in testTargList:
        print('Completing for: %s'%txtFname)
        file = open('%s/%s'%(testDir,txtFname),'r')
        fname, wordList = file.read().splitlines()
        fname = fname.replace('# ','')
                              
#        gtfile = open('%s/%s_ground_truth.txt'%(trainDir,fname))
#        gtList = gtfile.read()

        wordList = list(filter(lambda k: len(k), wordList.split(' ')))
        
        wavFname = '%s/%s.wav'%(testDir,fname)
        samp_sampr, sampData = wavfile.read(wavFname,'r')

        speechFile = sr.AudioFile(wavFname)
        with speechFile as source:
            audio = r.record(source)
        guessList = r.recognize_google(audio)
        guessList = list(map(lambda k: k.lower(), guessList.split(' ')))
        
        rID = int(fname.split('-')[1].split('_')[0])
        
        itemList = [rID]
        for word in wordList:
            item = 0
            if word in guessList:
                item = 1
            itemList.append(item)
            
        outVec.append(itemList)
        
    outDf = pd.DataFrame(outVec,columns=['Id']+list(map(lambda k: 'R%d'%k,range(1,16))))
    outDf.to_csv('output/submission.csv',index=False)
            

#        for word in wordList:
#            tts = gTTS(text=word, lang='en')
#            tts.save('temp.mp3')
#
#            sound = AudioSegment.from_mp3('temp.mp3')
#            sound.export('temp.wav', format='wav')
#            
#            targ_sampr, targData = wavfile.read('temp.wav','r')
#            resampData = resample(targData,int(len(targData)*(targ_sampr/samp_sampr)))
    
print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')