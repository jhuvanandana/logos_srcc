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

# https://www.howmanysyllables.com/howtocountsyllables
def countSyllables(word):
    nSyl = 0
    nLetters = len(word)
    vowelList = 'aeiouy'
    lastWasVowel = False
    
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
    i_do_sim_pred = 0
    i_do_log_train = 1
    nWords = 15
    r = sr.Recognizer()
    
    if i_do_sim_train:
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
                
            
            
    if i_do_log_train:
        allTarg = []
        allPred = []
        for txtFname in logTargList:
            print('Completing for: %s'%txtFname)
            file = open('%s/%s'%(logTrainDir,txtFname),'r')
            fname, wordList = file.read().splitlines()
            fname = fname.replace('# ','')
            rID = int(fname.split('-')[1].split('_')[0])
                                  
            gtfile = open('%s/%s_ground_truth.txt'%(logTrainDir,fname))
            gtList = gtfile.read()
            gtList = list(filter(lambda k: len(k), list(gtList.split(' '))))
    
            wordList = list(filter(lambda k: len(k), wordList.split(' ')))
                
            sndFname = '%s/%s.mp3'%(logTrainDir,fname)
            if os.path.isfile(sndFname): # handle missing files

                sound = AudioSegment.from_mp3(sndFname)
                wavFname = sndFname.replace('.mp3','.wav')
                sound.export(wavFname, format='wav')
                speechFile = sr.AudioFile(wavFname)
                
                with speechFile as source:
                    audio = r.record(source)
                    
                os.remove(wavFname)
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
                        
                targList = list(map(lambda k: int(k in gtList), wordList))
                allPred.append([rID]+itemList)
                allTarg.append([rID]+targList)
        ## what is the accuracy score?
        y_pred = unbracket(list(map(lambda k: k[1:], allPred)))
        y_true = unbracket(list(map(lambda k: k[1:], allTarg)))

        print('Accuracy score from Logos Training: %0.3f'%accuracy_score(y_true,y_pred))
    
    if i_do_sim_pred:
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