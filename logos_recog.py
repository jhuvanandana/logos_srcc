# Submissions must:
# - take as input a data directory (ie a path string) containing corresponding pairs of
#       - audio sound files (mp3 or wav audio)
#       - target word lists (unformatted text file)
# + and output a comma delimited (.csv) file indicating the presence or absense of words from the target list on their corresponding recordings
#       + these presences or absences should be indicated by 1s or 0s in a vector, one per line corresponding to each recording
#       + the vector must be preceded by a 'case' ID, which is the ID number of the recording/target list as labelled by their respective files
#       + the order of indicator variables (1s and 0s) must match the word order of the target list
#       + the file must have column headers as follows: "Id, R1, R2, ..., R15"

import numpy as np
import os
import re
import sys
import glob
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from difflib import SequenceMatcher
from datetime import datetime

def mapToCV(word):
    vowelList = 'aeiouy'
    return ''.join(list(map(lambda k: 'v' if k in vowelList else 'c', word)))
    
# modification of https://www.howmanysyllables.com/howtocountsyllables
def countSyllables(word):
    nSyl = 0
    nLetters = len(word)
    vowelList = 'aeiouy'
    lastWasVowel = False
    if nLetters>3 and word[-2:]=='ed':
        cvMap = mapToCV(word)
        if cvMap[-4:]=='vcvc':
            ## redefine word
            word = word[:-1]
        elif cvMap[-4:]=='ccvc' and word[-4]==word[-3]:
            word = word[:-2]
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

def main(dataDir='../input', submissionFile='submission.csv'):
    
    # append team specific initials
    submissionFile = submissionFile.replace('.csv','_teamJH.csv')
    
    # append / as needed
    if dataDir[-1]!='/':
        dataDir+='/'
    
    print('This is {}, running at {}.\n'.format(sys.argv[0], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Input data
    print('Using input data directory: {}'.format(dataDir))
    audioFlist = glob.glob1(dataDir,'*.mp3') + glob.glob1(dataDir,'*.wav')
    Nfiles = len(audioFlist)

    print('..directory contains {} test audio files:'.format(Nfiles))
    print(os.listdir(dataDir))

    # Generate solution
    nWords = 15
    r = sr.Recognizer()
    colNames = ['Id']+list(map(lambda k: 'R%d'%k,range(1,nWords+1)))

    outVec = []
    print('\n\nBeginning Prediction')
    
    for sndFname in audioFlist:
        print('Completing for: %s'%sndFname)
        fullFname = '%s%s'%(dataDir,sndFname)
        iGenFile = False

        # export to .wav if .mp3
        if '.mp3' in sndFname:
            iGenFile = True
            sound = AudioSegment.from_mp3(fullFname)
            fullFname = fullFname.replace('.mp3','.wav')
            sound.export(fullFname, format='wav')
        
        txtFname = fullFname.replace('.wav','_targets.txt')
        
        file = open(txtFname,'r')
        fname, wordList = file.read().splitlines()
        fname = fname.replace('# ','')
        caseID = re.match(r'[rt]ID-(\d+)',fname).group(1)

        wordList = list(filter(lambda k: len(k), wordList.split(' ')))
        speechFile = sr.AudioFile(fullFname)

        with speechFile as source:
            audio = r.record(source)

        guessList = r.recognize_google(audio,language="en-AU")
        guessList = list(map(lambda k: k.lower(), guessList.split(' ')))
        
        ## remove generated file
        if iGenFile:
            os.remove(fullFname)
        
        itemList = [0]*nWords # prepare storage
        exclList = []
        for guess in guessList:
            idxList = list(filter(lambda k: wordList[k]==guess, range(nWords)))
            if len(idxList):
                itemList[idxList[0]]=1
            else:
                exclList.append(guess)

        filtIdx = list(filter(lambda k: not itemList[k], range(nWords)))
        filtWordList = list(map(lambda k: wordList[k], filtIdx))

        for guess in exclList:
            ratVec = []
            for word in filtWordList:
                s = SequenceMatcher(lambda x: x == ' ',word,guess)
                ratVec.append(s.ratio())
            i_mx = np.argmax(ratVec)
            mx_ratio = ratVec[i_mx]
            mx_word  = filtWordList[i_mx]
            same_syl = countSyllables(mx_word)==countSyllables(guess)
            same_first = mx_word[0]==guess[0]
            same_last = mx_word[-1]==guess[-1]
            iFound = False
            if mx_ratio>0.85:
                iFound = True
            elif mx_ratio>0.8:
                iFound = same_syl
            elif mx_ratio>0.6:
                iFound = same_first or same_last                    
            if iFound:
                idx = list(filter(lambda k: wordList[k]==word, range(nWords)))[0]
                itemList[idx]=1

        outVec.append([caseID]+itemList)

    df = pd.DataFrame(outVec,columns=colNames)
    df.Id = df.Id.astype(int)
    df.sort_values('Id', inplace=True)

    print()
    print('Solution table:')
    print(df)

    print()
    print('Writing output solution file to: {}'.format(submissionFile))
    df.to_csv(submissionFile,index=False)

    print('Current directory contains:')
    print(os.listdir('./'))

if __name__ == '__main__':

    argList = list(sys.argv)
    if len(argList)==3:
        pythonScript, dataDir, submissionFile = argList
        main(dataDir,submissionFile)
    else:
        print('Please input a data directory and path to submission file\nNumber of arguments: %d, %s'%(len(argList), str(argList)))