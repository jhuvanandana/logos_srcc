# Submissions must:
# - take as input a data directory (ie a path string) containing corresponding pairs of
#       - audio sound files (mp3 or wav audio)
#       - target word lists (unformatted text file)
# + and output a comma delimited (.csv) file indicating the presence or absense of words from the target list on their corresponding recordings
#       + these presences or absences should be indicated by 1s or 0s in a vector, one per line corresponding to each recording
#       + the vector must be preceded by a 'case' ID, which is the ID number of the recording/target list as labelled by their respective files
#       + the order of indicator variables (1s and 0s) must match the word order of the target list
#       + the file must have column headers as follows: "Id, R1, R2, ..., R15"

import os
import re
import sys
import glob
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from difflib import SequenceMatcher
from datetime import datetime

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

def main(data_dir='../input', solution_file='TeamJH.csv'):
    
    # append / as needed
    if data_dir[-1]!='/':
        data_dir+='/'
    
    print('This is {}, running at {}.\n'.format(sys.argv[0], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Input data
    print('Using input data directory: {}'.format(data_dir))
    audio_files = glob.glob1(data_dir,'*.mp3') + glob.glob1(data_dir,'*.wav')
    Nfiles = len(audio_files)

    print('..directory contains {} test audio files:'.format(Nfiles))
    print(os.listdir(data_dir))

    # Generate solution    
    nWords = 15
    r = sr.Recognizer()
    col_names = ['Id']+list(map(lambda k: 'R%d'%k,range(1,nWords+1)))

    outVec = []
    
    for sndFname in audio_files:
        print('Completing for: %s'%sndFname)

        # export to .wav if .mp3
        if '.mp3' in sndFname:
            sound = AudioSegment.from_mp3(sndFname)
            sndFname = sndFname.replace('.mp3','.wav')
            sound.export(sndFname, format='wav')
        
        txtFname = sndFname.replace('.wav','_targets.txt')
        
        file = open('%s%s'%(data_dir,txtFname),'r')
        fname, wordList = file.read().splitlines()
        fname = fname.replace('# ','')
        caseID = re.match(r'[rt]ID-(\d+)',fname).group(1)

        wordList = list(filter(lambda k: len(k), wordList.split(' ')))

        speechFile = sr.AudioFile(sndFname)
            
        with speechFile as source:
            audio = r.record(source)

        guessList = r.recognize_google(audio,language="en-AU")
        guessList = list(map(lambda k: k.lower(), guessList.split(' ')))
        
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

        outVec.append([caseID]+itemList)

    df = pd.DataFrame(outVec,columns=col_names)
    df.Id = df.Id.astype(int)
    df.sort_values('Id', inplace=True)
    
    print()
    print('Solution table:')
    print(df)

    print()
    print('Writing output solution file to: {}'.format(solution_file))
    df.to_csv(solution_file,index=False)

    print('Current directory contains:')
    print(os.listdir('./'))

if __name__ == '__main__': main()