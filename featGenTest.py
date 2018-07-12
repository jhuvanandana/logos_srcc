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
from difflib import SequenceMatcher
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression

def unbracket(vector):
    lst  = [val for sublist in vector for val in sublist]
    return np.array(lst)

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

if __name__ == '__main__':
    
    ## initialise
    start = datetime.now()
    os.chdir('C:/Users/Jacqueline/Documents/DataScience/Projects/logos')
    
    df = pd.read_csv('output/features_sim_train.csv')
    df['true_syl'] = df.target.map(lambda k: countSyllables(k))
    df['pred_syl'] = df.mx_word.map(lambda k: countSyllables(k))
    df['same_syl'] = (df.true_syl==df.pred_syl).astype(np.int64)
    
    ## first letter, last letter
    df['same_first'] = (df.mx_word.map(lambda k: k[0])==df.target.map(lambda k: k[0])).astype(np.int64)
    df['same_last'] = (df.mx_word.map(lambda k: k[-1])==df.target.map(lambda k: k[-1])).astype(np.int64)
    
    y_true = df.y_true
    y_pred = df.y_pred
    
    print('Initial Accuracy: %0.3f'%(accuracy_score(y_true,y_pred)))
    
    ## extract cases where y_pred = 0, same #syllables
    subDf = df.loc[df.y_pred==0]
    
    selFeatNames = ['same_syl','same_first','same_last','mx_ratio']
    featDf = subDf.loc[:,selFeatNames]
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=3)
    clf.fit(featDf,subDf.y_true)
    sub_pred = clf.predict(featDf)
    
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz

    with open('output/decision_tree.txt','w') as f:
        f = export_graphviz(clf, out_file=f, feature_names=selFeatNames)
    
    ## logistic regression
    lr = LogisticRegression()
    feats = np.reshape(np.array(subDf.mx_ratio),(-1,1))
    lr.fit(feats, subDf.y_true)
    
    probs = lr.predict_proba(feats)
    fpr,tpr,thresh = roc_curve(subDf.y_true, feats)
    
    auc_score = auc(fpr,tpr)
    
    accVec = []
    for thresh in np.linspace(0,1,101):
        tmp_pred = subDf.mx_ratio.map(lambda k: int(k>=thresh))
        accScore = accuracy_score(subDf.y_true, tmp_pred)
        print('%0.2f Threshold, Accuracy: %0.3f'%(thresh,accScore))
        accVec.append(accScore)
        
    print('Maximum accuracy: %0.3f'%(max(accVec)))
    
print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')