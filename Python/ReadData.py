import csv
import os
import pdb
import numpy as np
import scipy.io.wavfile as wav
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from python_speech_features.features import mfcc
from python_speech_features.features import logfbank

DATA_DIR = os.getcwd() + '/../data/'

def main():
    clf = LogisticRegression()
    (X_train, y_train) = readTrainData(DATA_DIR + '/train_1000_v.csv',clf)
    clf.fit(X_train, y_train)
    #print clf.predict(X_train)

    return (clf, X_train, y_train)
    #test(DATA_DIR +'/test_1000_v.csv',clf)
    #pdb.set_trace()

    #test(DATA_DIR + '/train_1000_v.csv',clf)

    #pdb.set_trace()

def readTrainData(trainFile, clf):
    i_f = open(trainFile)
    reader = csv.reader(i_f)
    soundList = list()
    isWhaleList = list()
    next(i_f)
    for line in reader:
        fileName = line[0]
        fileName = fileName[:line[0].find('.aiff')] + '.wav' #Change .aiff ending to .wav
        (rate, sig) = wav.read(DATA_DIR + 'train/' + fileName)
        mfcc_feat = mfcc(sig,rate)
        soundList.append(mfcc_feat)
        isWhaleList.append(line[1])
    soundArr = np.array(soundList)
    isWhaleArr = np.array(isWhaleList)#, dtype = type(np.int16))
    return (soundArr, isWhaleArr)
    #clf.fit(soundArr, isWhaleArr)

    #pdb.set_trace()

def test(testFile, clf):
    i_f = open(testFile)
    reader = csv.reader(i_f)
    soundList = list()
    isWhaleList = list()
    next(i_f)
    for line in reader:
        fileName = line[0]
        fileName = fileName[:line[0].find('.aiff')] + '.wav' #Change .aiff ending to .wav
        (rate, sig) = wav.read(DATA_DIR + 'train/' + fileName)
        mfcc_feat = mfcc(sig,rate)
        soundList.append(mfcc_feat)
        isWhaleList.append(line[1])
    soundArr = np.array(soundList)
    isWhaleArr = np.array(isWhaleList)#, dtype = type(np.int16))
    
    #predictedWhaleArr = clf.predict(soundArr)
    
    #accuracy = (predictedWhaleArr == isWhaleArr).all()
    #print accuracy

    return 
    
if __name__ == "__main__":
    (clf, X, y) = main()
