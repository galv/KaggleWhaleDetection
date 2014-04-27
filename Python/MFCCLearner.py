import csv
import os
import pdb
import numpy as np
import scipy.io.wavfile as wav
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from python_speech_features.features import mfcc
import cv2

DATA_DIR = os.getcwd() + '/../data/'

def main():
    clf = LogisticRegression()
    (X_train, y_train) = readTrainData(DATA_DIR + '/train_1000_v.csv',clf)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return (clf, X_train, y_train, y_pred)

""" 
Returns training data as a tuple (X,y) where X is  an n-by-m unraveled matrix where n is the number of samples, and y is 1-by-n.
"""
def readTrainData(trainFile, clf):
    i_f = open(trainFile)
    reader = csv.reader(i_f)
    soundList = list()
    isWhaleList = list()
    next(i_f)
    for line in reader:
        fileName = line[0]
        fileName = fileName[:line[0].find('.aiff')] + '.wav' 
        #Change .aiff ending to .wav
        (rate, sig) = wav.read(DATA_DIR + 'train/' + fileName)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.ndarray.flatten(mfcc_feat)
        #Unravel the matrix into a 1-D array. 
        soundList.append(mfcc_feat)
        isWhaleList.append(line[1])
    soundArr = np.array(soundList)
    isWhaleArr = np.array(isWhaleList)
    isWhaleArr = isWhaleArr.astype(np.bool)
    #Convert from string to boolean for convenience.
    return (soundArr, isWhaleArr)

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
    (clf, X_train, y_train, y_pred) = main()
