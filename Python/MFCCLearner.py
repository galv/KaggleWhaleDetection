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

"""
Training a logistic regression classifier and validates it.
"""
def main():
    clf = LogisticRegression()
    (X_train, y_train, X_val, y_val) = readData(DATA_DIR + '/train.csv',clf)
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    
    successes = np.isclose(y_val, y_val_pred)

    return (clf, X_train, y_train, X_val, y_val, y_val_pred)

""" 
Returns training data as a tuple (X,y) where X is  an n-by-m unraveled matrix where n is the number of samples, and y is 1-by-n.
"""
def readData(trainFile, clf):
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

    
    #randomize
    shuffleInUnison(soundArr, isWhaleArr)
    
    #Split train and validation sets 50-50.
    trainSoundArr = soundArr[0:len(soundArr)/2]
    valSoundArr = soundArr[len(soundArr)/2: len(soundArr)]

    trainIsWhaleArr = isWhaleArr[0: len(isWhaleArr)/2]
    valIsWhaleArr = isWhaleArr[len(isWhaleArr)/2: len(isWhaleArr)]

    return (trainSoundArr, trainIsWhaleArr, valSoundArr, valIsWhaleArr)

def shuffleInUnison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return (a[p], b[p])
    
if __name__ == "__main__":
    (clf, X_train, y_train, X_val, y_val, y_val_pred) = main()
