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
    
    train(DATA_DIR + '/train_1000_v.csv',clf)
    test(DATA_DIR +'/test_1000_v.csv',clf)
    '''
    i_f = open( '../data/train_1000.csv', 'r' ) #Make file smaller.
    reader = csv.reader(i_f)
    next(i_f) #Skip header
    #Load files into hash table.
    call_dict = {} #fileName.wav : isWhaleCall
    for line in reader:
        label = line[0]
        isWhale = line[1]
        call_dict[label] = isWhale;
    '''
def train(trainFile, clf):
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
    clf.fit(soundArr, isWhaleArr)
    pdb.set_trace()

def test(testFile, clf):
    i_f = open(testFile)
    #clf.score(X,y)
    
if __name__ == "__main__":
    main()