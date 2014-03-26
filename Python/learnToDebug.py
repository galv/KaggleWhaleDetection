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

    xStuff1 = (100, 110, 120, 130)
    xStuff2 = (1000, 1100, 1200, 1300)
    xStuff = np.array([xStuff1, xStuff2])
    yStuff = (0, 1)
    clf.fit(xStuff, yStuff)
    
    pdb.set_trace()
    return clf

    #train(DATA_DIR + '/train_1000_v.csv',clf)
    #test(DATA_DIR +'/test_1000_v.csv',clf)
    #test(DATA_DIR + '/train_1000_v.csv',clf)
    #pdb.set_trace()

if __name__ == "__main__":
    clf = main()
    
