import numpy as np
import csv
#import pandas as pd    # not using pandas at the moment
import matplotlib.pyplot as plt     # pip install matplotlib
import sklearn
from sklearn.model_selection import train_test_split    # use to split up the data set
#classification models
from sklearn.linear_model import LogisticRegression   # do: pip install scikit-learn
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


dataPath = "mccomasData.csv"    #data file path
targetPath = "mccomasTarget.csv"    #target file path
#read data from csvs
initData = np.loadtxt(dataPath, delimiter=",", dtype=str, encoding="utf-8-sig")  
initTarget = np.loadtxt(targetPath, delimiter=",", dtype=str, encoding="utf-8-sig")

# save temps as floats (string -> float)
floatTemp = []
for temp in initData:
    floatTemp.append(float(temp[0]))

# save times as floats (string -> float)
floatTime = []
for time in initData:
    breaked = time[2].split(':')
    hour = float(breaked[0])
    minutes = float(breaked[1])
    # time saved as hour.(minute/60)
    floatTime.append((hour + (minutes / 60.0)))

# save day of week as float (string -> float) 0 is Monday, 6 is Sunday
floatDay = []
for day in initData:
    floatDay.append(float(day[3]))

# make occupancy a bool of < half full capacity (650)
threshold = 650.0 / 2.0 # half of full capacity
occupancy = []
for ppl in initTarget:
    occupancy.append((float(ppl[0]) < threshold))   #turn to bool (0/1) for classification

#put input data into a single array/list
inputData = []
for i in range(len(floatTime)):
    inputData.append([floatTemp[i], floatTime[i], floatDay[i]])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#gnb = GaussianNB()
#y_pred = gnb.fit(X_train, y_train).predict(X_test)