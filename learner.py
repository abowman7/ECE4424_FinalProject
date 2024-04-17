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
from sklearn.ensemble import RandomForestClassifier

dataPath = "mccomasData.csv"    #data file path
targetPath = "mccomasTarget.csv"    #target file path
#read data from csvs
initData = np.loadtxt(dataPath, delimiter=",", dtype=str, encoding="utf-8-sig")  
initTarget = np.loadtxt(targetPath, delimiter=",", dtype=str, encoding="utf-8-sig")

#-----------------Getting data ready for processing----------------

# save temps as floats (string -> float)
floatTemp = []
for temp in initData:
    floatTemp.append(float(temp[0]))
#print(floatTemp)

# save times as floats (string -> float)
floatTime = []
for time in initData:
    breaked = time[2].split(':')
    hour = float(breaked[0])
    minutes = float(breaked[1])
    # time saved as hour.(minute/60)
    floatTime.append(hour + round((minutes / 60.0), 2))
#print(floatTime)

# save day of week as float (string -> float) 0 is Monday, 6 is Sunday
floatDay = []
for day in initData:
    floatDay.append(float(day[3]))
#print(floatDay)

# make occupancy a bool of < half full capacity (600)
threshold = 600.0 / 2.0 # half of full capacity
occupancy = []
for ppl in initTarget:
    occupancy.append((float(ppl[0]) < threshold))   #turn to bool (0/1) for classification
#print(occupancy)

#put input data into a single array/list
inputData = []
for i in range(len(floatTime)):
    inputData.append([floatTemp[i], floatTime[i], floatDay[i]])
#print(inputData)

# Splitting into training and test data
X_train, X_test, y_train, y_test = train_test_split(inputData, occupancy, test_size=0.2, random_state=0)
#print(X_train)

#gnb = GaussianNB()
#y_pred = gnb.fit(X_train, y_train).predict(X_test)


# Applying KNN Classification 

#knn using ball tree algorithm
bestBallAccuracy = 0
bestBallKValue = 0
BallAccuracy = []

for i in range (1, 31):
    knn_ball = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree')    
    knn_ball.fit(X_train, y_train)
    mean_accuracy = knn_ball.score(X_test, y_test)
    BallAccuracy.append(mean_accuracy)
    #print(mean_accuracy)

    if bestBallAccuracy < mean_accuracy :
        bestBallAccuracy = mean_accuracy
        bestBallKValue = i
        
print("Using the ball tree algorithm with knn, the best k value was {}, with a mean accuracy of {}".format(bestBallKValue, bestBallAccuracy))


#knn using kd tree algorithm
bestKdAccuracy = 0
bestKdKValue = 0
KdAccuracy = []

for i in range (1, 31):
    knn_kd = KNeighborsClassifier(n_neighbors=i, algorithm='kd_tree')    
    knn_kd.fit(X_train, y_train)
    mean_accuracy = knn_kd.score(X_test, y_test)
    KdAccuracy.append(mean_accuracy)
    #print(mean_accuracy)

    if bestKdAccuracy < mean_accuracy :
        bestKdAccuracy = mean_accuracy
        bestKdKValue = i
        
print("Using the kd tree algorithm with knn, the best k value was {}, with a mean accuracy of {}".format(bestKdKValue, bestKdAccuracy))


#knn using brute force search
bestBruteAccuracy = 0
bestBruteKValue = 0
BruteAccuracy = []

for i in range (1, 31):
    knn_brute = KNeighborsClassifier(n_neighbors=i, algorithm='brute')    
    knn_brute.fit(X_train, y_train)
    mean_accuracy = knn_brute.score(X_test, y_test)
    BruteAccuracy.append(mean_accuracy)
    #print(mean_accuracy)

    if bestBruteAccuracy < mean_accuracy :
        bestBruteAccuracy = mean_accuracy
        bestBruteKValue = i
        

print("Using the brute-force search algorithm with knn, the best k value was {}, with a mean accuracy of {}".format(bestBruteKValue, bestBruteAccuracy))


#Logistic Regression
logReg = LogisticRegression()
#fit data to model
logReg.fit(X_train, y_train)
#predict test data
logPredict = logReg.predict(X_test)
#save accuracies into a list
logRegAccuracies = [prediction == testVal for prediction, testVal in zip(logPredict, y_test)]
#total accuracy
logAcc = sum(logRegAccuracies) / len(logRegAccuracies)
#print
print("Using the Logistic Regression model, the mean accuracy was: ", logAcc)



# Applying Support Vector Machines for Classification

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)
print("Using Support Vector Machines, an accuracy of {} was obtained".format(svm_accuracy))