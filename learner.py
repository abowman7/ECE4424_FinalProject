import numpy as np
import matplotlib.pyplot as plt                          # pip install matplotlib
import sklearn
from sklearn.model_selection import train_test_split     # use to split up the data set
from sklearn import metrics #use to check accuracy

#classification models
from sklearn.linear_model import LogisticRegression      # do: pip install scikit-learn
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import svm
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


# Applying KNN Classification 
#knn using ball tree algorithm
bestBallAccuracy = 0
bestBallKValue = 0
BallAccuracy = []
kValues = np.linspace(1, 30, 30)

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
print("")

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
print("")

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
print("")

# KNN Plots
"""
# Ball tree plot
plt.plot(kValues, BallAccuracy)
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different K Values Using KNN Classification with Ball Tree Algorithm')
plt.show()

# Kd tree tree plot
plt.plot(kValues, KdAccuracy)
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different K Values Using KNN Classification with KD Tree Algorithm')
plt.show()

# Brute force search plot
plt.plot(kValues, BruteAccuracy)
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different K Values Using KNN Classification with Brute force search Algorithm')
plt.show()
"""

# Additional Knn test, with different random states for training split and k value of 3

"""
randomStateValues = []
ballAccuracies= []
kdAccuracies = []
bruteAccuracies = []
for i in range (15):
    randomStateValues.append(i)   
    # Splitting into training and test data
    X_train, X_test, y_train, y_test = train_test_split(inputData, occupancy, test_size=0.2, random_state=i)


    knn_ball = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')    
    knn_ball.fit(X_train, y_train)
    mean_accuracy = knn_ball.score(X_test, y_test)
    ballAccuracies.append(mean_accuracy)
    
    knn_kd = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')    
    knn_kd.fit(X_train, y_train)
    mean_accuracy = knn_kd.score(X_test, y_test)
    kdAccuracies.append(mean_accuracy)

    knn_brute = KNeighborsClassifier(n_neighbors=3, algorithm='brute')    
    knn_brute.fit(X_train, y_train)
    mean_accuracy = knn_brute.score(X_test, y_test)
    bruteAccuracies.append(mean_accuracy)
    
plt.plot(randomStateValues, ballAccuracies, label = 'Ball Tree')
plt.plot(randomStateValues, kdAccuracies, label = 'KD Tree')
plt.plot(randomStateValues, bruteAccuracies, label = 'Brute Force')
plt.xlabel('Random State values')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different KNN Classification Algorithms Based on Different Data Splits (k=3)')
plt.legend()
plt.show()
"""

#Logistic Regression
bestlogAccuracy = 0
bestCValue = 0
logAccuracies = []
CValues = []

for i in range (1, 11):
    currC = i/10
    CValues.append(currC)
    logReg = LogisticRegression(C=currC)    
    logReg.fit(X_train, y_train)
    logPredict = logReg.predict(X_test)
    #save accuracies into a list
    logRegAccuracies = [prediction == testVal for prediction, testVal in zip(logPredict, y_test)]
    #total accuracy
    logAcc = sum(logRegAccuracies) / len(logRegAccuracies)
    logAccuracies.append(logAcc)
    #print(mean_accuracy)

    if bestlogAccuracy < logAcc :
        bestlogAccuracy = logAcc
        bestCValue = currC
    
print("Using the Logistic Regression Model, the best C value was {}, with a mean accuracy of {}".format(bestCValue, bestlogAccuracy))
print("")

# Applying Support Vector Machines for Classification

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)
print("Using Support Vector Machines, an accuracy of {} was obtained".format(svm_accuracy))
print("")

#Random Forest
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfcPred = rfc.predict(X_test)

print("Using a Random Forest Classifier with 100 estimators, we got an accuracy of ", metrics.accuracy_score(y_test, rfcPred))
print("")


#Accuracy Bar Chart
avgKnn = (bestKdAccuracy + bestBruteAccuracy + bestBallAccuracy)/3
rfcAcc = metrics.accuracy_score(y_test, rfcPred)
plotAcc = [avgKnn, bestlogAccuracy, rfcAcc, svm_accuracy]
plotLabels = ['Avg KNN', 'Logistic Regression', 'Random Forest', 'SVC']

plt.bar(plotLabels, plotAcc)
plt.title("Comparing the Results of Different Classification Methods")
plt.xlabel("Classification Methods")
plt.ylabel("Accuracies")
plt.show()

# bagging
bagRFC = RandomForestClassifier(n_estimators=100).fit(X_train, y_train).predict(X_test)
bagLog = LogisticRegression(C=bestCValue).fit(X_train, y_train).predict(X_test)
bagKNN = KNeighborsClassifier(n_neighbors=bestBallKValue).fit(X_train, y_train).predict(X_test)
bagSVC = svm.SVC().fit(X_train, y_train).predict(X_test)

bagPreds = []
for i in range(len(bagRFC)):
    votes = [bagRFC[i], bagLog[i], bagKNN[i], bagKNN[i], bagSVC[i]]
    prob = sum(votes) / len(votes)
    bagPreds.append(prob > 0.5)

print("Using our Bagging implemented from scratch, we got an accuracy of ", metrics.accuracy_score(y_test, bagPreds))
print("")

#breaking data into day of week
mond = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 0])
tues = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 1])
wedn = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 2])
thur = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 3])
frid = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 4])
satu = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 5])
sund = np.array([[inputData[i][1], float(initTarget[i][0])] for i in range(len(inputData)) if inputData[i][2] == 6])

#plot all data
plt.scatter(mond[:,0], mond[:,1], color='red', label='Monday', marker='.', alpha=0.5)
plt.scatter(tues[:,0], tues[:,1], color='blue', label='Tuesday', marker='o', alpha=0.5)
plt.scatter(wedn[:,0], wedn[:,1], color='green', label='Wednesday', marker='+', alpha=0.5)
plt.scatter(thur[:,0], thur[:,1], color='orange', label='Thursday', marker='x', alpha=0.5)
plt.scatter(frid[:,0], frid[:,1], color='yellow', label='Friday', marker='^', alpha=0.5)
plt.scatter(satu[:,0], satu[:,1], color='purple', label='Saturday', marker='s', alpha=0.5)
plt.scatter(sund[:,0], sund[:,1], color='brown', label='Sunday', marker='*', alpha=0.5)
plt.axhline(y=300, color='black', label='1/2 Cap', alpha=0.3)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Occupancy")
plt.title("All Gathered McComas Data (Time vs. Occupancy)")
plt.show()

#predict user inputted values
predKNN = KNeighborsClassifier(n_neighbors=3).fit(inputData, occupancy)
#get user inputs
faren = float(input("Enter the current Temperature in Farenheit: "))
cHour = float(input("Enter the current Time in the format hr.(min/60): "))
weekday = float(input("Enter the current day of the week (0-Monday, 6-Sunday): "))
userValues = []
userValues.append([faren, cHour, weekday])
#predict
prediction = predKNN.predict(userValues)

#output prediction
if prediction[0] == 1:
    print("The gym will not be busy.")
else:
    print("The gym will be busy.")
