import pandas as pd
import pickle
from datetime import datetime
from os import path


# ml libraries
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# define global variable for reproducible results
randomSeed = 11


# todo more preprocessing (ex. sklearn.preprocessing.MinMaxScaler())
# todo add more complex validation technique (ex. k fold cross validation)
# todo additional metrics (ex. f1; https://scikit-learn.org/stable/modules/model_evaluation.html)
# todo additional algorithms (ex. neural networks; https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
# todo additional tuning, check additional parameters (ex. max depth (tree), svm (gamma, C), ...)



def fitMain():

    print(datetime.now().time())

    # load data
    trainDataTfidf = pd.read_csv("../data/trainDataTfidf.csv")
    testDataTfidf = pd.read_csv("../data/testDataTfidf.csv")
    trainDataY = pd.read_csv("../data/trainDataY.csv")
    testDataY = pd.read_csv("../data/testDataY.csv")
    print(datetime.now().time())

    # prepare dict for saving accuracies
    accuracyAll = dict()

    # fit logistic regression
    accuracyAll["logReg"] = fitLogRegression(trainDataTfidf, trainDataY, testDataTfidf, testDataY)
    print(datetime.now().time())


    # fit tree
    accuracyAll["tree"] = fitTree(trainDataTfidf, trainDataY, testDataTfidf, testDataY)
    print(datetime.now().time())

    # fit random forest
    for n in range(2, 25):
        accuracyAll["rf" + str(n)] = fitRandomForest(trainDataTfidf, trainDataY, testDataTfidf, testDataY, n)
        print(datetime.now().time())

    # fit svm
    for kernel in ["rbf", "linear", "poly"]:
        accuracyAll["svm" + kernel.capitalize()] = fitSvm(trainDataTfidf, trainDataY, testDataTfidf, testDataY, kernel)
        print(datetime.now().time())


    # naive bayes
    accuracyAll["naiveBayes"] = fitNaiveBayes(trainDataTfidf, trainDataY, testDataTfidf, testDataY)
    print(datetime.now().time())


    print(1)

    return



#######################################################################################


def fitLogRegression(trainDataTfidf, trainDataY, testDataTfidf, testDataY):

    # define path to saved model
    filePath = "../models/logRegressionModel-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            lrModel = pickle.load(f)

    else: # fit
        clf = LogisticRegression()
        lrModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(lrModel, f)

    # calculate accuracy
    lrAccuracy = accuracy_score(testDataY, lrModel.predict(testDataTfidf), normalize=True) * 100
    print("lr: " + str(lrAccuracy))

    return lrAccuracy



def fitTree(trainDataTfidf, trainDataY, testDataTfidf, testDataY):

    # define path to saved model
    filePath = "../models/treeModel-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            treeModel = pickle.load(f)

    else: # fit
        clf = tree.DecisionTreeClassifier(random_state=randomSeed)
        treeModel = clf.fit(trainDataTfidf, trainDataY)
        with open(filePath, "wb") as f:
            pickle.dump(treeModel, f)


    # calculate accuracy
    treeAccuracy = accuracy_score(testDataY, treeModel.predict(testDataTfidf), normalize=True) * 100
    print("tree: " + str(treeAccuracy))

    return treeAccuracy




def fitRandomForest(trainDataTfidf, trainDataY, testDataTfidf, testDataY, numberOfTrees):

    # define path to saved model
    filePath = "../models/rfModel" + str(numberOfTrees) + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            rfModel = pickle.load(f)

    else: # fit
        clf = RandomForestClassifier(n_estimators=numberOfTrees, random_state=11)
        rfModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(rfModel, f)


    # calculate accuracy
    rfAccuracy = accuracy_score(testDataY, rfModel.predict(testDataTfidf), normalize=True) * 100
    print("rf (" + str(numberOfTrees) + "): " + str(rfAccuracy))

    return rfAccuracy




def fitSvm(trainDataTfidf, trainDataY, testDataTfidf, testDataY, kernel):

    # define path to saved model
    filePath = "../models/svmModel" + kernel.capitalize() + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            svmModel = pickle.load(f)


    else: # fit
        clf = svm.SVC(kernel=kernel, random_state=randomSeed)  # linear kernel for starter
        svmModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(svmModel, f)


    # calculate accuracy
    svmAccuracy = accuracy_score(testDataY, svmModel.predict(testDataTfidf), normalize=True) * 100
    print("svm (" + kernel + "): " + str(svmAccuracy))

    return svmAccuracy





def fitNaiveBayes(trainDataTfidf, trainDataY, testDataTfidf, testDataY):

    # define path to saved model
    filePath = "../models/naiveBayes" + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            nbModel = pickle.load(f)


    else: # fit
        clf = GaussianNB()  # linear kernel for starter
        nbModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(nbModel, f)


    # calculate accuracy
    nbAccuracy = accuracy_score(testDataY, nbModel.predict(testDataTfidf), normalize=True) * 100
    print("nb: " + str(nbAccuracy))

    return nbAccuracy



fitMain()
