import pandas as pd
import pickle
from datetime import datetime
from os import path

# ml libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


# define global variable randomSeed for reproducible results
randomSeed = 11


def fitMain(trainDataTfidf, trainDataY):

    # 3-fold cross validation to omit overfitting
    cvResults = fitWithCrossValidation(trainDataTfidf, trainDataY, K=3, fromFile=True)

    # average cross validation results to get the best model
    avgCvResults = averageCVresults(cvResults)

    # choose the best model according to accuracy and f1 measure
    bestAlgorithms = chooseFinalModels(avgCvResults)

    # fit the best model on full train data
    finalModels = fitBestModels(bestAlgorithms, trainDataTfidf, trainDataY)

    # return final model for future prediction
    return finalModels



#######################################################################################


def fitBestModels(bestAlgorithms, trainDataTfidf, trainDataY):

    # test
    bestAlgorithms.append(tree)

    # define dictionary {algName: function to fit this alg}
    algToFunction = {"logReg": fitLogRegression, "tree": fitTree, "naiveBayes": fitNaiveBayes}
    algWithParamToFunction = {"knn": fitKnn, "rf": fitRandomForest, "svm": fitSvm}

    # if algorithm defined with parameter
    finalModels = []
    for bestAlg in bestAlgorithms:
        if "-" in bestAlg:
            tmp = bestAlg.split("-")
            algorithm, parameter = tmp[0], tmp[1]

            # fit with appropriate function (wtihout test data)
            finalModel = algWithParamToFunction[algorithm](trainDataTfidf, trainDataY, None , None, parameter, finalFit = True)
            finalModels.append(finalModel)

        else:
            finalModel = algToFunction[bestAlg](trainDataTfidf, trainDataY, None , None, finalFit = True)


    print(finalModel)
    return finalModels



# choose models with higher accuracy/f1
def chooseFinalModels(avgCvResults):

    # convert tuple of lists to pandas df (and transpose to get algorithms in rows)
    avgResultsTable = pd.DataFrame.from_dict(avgCvResults).transpose()

    # name columns
    avgResultsTable.rename(columns={0: "accuracy", 1: "f1"}, inplace=True)

    # get 5 best by both measures
    bestByAccuracy = list(avgResultsTable.nlargest(5, ['accuracy']).index.values)
    bestByF1 = list(avgResultsTable.nlargest(5, ['f1']).index.values)

    # get best in both
    bestAlgorithms = list(set(bestByAccuracy) & set(bestByF1))


    print(bestAlgorithms)


    return bestAlgorithms





# get average results for each algorithm setting
def averageCVresults(cvResults):

    cvAccuracy, cvF1 = cvResults[0], cvResults[1]
    avgAccuracy, avgF1 = {}, {}

    # for each algorithm
    for alg in cvAccuracy:

        algAccuracy, algF1 = cvAccuracy[alg], cvF1[alg]

        # if multiple parameters i.e. value is a dict
        if isinstance(algAccuracy, dict):

            # for each used parameter in one algorithm
            for parameter in algAccuracy:

                algParAccuracy, algParF1 =  algAccuracy[parameter], algF1[parameter]

                # get average
                avgAccuracy[alg + "-" + str(parameter)] = sum(algParAccuracy)/len(algParAccuracy)
                avgF1[alg + "-" + str(parameter)] = sum(algParF1)/len(algParF1)

        else: # if no parameters i.e. value is a list

            # get average
            avgAccuracy[alg], avgF1[alg] = sum(algAccuracy)/len(algAccuracy), sum(algF1)/len(algF1)

    return avgAccuracy, avgF1




# k-fold cross validation of all the chosen algorithms
def fitWithCrossValidation(trainDataTfidf, trainDataY, K, fromFile):


    # define filepath for saved results (for later checking)
    filePath = "./data/validationResults"

    if fromFile:
        with open(filePath, "rb") as f:
             accuracyAll, f1All = pickle.load(f)


    else: # calculate

        # use 3-fold cross validation
        kf = KFold(n_splits=K)

        # prepare dict for saving all results
        # todo: find a better way to initialize dicts for specific parameters (perhaps defaultDict?)
        accuracyAll = dict(logReg=[], tree=[], naiveBayes=[],
                           knn={2:[],4:[],6:[],8:[],10:[],12:[],14:[]},
                           rf={20:[], 21:[], 22:[], 23:[], 24:[]},
                           svm={"rbf":[], "sigmoid":[]})

        f1All = dict(logReg=[], tree=[], naiveBayes=[],
                     knn={2: [], 4: [], 6: [], 8: [], 10: [], 12: [], 14: []},
                     rf={20:[], 21:[], 22:[], 23:[], 24:[]},
                     svm={"rbf":[], "sigmoid":[]})


        # for each fold fit all the models
        for trainIndex, validateIndex in kf.split(trainDataTfidf):

            # define sets for each fold
            trainDataX_k, validateDataX_k = trainDataTfidf.iloc[trainIndex, :], trainDataTfidf.iloc[validateIndex, :]
            trainDataY_k, validateDataY_k = trainDataY.iloc[trainIndex, :], trainDataY.iloc[validateIndex, :]



            # fit knn
            for k in range(2,16,2):
                knnResultMetrics = fitKnn(trainDataX_k, trainDataY_k, validateDataX_k,
                                          validateDataY_k, k, finalFit=False)

                accuracyAll["knn"][k].append(knnResultMetrics[0])
                f1All["knn"][k].append(knnResultMetrics[1])





            # fit logistic regression
            lrResultMetrics = fitLogRegression(trainDataX_k, trainDataY_k, validateDataX_k,
                                               validateDataY_k, finalFit=False)

            accuracyAll["logReg"].append(lrResultMetrics[0])
            f1All["logReg"].append(lrResultMetrics[1])




            # fit decition trees
            treeResultMetrics = fitTree(trainDataX_k, trainDataY_k, validateDataX_k,
                                        validateDataY_k, finalFit=False)

            accuracyAll["tree"].append(treeResultMetrics[0])
            f1All["tree"].append(treeResultMetrics[1])





            # fit random forest
            for n in range(20, 25):
                rfResultMetrics = fitRandomForest(trainDataX_k, trainDataY_k, validateDataX_k,
                                                  validateDataY_k, n, finalFit=False)

                accuracyAll["rf"][n].append(rfResultMetrics[0])
                f1All["rf"][n].append(rfResultMetrics[1])




            # fit svm
            for kernel in ["rbf", "sigmoid"]:  # "linear", "poly" very low accuracy
                svmResultMetrics = fitSvm(trainDataX_k, trainDataY_k, validateDataX_k,
                                          validateDataY_k, kernel, finalFit=False)


                accuracyAll["svm"][kernel].append(svmResultMetrics[0])
                f1All["svm"][kernel].append(svmResultMetrics[1])




            # naive bayes
            nbResultMetrics = fitNaiveBayes(trainDataX_k, trainDataY_k, validateDataX_k,
                                            validateDataY_k, finalFit=False)
            accuracyAll["naiveBayes"].append(nbResultMetrics[0])
            f1All["naiveBayes"].append(nbResultMetrics[1])



            print(accuracyAll)
            print(f1All)

        # save accuracy and f1 values to file
        with open(filePath, "wb") as f:
            pickle.dump([accuracyAll, f1All], f)


    return accuracyAll, f1All




### Fit functions
# todo: optimize the structure so there is no duplicated code or seperate into two functions ?!?

def fitKnn(trainDataTfidf, trainDataY, testDataTfidf, testDataY, k, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = KNeighborsClassifier(n_neighbors=k)
        knnModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        predictedY = knnModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("knn: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1


    # define path to saved model
    filePath = "./models/knnModel-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            knnModel = pickle.load(f)
            return knnModel

    else: # fit
        clf = LogisticRegression()
        knnModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(knnModel, f)
            return knnModel

    # calculate accuracy
    # predictedY = knnModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("knn: " + str(accuracy))
    #
    #return accuracy, f1


def fitLogRegression(trainDataTfidf, trainDataY, testDataTfidf, testDataY, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = LogisticRegression()
        lrModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        predictedY = lrModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("lr: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1


    # define path to saved model
    filePath = "./models/logRegressionModel-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            lrModel = pickle.load(f)
            return lrModel

    else: # fit
        clf = LogisticRegression()
        lrModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(lrModel, f)
            return lrModel

    # calculate accuracy
    # predictedY = lrModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("lr: " + str(accuracy))
    #
    # return accuracy, f1


def fitTree(trainDataTfidf, trainDataY, testDataTfidf, testDataY, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = tree.DecisionTreeClassifier(random_state=randomSeed)
        treeModel = clf.fit(trainDataTfidf, trainDataY)
        predictedY = treeModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("tree: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1


    # define path to saved model
    filePath = "../models/treeModel-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            treeModel = pickle.load(f)
            return treeModel

    else: # fit
        clf = tree.DecisionTreeClassifier(random_state=randomSeed)
        treeModel = clf.fit(trainDataTfidf, trainDataY)
        with open(filePath, "wb") as f:
            pickle.dump(treeModel, f)
            return treeModel


    # calculate accuracy
    # predictedY = treeModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("tree: " + str(accuracy))
    #
    # return accuracy, f1



def fitRandomForest(trainDataTfidf, trainDataY, testDataTfidf, testDataY, numberOfTrees, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = RandomForestClassifier(n_estimators=numberOfTrees, random_state=11)
        rfModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        predictedY = rfModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("rf: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1

    # define path to saved model
    filePath = "../models/rfModel" + str(numberOfTrees) + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            rfModel = pickle.load(f)
            return rfModel

    else: # fit
        clf = RandomForestClassifier(n_estimators=numberOfTrees, random_state=11)
        rfModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(rfModel, f)
            return rfModel


    # calculate accuracy
    # predictedY = rfModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("rf (" + str(numberOfTrees) + "): " + str(accuracy))

    return accuracy, f1




def fitSvm(trainDataTfidf, trainDataY, testDataTfidf, testDataY, kernel, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = svm.SVC(kernel=kernel, random_state=randomSeed)  # linear kernel for starter
        svmModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        predictedY = svmModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("svm: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1


    # define path to saved model
    filePath = "../models/svmModel" + kernel.capitalize() + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            svmModel = pickle.load(f)
            return fitSvm()


    else: # fit
        clf = svm.SVC(kernel=kernel, random_state=randomSeed)  # linear kernel for starter
        svmModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(svmModel, f)
            return svmModel


    # calculate accuracy
    # predictedY = svmModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, svmModel.predict(testDataTfidf), normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("svm (" + kernel + "): " + str(accuracy))
    #
    # return accuracy, f1





def fitNaiveBayes(trainDataTfidf, trainDataY, testDataTfidf, testDataY, finalFit):

    # for cross validation - without saving
    if not finalFit:
        clf = GaussianNB()  # linear kernel for starter
        nbModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        predictedY = nbModel.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        print("nb: " + str(accuracy) + "  " + str(f1))
        return accuracy, f1

    # define path to saved model
    filePath = "../models/naiveBayes" + "-" + str(randomSeed)

    if path.exists(filePath):
        with open(filePath, "rb") as f:
            nbModel = pickle.load(f)
            return nbModel


    else: # fit
        clf = GaussianNB()  # linear kernel for starter
        nbModel = clf.fit(trainDataTfidf, trainDataY.values.ravel())
        with open(filePath, "wb") as f:
            pickle.dump(nbModel, f)
            return nbModel


    # calculate accuracy
    # predictedY = nbModel.predict(testDataTfidf)
    # accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
    # f1 = f1_score(testDataY, predictedY, average="macro")
    # print("nb: " + str(accuracy))
    #
    # return accuracy, f1





# todo: more preprocessing (for ex. sklearn.preprocessing.MinMaxScaler())
# todo: check additional metrics (for ex. ROC)
# todo: additional algorithms (ex. neural networks; https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
# todo: additional tuning, check additional parameters (for ex. max depth (tree), svm (gamma, C), ...)
