from src import textCleanup, parser, tfidf, fit
from sklearn.model_selection import train_test_split
import collections
import pandas as pd



def main():

    # # parse/import parsed articles
    # parsedData = parser.getParsedData(fromFile=True)
    #
    # # split parsed data to train and test set (train: 3190 x 2, test: 798 x 2)
    # trainData, testData = train_test_split(parsedData, test_size=0.2, random_state=11)
    #
    # # write to file train & test data
    # trainData.to_csv("./data/parsedDataTrain.csv", index=False)
    # testData.to_csv("./data/parsedDataTest.csv", index=False)
    #
    # # import from file
    # trainData = pd.read_csv("./data/parsedDataTrain.csv")
    # testData = pd.read_csv("./data/parsedDataTest.csv")
    #
    # # apply function for cleaning article content and splitting it into terms
    # trainData["articleTerms"] = trainData["text"].apply(textCleanup.getRelevantTerms)
    # testData["articleTerms"] = testData["text"].apply(textCleanup.getRelevantTerms)
    #
    # # save variable we want to predict
    # trainDataY, testDataY = trainData["class"], testData["class"]
    # trainDataY.to_csv("./data/trainDataY.csv", index=False)
    # testDataY.to_csv("./data/testDataY.csv", index=False)
    #
    # # get final terms used as features
    # finalTerms = getTermFeatures(trainData)
    #
    # # get tf-idf table from training data i.e. document terms
    # trainDataTfidf = tfidf.getTfIdfTable(trainData, listOfRelevantTerms=finalTerms)
    # testDataTfidf = tfidf.getTfIdfTable(testData, listOfRelevantTerms=finalTerms)
    #
    # # save - testing
    # trainDataTfidf.to_csv("./data/trainDataTfidf.csv", index=False)
    # testDataTfidf.to_csv("./data/testDataTfidf.csv", index=False)

    # load train data
    trainDataTfidf = pd.read_csv("./data/trainDataTfidf.csv")
    trainDataY = pd.read_csv("./data/trainDataY.csv")

    # fit models on train data
    fit.fitMain(trainDataTfidf, trainDataY)

    # load test data
    # testDataTfidf = pd.read_csv("./data/testDataTfidf.csv")
    # testDataY = pd.read_csv("./data/testDataY.csv")

    # predict
    # todo: predict on the test set
    # finalResults = predictMain(testDataTfidf, testDataY)



##############################################################################


# todo: find a smarter way to choose final features
# function to get final terms, i.e. features out of the training data
def getTermFeatures(trainData):

    # TESTING: class distribution
    # classDistribution = trainData["class"].value_counts()
    # classDistribution.plot.bar() # check distribution with barplot

    # combine terms from all observations to one list
    listOfAllTerms = [] # 582721 non-unique terms
    for articleTerms in trainData["articleTerms"]:
        listOfAllTerms.extend(articleTerms)

    # create dictionary of all term frequencies
    termFreq = collections.Counter(listOfAllTerms) # distribution of terms; 42773 distinct terms

    # TESTING: distribution of frequencies (to decide how much we remove)
    # termFreqDist = collections.Counter(termFreq.values())

    # remove terms with overall frequency less than 10
    filteredTermFreq = {x: count for x, count in termFreq.items() if count >= 10}  # 7697 terms left

    # save only keys to get the list of final terms
    finalTerms = list(filteredTermFreq.keys())

    return finalTerms



main()







