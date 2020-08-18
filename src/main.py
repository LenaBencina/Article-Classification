from src import textCleanup, prepareTrainTest, tfidf, fit
import collections
import pandas as pd



def main():

    trainTestData = prepareTrainTest.getTrainTestData(fromFile = True)
    trainData, testData = trainTestData[0], trainTestData[1]


    # apply function for cleaning article content and splitting it into terms
    trainData["articleTerms"] = trainData["text"].apply(textCleanup.getRelevantTerms)
    testData["articleTerms"] = testData["text"].apply(textCleanup.getRelevantTerms)


    # save variable we want to predict
    trainDataY, testDataY = trainData["class"], testData["class"]
    #trainDataY.to_csv("./data/trainDataY.csv", index=False)
    #testDataY.to_csv("./data/testDataY.csv", index=False)


    # check class distribution
    # classDistribution = trainData["class"].value_counts()
    # classDistribution.plot.bar() # check distribution with barplot


    # combine terms from all observations to one list
    listOfAllTerms =[] # 582721 non-unique terms
    for articleTerms in trainData["articleTerms"]:
        listOfAllTerms.extend(articleTerms)


    # create dictionary of all term frequencies
    termFreq = collections.Counter(listOfAllTerms) # distribution of terms (42773 distinct terms)
    #termFreqDist = collections.Counter(termFreq.values()) # distribution of frequencies (to decide how much we remove)

    # remove terms with frequency less than 10
    filteredTermFreq = {x: count for x, count in termFreq.items() if count >= 10} # 7697

    # todo fix reading from file vs calculating
    # get tf-idf table from training data (=document terms)
    #trainDataTfidf = tfidf.getTfIdfTable(trainData, listOfRelevantTerms=list(filteredTermFreq.keys()))
    #testDataTfidf = tfidf.getTfIdfTable(testData, listOfRelevantTerms=list(filteredTermFreq.keys()))

    # save - testing
    #trainDataTfidf.to_csv("./data/trainDataTfidf.csv", index=False)
    #testDataTfidf.to_csv("./data/testDataTfidf.csv", index=False)


    # todo add calling fitMain from main
    fit.fitMain()


main()