from helpers import textCleanup, parser, tfidf, featureSelection, fit, predict
from sklearn.model_selection import train_test_split
import pickle
import nltk
import os

# can be removed after the first run
#nltk.download("punkt")
#nltk.download("stopwords")



def main():

    # parse/import parsed articles
    parsedData = parser.getParsedData(fromFile=True, newData=False, pathNewData=None)

    # split parsed data to train and test set (train: 3190 x 2, test: 798 x 2)
    trainData, testData = train_test_split(parsedData, test_size=0.2, random_state=11)

    # apply function for cleaning article content and splitting it into terms
    trainData = trainData.assign(articleTerms=trainData["text"].apply(textCleanup.getRelevantTerms))
    testData = testData.assign(articleTerms=testData["text"].apply(textCleanup.getRelevantTerms))

    # save variable we want to predict
    trainDataY, testDataY = trainData["class"], testData["class"]

    # get final terms used as features
    finalTerms = featureSelection.getTermFeatures(trainData)

    # save final terms for future predictions
    with open("data/finalTerms", "wb") as f:
        pickle.dump(finalTerms, f)

    # get tf-idf table from final terms defined on training data
    trainDataTfidf = tfidf.getTfIdfTable(trainData, listOfRelevantTerms=finalTerms)
    testDataTfidf = tfidf.getTfIdfTable(testData, listOfRelevantTerms=finalTerms)

    # fit models on training data (return tuple ({models}, {cvResults}))
    finalModels = fit.fitMain(trainDataTfidf, trainDataY)

    # predict on test set and choose the best model
    finalResults = predict.predict(finalModels[0], testDataTfidf, testDataY)

    return finalResults



main()
