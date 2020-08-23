from helpers import textCleanup, parser, tfidf
import pickle
from sklearn.metrics import accuracy_score

def main():

    # parse/import parsed articles
    parsedData = parser.getParsedData(fromFile=False, newData=True, pathNewData="data/unlabeledUrlsForNewPredictions.csv")

    # apply function for cleaning article content and splitting it into terms
    parsedData = parsedData.assign(articleTerms=parsedData["text"].apply(textCleanup.getRelevantTerms))

    # import final terms for calculating tfidf
    with open("data/finalTerms", "rb") as f:
        finalTerms = pickle.load(f)

    # get tf-idf table from training data i.e. document terms
    tfidfData = tfidf.getTfIdfTable(parsedData, listOfRelevantTerms=finalTerms)

    # predict article class
    with open("models/svmModelSigmoid-11", "rb") as f:
        model = pickle.load(f)


    # prepare final table
    finalTable = parsedData["url"].to_frame()
    finalTable = finalTable.assign(predictedClass=model.predict(tfidfData))

    # parse actual class from link and add it to table
    finalTable["actualClass"] = finalTable["url"].str.split('/', n = 4, expand = True)[3]

    # calculate accuracy
    accuracy = accuracy_score(finalTable["actualClass"], finalTable["predictedClass"], normalize=True) * 100

    # report predictions
    print("\n Prediction results:")
    print(finalTable)

    # report accuracy
    print("\n Accuracy: " + str(accuracy))

    # save final table
    finalTable.to_csv("data/finalPredictions.csv", index=False)

    return



main()