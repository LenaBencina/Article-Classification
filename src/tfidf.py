# TF-IDF comments:
# TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
# IDF = (Total number of sentences (documents) + 1)/(Number of sentences (documents) containing the word + 1)
# TF-IDF = TF * [log(IDF) + 1]

import pandas as pd
import numpy as np



# main function to get tf-idf table from article terms
def getTfIdfTable(trainData, listOfRelevantTerms):

    # get tf (term frequency) for each article
    tfSeries = trainData["articleTerms"].apply(getTfTable, listOfAllTerms=listOfRelevantTerms)

    # split term counts for each article (=row) from list to multiple columns
    docTermFreqTable = pd.DataFrame(tfSeries.to_list(), columns=listOfRelevantTerms)

    # name rows with IDs
    idDict = trainData["id"].to_dict()
    docTermFreqTable = docTermFreqTable.rename(index=idDict)

    # get idf for each term (is the same for each document)
    numberOfAllDocs = len(docTermFreqTable)
    IdfSeries = docTermFreqTable.apply(getIdfList, numberOfAllDocs=numberOfAllDocs)

    # multiply tf table with idf to get tf-idf table
    tfIdfTable = docTermFreqTable * (IdfSeries.apply(np.log) + 1)
    # todo: test different ways of tfidf

    return tfIdfTable


##################################################################################################################

# count occurences of specific terms in one article's text
def getTfTable(oneArticleTerms, listOfAllTerms):

    tfList = list()

    # for each term in list of all terms
    for term in listOfAllTerms:

        # count occurences of the term in the article
        termCount = oneArticleTerms.count(term)

        # divide count with # of terms in one article
        tf = termCount/len(oneArticleTerms)
        tfList.append(tf)

    return tfList


# count documents that includes specific terms
def getIdfList(oneTermTfs, numberOfAllDocs):

    # number of documents that include this term (tf not zero)
    numberOfDocsWithTerm = sum(x > 0 for x in oneTermTfs)

    # divide with number of all documents to get idf (add 1 to prevent division with 0)
    idf = (numberOfDocsWithTerm + 1)/(numberOfAllDocs + 1)

    return idf

