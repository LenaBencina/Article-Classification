# TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
# IDF: (Total number of sentences (documents) + 1)/(Number of sentences (documents) containing the word + 1)

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

    return tfIdfTable


#######################################################################################

# count occurences of specific words in one article's text
def getTfTable(oneArticleTerms, listOfAllTerms):

    # count all terms in one article
    tfList = list()
    for term in listOfAllTerms: # for each term in list of all terms
        termCount = oneArticleTerms.count(term) # count occurences of the term in the article
        tf = termCount/len(oneArticleTerms)
        tfList.append(tf)

    # divide list of counts with # of terms in one article
    return tfList

# testing function
#getTFtable(['A', 'B', 'A'], ['A', 'B', 'C', 'D' 'E', 'F'])


def getIdfList(oneTermTfs, numberOfAllDocs):

    numberOfDocsWithTerm = sum(x > 0 for x in oneTermTfs) # number of documents that include this term
    idf = (numberOfDocsWithTerm + 1)/(numberOfAllDocs + 1) # divide with number of all documents to get idf

    return idf # TODO apply log




# testing alltogether
#testingData = pd.DataFrame({"articleTerms": [["A", "B", "A"], ["B", "C", "A", "B"], ["C", "C", "C", "C", "C"], ["A", "B", "E"]]})
#testingData["id"] = [1,2,3,4]
#tfidfTest = getTfIdfTable(testingData, ["A", "B", "C", "D", "E"])


