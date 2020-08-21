import collections

# todo: find a smarter way to choose final features
# todo: PCA or some similar algorithm for demansionality reduction

# function to get final terms, i.e. features out of the training data
def getTermFeatures(trainData):

    # TESTING: class distribution
    # classDistribution = trainData["class"].value_counts()
    # classDistribution.plot.bar() # check distribution with barplot

    # combine terms from all observations to one list
    listOfAllTerms = []  # 582721 non-unique terms
    for articleTerms in trainData["articleTerms"]:
        listOfAllTerms.extend(articleTerms)

    # create dictionary of all term frequencies
    termFreq = collections.Counter(listOfAllTerms)  # distribution of terms; 42773 distinct terms

    # TESTING: distribution of frequencies (to decide how much we remove)
    # termFreqDist = collections.Counter(termFreq.values())

    # remove terms with overall frequency less than 10
    filteredTermFreq = {x: count for x, count in termFreq.items() if count >= 10}  # 7697 terms left

    # save only keys to get the list of final terms
    finalTerms = list(filteredTermFreq.keys())

    return finalTerms
