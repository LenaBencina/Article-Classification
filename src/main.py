from src import textCleanup, prepareTrainTest, tfidf
import collections





trainTestData = prepareTrainTest.getTrainTestData(fromFile = True)
trainData, testData = trainTestData[0], trainTestData[1]

# apply function for cleaning article content and splitting it into terms
trainData["articleTerms"] = trainData["text"].apply(textCleanup.getRelevantTerms)

# check class distribution
# classDistribution = trainData["class"].value_counts()
# classDistribution.plot.bar() # check distribution with barplot


# combine terms from all observations to one list
listOfAllTerms =[] # 582721 non-unique terms
for articleTerms in trainData["articleTerms"]:
    listOfAllTerms.extend(articleTerms)


# create dictionary of all term frequencies
termFreq = collections.Counter(listOfAllTerms) # distribution of terms (42773 distinct terms)
termFreqDist = collections.Counter(termFreq.values()) # distribution of frequencies (to decide how much we remove)

# remove terms with frequency less than 10
filteredTermFreq = {x: count for x, count in termFreq.items() if count >= 10} # 7697

# get tf-idf table from training data (=document terms)
tfIdfTable = tfidf.getTfIdfTable(trainData, listOfRelevantTerms=list(filteredTermFreq.keys()))


print(1)

