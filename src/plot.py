import matplotlib.pyplot as plt


# PLOT FREQUENCIES
def plotFrequencies(freqDict, n, first):

    # sort dictionary of term:frequency by frequency
    wordFreqSorted = sorted(freqDict.items(), key=lambda x: x[1], reverse=True)

    # get only n most frequent terms
    if first:
        wordFreqSorted = wordFreqSorted[:n]
    else:
        wordFreqSorted = wordFreqSorted[-n:]


    # plot
    plt.bar(range(len(wordFreqSorted)), [val[1] for val in wordFreqSorted], align='center')
    plt.xticks(range(len(wordFreqSorted)), [val[0] for val in wordFreqSorted])
    plt.xticks(rotation=70)
    plt.show()




### TESTING FROM MAIN

########## Plotting terms frequencies
# plot frequencies of n most/lest frequent terms
#plot.plotFrequencies(testingTermFreq, n=20, first=True)
#plot.plotFrequencies(testingTermFreq, n=20, first=False)


# plot distribution of frequencies
#plt.hist(collections.Counter(filteredTermFreq).values())
##########
