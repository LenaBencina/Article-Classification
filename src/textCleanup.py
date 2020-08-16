import nltk
from nltk.corpus import stopwords
import re
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
# TODO add README for commands for downloading stopwords and punkt for nltk



# main function for cleaning article's content
def getRelevantTerms(text):

    ## 1. clean text and split it to list of terms
    termsList = getTermsFromArticle(text)

    ## 2. remove stopwords
    filteredTermsList = removeStopWords(termsList)

    ## 3. lemmatize
    lemmatizedTermsList = lemmatize(filteredTermsList)

    return lemmatizedTermsList





##########################################################

# cleanup the article content and list of words
def getTermsFromArticle(text):

    cleanedText = text.lower()  # remove caps
    # TODO remove quotes !!!!!
    cleanedText = re.sub(r"\W", " ", cleanedText)  # change any character which is not a word character to space
    cleanedText = re.sub(r"\s+", " ", cleanedText)  # remove double spaces from the text
    cleanedText = re.sub(r"\b[0-9]+\b", "", cleanedText) # remove standalone numbers
    cleanedText = cleanedText.strip()  # remove empty spaces from the beginning/end of the text

    # split into words
    splitText = nltk.word_tokenize(cleanedText)

    return splitText


# remove stop words
def removeStopWords(termList):

    stopWords = set(stopwords.words("slovene"))
    filteredTerms = [word for word in termList if word not in stopWords]

    return filteredTerms



# lemmatize terms
def lemmatize(filteredTermsList):

    # TODO rethink to move lemmatizer call to main (singleton)
    lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

    # lemmatize each term and save it to list
    lemmatizedTermsList = []
    for term in filteredTermsList:
        lemmatizedTerm = lemmatizer.lemmatize(term)
        lemmatizedTermsList.append(lemmatizedTerm)

    return lemmatizedTermsList



