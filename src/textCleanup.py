import nltk
from nltk.corpus import stopwords
import re
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
# nltk.download('punkt') RUN
# nltk.download('stopwords') RUN



# main function for cleaning article's content
def getRelevantTerms(text):

    # clean text and split it to list of terms
    termsList = getTermsFromArticle(text)

    # remove stopwords
    filteredTermsList = removeStopWords(termsList)

    # lemmatize
    lemmatizedTermsList = lemmatize(filteredTermsList)

    return lemmatizedTermsList



#################################################################################

# cleanup the article content and list of words
def getTermsFromArticle(text):

    # remove caps
    cleanedText = text.lower()

    # change any character which is not a word character to space
    cleanedText = re.sub(r"\W", " ", cleanedText)

    # remove double spaces from the text
    cleanedText = re.sub(r"\s+", " ", cleanedText)

    # remove standalone numbers
    cleanedText = re.sub(r"\b[0-9]+\b", "", cleanedText)

    # remove empty spaces from the beginning/end of the text
    cleanedText = cleanedText.strip()

    # split into words
    splitText = nltk.word_tokenize(cleanedText)

    return splitText


# remove stop words
def removeStopWords(termList):

    # define slovenian stop words
    stopWords = set(stopwords.words("slovene"))

    # remove stopwords
    filteredTerms = [word for word in termList if word not in stopWords]

    return filteredTerms



# lemmatize terms
def lemmatize(filteredTermsList):

    # todo: rethink to move lemmatizer call to main (singleton)
    lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

    # lemmatize each term and save it to list
    lemmatizedTermsList = []
    for term in filteredTermsList:
        lemmatizedTerm = lemmatizer.lemmatize(term)
        lemmatizedTermsList.append(lemmatizedTerm)

    return lemmatizedTermsList



