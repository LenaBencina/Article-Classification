import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
from bs4 import BeautifulSoup
import requests
import re
import os


# main function for preparing train and test data
def getTrainTestData(fromFile):

    if fromFile:
        trainData = pd.read_csv("./data/parsedDataTrain.csv")
        testData = pd.read_csv("./data/parsedDataTest.csv")

    else:
        # import labeled data (4036 x 2)
        allLabeledData = pd.read_csv("./data/labeled_urls.tsv", sep="\t", header=None, names=["url", "class"])

        # filter labeled data (4034 x 2)
        filteredLabeledData = filterUrls(allLabeledData)

        # apply function for parsing url to the url column
        filteredLabeledData["text"] = filteredLabeledData["url"].apply(parseTextFromUrl)

        # remove articles without any content ( 3988 x 2 )
        filteredLabeledData = filteredLabeledData[filteredLabeledData["text"].notnull()]

        # split data to train and test set ( train: 3190 x 2, test: 798 x 2)
        trainData, testData = train_test_split(filteredLabeledData, test_size=0.2, random_state=11)

        # write to file train & test data
        trainData.to_csv("./data/parsedDataTrain.csv", index=False)
        testData.to_csv("./data/parsedDataTest.csv", index=False)

    return trainData, testData

##############################################################################

# (1) function for filtering urls
def filterUrls(labeledData):

    # first validate if article from zurnal
    labeledData["isZurnal"] = labeledData["url"].apply(validateZurnal)

    # get id from url
    labeledData["id"] = labeledData["url"].apply(getID)
    labeledData["isDuplicate"] = labeledData.duplicated(subset=["id"])


    ############ print articles that will be removed ################################
    removedNonZurnal = labeledData[labeledData.isZurnal == False]["url"].to_list()
    removedDuplicated = labeledData[labeledData.isDuplicate]["url"].to_list()

    print("Removed (non zurnal) articles: " + str(removedNonZurnal))
    print("Removed (duplicated) articles: " + str(removedDuplicated))
    #################################################################################

    # remove non zurnal articles
    filteredData = labeledData[labeledData.isZurnal]

    # remove articles with duplicate ids
    filteredData = filteredData[filteredData.isDuplicate == False]

    # remove columns used for filtering
    filteredData = filteredData.drop(columns=["isZurnal", "isDuplicate"])

    return filteredData



# (1)a
def validateZurnal(url):

    # test if substring 'zurnal24' in url
    if "zurnal24.si" not in url:
        return False

    else:
        return True


# (1)b
def getID(url):

    # get article ID from url
    urlID = re.findall(r"-\d+", url)

    if urlID:
        return urlID[-1].replace("-", "")

    else:
        None






# (2) function for parsing title, subtitle and main content of the article
def parseTextFromUrl(url):

    urlNameID = re.findall(r'-\d+', url)[-1]  # get article ID from url
    filePath = "../data/Articles/article" + urlNameID # define path for saving parsed article

    if path.exists(filePath): # if article already parsed
        #print('if: ', filePath, ', ', url)
        file = open(filePath, "r") # read the file
        articleTextAll = file.read()
        file.close()

    else:
        #print('else: ', filePath, ', ', url)

        # parse the article
        resp = requests.get(url)  # send request
        html_page = resp.content  # get content from url
        soup = BeautifulSoup(html_page, "html.parser")  # parse html

        ## get relevant texts
        # 1. title
        titleH1 = soup.find("h1", class_="article__title")
        titleText = titleH1.text if titleH1 else ""

        # 2. subtitle
        subtitleDiv = soup.find("div", class_="article__leadtext")
        subtitleText = subtitleDiv.text if subtitleDiv else ""

        # 3. content
        contentDiv = soup.find("div", class_="article__content")
        contentText = contentDiv.text if contentDiv else ""
        contentTextAllJoined = " ".join(contentText.split())

        # check if empty article or article without title
        if not titleText or not contentTextAllJoined:
            articleTextAll = None
            print('EMPTY ARTICLE: ' + url)

        else: # join all text into one string

            articleTextAll = " ".join((titleText, subtitleText, contentTextAllJoined))

            # save parsed content to file
            file = open(filePath, "w")
            file.write(articleTextAll)
            file.close()

    return articleTextAll





