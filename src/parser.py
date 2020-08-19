import pandas as pd
from os import path
from bs4 import BeautifulSoup
import requests
import re



# main function for getting the data from urls
def getParsedData(fromFile):

    if fromFile: # import already parsed data
        parsedData = pd.read_csv("./data/parsedData.csv")

    else:
        # import labeled data with url and article class variables (4036 x 2)
        allLabeledData = pd.read_csv("./data/labeled_urls.tsv", sep="\t", header=None, names=["url", "class"])

        # filter labeled data (4034 x 2)
        filteredLabeledData = filterUrls(allLabeledData)

        # apply function for parsing url to the url column
        filteredLabeledData["text"] = filteredLabeledData["url"].apply(parseTextFromUrl)

        # remove articles without any content (3988 x 2)
        filteredLabeledData = filteredLabeledData[filteredLabeledData["text"].notnull()]

    return filteredLabeledData




#####################################################################################################################

# (1) filter urls
def filterUrls(labeledData):

    # first validate if article is from zurnal
    labeledData["isZurnal"] = labeledData["url"].apply(validateZurnal)

    # get id from url
    labeledData["id"] = labeledData["url"].apply(getID)
    labeledData["isDuplicate"] = labeledData.duplicated(subset=["id"]) # mark duplicated ids
    # todo: define a smarter way for validating correct/unique article/url

    ######### print articles that will be removed
    removedNonZurnal = labeledData[labeledData.isZurnal==False]["url"].to_list()
    removedDuplicated = labeledData[labeledData.isDuplicate]["url"].to_list()

    print("Removed (non zurnal) articles: " + str(removedNonZurnal))
    print("Removed (duplicated) articles: " + str(removedDuplicated))
    ##########

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

    # if there is no ID, not a valid article
    else:
        None






# (2) parse title, subtitle and main content of the article from a given url
def parseTextFromUrl(url):

    urlNameID = re.findall(r'-\d+', url)[-1]  # get article ID from url
    filePath = "../data/Articles/article" + urlNameID # define path for saving parsed article

    if path.exists(filePath): # if article already parsed
        # read the file
        file = open(filePath, "r")
        articleTextAll = file.read()
        file.close()

    else:
        # parse the article
        resp = requests.get(url) # send request
        html_page = resp.content # get content from url
        soup = BeautifulSoup(html_page, "html.parser")  # parse html

        ## get relevant texts; if there is no text, return empty string
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
            print("EMPTY ARTICLE: " + url)

        else:
            # join all text into one string
            articleTextAll = " ".join((titleText, subtitleText, contentTextAllJoined))

            # save parsed content to file
            file = open(filePath, "w")
            file.write(articleTextAll)
            file.close()

    return articleTextAll





