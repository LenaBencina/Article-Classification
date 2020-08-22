# Article-Classification


The main idea of this project is to get the article's content (title, subtitle and main content) from its url and build a model which will be able to classify 
parsed article into one of the five classes (svet, sport, magazin, slovenija, avto). 

We process and tranform text with [tf-idf statistic](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to get numerical features. 

At this point we select features we will use for model training; 
for now very simple method based on its frequency is used, which can be improved with PCA or some other method for dimensionality reduction, as well as with 
some other method for smarter feature selection.

On this dataset we then train the following classification algorithms: *logistic regression*, *k-nearest neighbours*, *decision tree*, *random forest*, 
*support vector machines* and *naive bayes*. We use 3-fold cross validation with parameter tuning and choose the best model based on *accuracy* and *F1 measure*.

We save the best model and prepare a script **predictFinal** to classify new articles.

Notes for running **predictFinal** on new URLs:

- New URLs should be inserted in **data/unlabeled_urls_test.csv** (url by row)
- This script is working only on article's from [zurnal24.si](https://www.zurnal24.si/)
- Make predictions for new URLs with: **python predictFinal.py**
- Returns a table with "URL, predictedClass, actualClass" per row (actualClass is parsed from the URL string)


------------------------------------------------------------------------------------------------------------------------------------------------------------

This repo consist of:

### (1) src folder

**main.py** - main script to run the **parser**, **textCleanup**, **tfidf**, **fit** (on training data) and **predict** (on testing data)

**parser.py** - script for validating URLs, parsing content from URLs and saving the content into **data/Article/** by its ID (parsed from the URL string)
Note: when calling **parser**, there are two options available; parsing the articles (fromFile=False) or reading already parsed articles (fromFile=True)

**textCleanup.py** - script for cleaning text (article's content); removing non-word characters, double spaces, standalone numbers, 
empty spaces; tokenizing text into words i.e. terms; removing terms without meaning (i.e. stop words); and lemmatizing the terms

**featureSelection.py** - script for selecting features i.e. terms from list of all terms (defined in **textCleanup**) for final training

**tfidf.py** - script for calculating tf-idf (term frequency-inverse document frequency) for each (article, term) pair to get numerical representation 
of the text data

**fit.py** - script for building a model with 3-fold cross validation, parameter tuning and choosing the best three models based on accuracy and F1 measure

**predict.py** - script for choosing the best model from the three best models chosen in **fit** (using all three models on the testing data and choose the one 
with the best accuracy and F1 values)

**predictFinal.py** - script for parsing new article's content and classifying it with the best model chosen in **predict** 

### (2) data folder

**Article/** - folder for saving all the parsed articles (each article is saved by its ID parsed from its URL)

**labeled_urls.tsv** - initial input file with "url \t articleClass" per row

**unlabeled_urls_test.csv** - input file for **predictFinal** with "url" per row (file with URLs of new articles)

**parsedData.csv** - all parsed data; used for faster model training (to avoid parsing articles whenever we run the script; 
to use this set fromFile=True when calling **parser**)

**finalTerms** - pickle object with final terms i.e. features (needed for future predictions)

**validationResults** - pickle object with accuracy and F1 values from cross validation (for reruning cross validation set fromFile=False when
calling fitWithCrossValidation in fit.fitMain)


### (3) models folder

The best models chosen in **fit** are stashed in this folder. Model name consists of three parts: 
algorithm name, chosen parameter (if any) and random seed (included for reproducible results).
