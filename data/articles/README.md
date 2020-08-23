data/articles/ is a folder for stashing all the parsed articles saved with its IDs defined in the URL string (article-"ID")


(1)

When you first run the src/parser.py from main.main, set the fromFile parameter to False:

    parsedData = parser.getParsedData(fromFile=False, newData=False, pathNewData=None)

After you parse the articles, you can then switch to reading the parsed articles from files with setting

    parsedData = parser.getParsedData(fromFile=False, newData=False, pathNewData=None)




(2)

Similarly set the fromFile argument according to your needs when running the src/parser from finalPrediction.main.
