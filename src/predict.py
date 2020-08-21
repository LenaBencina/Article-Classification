from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd


def predict(finalModels, testDataTfidf, testDataY):

    # predict with each model and calculate accuracy & f1
    results = {}

    # key = name, value = model
    for key, value in finalModels.items():
        predictedY = value.predict(testDataTfidf)
        accuracy = accuracy_score(testDataY, predictedY, normalize=True) * 100
        f1 = f1_score(testDataY, predictedY, average="macro")
        results[key] = {"accuracy": accuracy, "f1": f1}

    # convert to table and sort by accuracy
    resultsTable = pd.DataFrame.from_dict(results).transpose().sort_values(by=["accuracy"])

    # report final results
    print("\n Final testing results:")
    print(resultsTable)

    return