import pandas as pd
from bs4 import BeautifulSoup
import re
# import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def cleanup(raw):
    plain_text = BeautifulSoup(raw, features="html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", plain_text)
    words = letters_only.lower().split()
    # set makes searching faster
    stops = set(stopwords.words("english"))
    actual_words = [w for w in words if w not in stops]
    return(" ".join(actual_words))


def train():
    # load data and split it into train and test
    df = pd.read_csv("data/labeledTrainData.tsv", header=0,
                     delimiter="\t", quoting=3)
    train, test = np.split(df, [20000], axis=0)

    # clean up training data
    num_reviews = train["review"].size
    clean_train_reviews = []
    for i in range(0, num_reviews):
        if((i + 1) % 1000 == 0):
            print("Review %d of %d\n" % (i + 1, num_reviews))
        clean_train_reviews.append(cleanup(train["review"][i]))

    # Create and fit the data model
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    # vectorizer = TfidfVectorizer(analyzer="word",
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None,
    #                              # max_df=0.9,
    #                              # min_df=0.1,
    #                              max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
    # vocab = vectorizer.get_feature_names()

    # Create and run the classifier
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])

    # Save the models
    joblib.dump(vectorizer, 'data/vectorizer.pkl')
    joblib.dump(forest, 'data/model.pkl')


def test():
    # Load the models
    vectorizer = joblib.load('data/vectorizer.pkl')
    forest = joblib.load('data/model.pkl')

    df = pd.read_csv("data/labeledTrainData.tsv", header=0,
                     delimiter="\t", quoting=3)
    train, test = np.split(df, [20000], axis=0)

    # test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t",
    #                    quoting=3)

    # clean up training data
    num_reviews = test["review"].size
    clean_test_reviews = [] 

    for i in range(20000, 20000 + num_reviews):
        if((i + 1) % 1000 == 0):
            print("Review %d of %d\n" % (i + 1, 20000 + num_reviews))
        clean_test_reviews.append(cleanup(test["review"][i]))

    # Get a bag of words for the test set
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test["sentiment"], "guess": result})

    # Use pandas to write the comma-separated output file
    output.to_csv("data/Bag_of_Words_model.csv", index=False, quoting=3)


def validate():
    df = pd.read_csv("data/Bag_of_Words_model.csv", header=0,
                     delimiter=",", quoting=3)
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    for index, row in df.iterrows():
        if (row['guess']):
            if(row['sentiment']):
                tps += 1
            else:
                fps += 1
        else:
            if(row['sentiment']):
                fns += 1
            else:
                tns += 1
    
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1 = 2 * precision * recall / (precision + recall)

    print("correct estimates: {} %".format((tns + tps) / 50.0))
    print("precision:         {} %".format(100 * precision))
    print("recall:            {} %".format(100 * recall))
    print("f1 score:          {} %".format(100 * f1))
    print("true positives:    {} %".format((tps) / 50.0))
    print("true negatives:    {} %".format((tns) / 50.0))
    print("false positives:   {} %".format((fps) / 50.0))
    print("false negatives:   {} %".format((fns) / 50.0))


train()
test()
validate()
