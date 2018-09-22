import pandas as pd
from bs4 import BeautifulSoup
import re
# import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def cleanup(raw):
    plain_text = BeautifulSoup(raw, features="html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", plain_text)
    words = letters_only.lower().split()
    # set makes searching faster
    stops = set(stopwords.words("english"))
    actual_words = [w for w in words if w not in stops]
    return(" ".join(actual_words))


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

train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
# vocab = vectorizer.get_feature_names()


# Create and run the classifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])


# Done with training


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
