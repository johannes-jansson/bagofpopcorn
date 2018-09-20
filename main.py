import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords


def cleanup(raw):
    plain_text = BeautifulSoup(raw, features="html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", plain_text)
    words = letters_only.lower().split()
    # set makes searching faster
    stops = set(stopwords.words("english"))
    actual_words = [w for w in words if w not in stops]
    return(" ".join(actual_words))


train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
num_reviews = train["review"].size

clean_train_reviews = []

for i in range(0, num_reviews):
    if((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(cleanup(train["review"][i]))
