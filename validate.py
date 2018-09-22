import pandas as pd
df = pd.read_csv("data/Bag_of_Words_model.csv", header=0,
                 delimiter=",", quoting=3)

hits = 0
for index, row in df.iterrows():
    if (row['sentiment'] == row['guess']):
        hits += 1
print(hits / 5000.0)
