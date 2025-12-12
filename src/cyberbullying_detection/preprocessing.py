from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


def split_data():
    data = pd.read_csv("cyberbullying_tweets.csv")
    texts = data["tweet_text"]
    labels = data["cyberbullying_type"]

    texts_train, texts_val_test, labels_train, labels_val_test = train_test_split(
        texts, labels, train_size=TRAIN_SIZE, random_state=SEED, shuffle=True
    )

    texts_val, texts_test, labels_val, labels_test = train_test_split(
        texts_val_test,
        labels_val_test,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=SEED,
        shuffle=True,
    )

    data_train = pd.concat([texts_train, labels_train], axis=1)
    data_val = pd.concat([texts_val, labels_val], axis=1)
    data_test = pd.concat([texts_test, labels_test], axis=1)

    data_train.to_csv(Path("./data/data_train.csv"))
    data_val.to_csv(Path("./data/data_val.csv"))
    data_test.to_csv(Path("./data/data_test.csv"))


if __name__ == "__main__":
    split_data()
