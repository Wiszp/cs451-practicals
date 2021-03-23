"""
Jack English
CSCI 0451 Practical 5, Professor Foley
March 16, 2021
"""

"""
In this lab, we once again have a mandatory 'python' challenge.
Then we have a more open-ended Machine Learning 'see why' challenge.

This data is the "Is Wikipedia Literary" that I pitched.
You can contribute to science or get a sense of the data here: https://label.jjfoley.me/wiki
"""

import gzip, json
from shared import (
    dataset_local_path,
    bootstrap_accuracy,
    simple_boxplot,
)  # added bootstrap_accuracy and simple_boxplot

from dataclasses import dataclass
from typing import Dict, List


"""
Problem 1: We have a copy of Wikipedia (I spared you the other 6 million pages).
It is separate from our labels we collected.
"""


@dataclass
class JustWikiPage:
    title: str
    wiki_id: str
    body: str


# Load our pages into this pages list.
pages: List[JustWikiPage] = []  # initialize pages as type JustWikiPage
with gzip.open(dataset_local_path("tiny-wiki.jsonl.gz"), "rt") as fp:
    for line in fp:
        entry = json.loads(line)
        pages.append(JustWikiPage(**entry))


@dataclass
class JustWikiLabel:
    wiki_id: str
    is_literary: bool


# Load our judgments/labels/truths/ys into this labels list:
labels: List[JustWikiLabel] = []
with open(dataset_local_path("tiny-wiki-labels.jsonl")) as fp:
    for line in fp:
        entry = json.loads(line)
        labels.append(
            JustWikiLabel(wiki_id=entry["wiki_id"], is_literary=entry["truth_value"])
        )


@dataclass
class JoinedWikiData:
    wiki_id: str
    is_literary: bool
    title: str
    body: str


print(len(pages), len(labels))
print(pages[0])
print(labels[0])

joined_data: Dict[str, JoinedWikiData] = {}


# create a list of JoinedWikiData from the ``pages`` and ``labels`` lists."
for p in pages:
    joined_data[p.wiki_id] = JoinedWikiData(
        p.wiki_id, is_literary=False, title=p.title, body=p.body
    )
# The element in the joined data that corresponds to the wiki id of that page is a JoinedWikiData object taking stuff from page p
# and then defaulting is_literary to false

for l in labels:
    joined_data[l.wiki_id].is_literary = l.is_literary
# Correctly update the labels for each page in the joined thing based on the label


# Make sure Problem 1 is solved correctly!
assert len(joined_data) == len(pages)
assert len(joined_data) == len(labels)
# Make sure it has *some* positive labels!
assert sum([1 for d in joined_data.values() if d.is_literary]) > 0
# Make sure it has *some* negative labels!
assert sum([1 for d in joined_data.values() if not d.is_literary]) > 0

# Construct our ML problem:
ys = []
examples = []
for wiki_data in joined_data.values():
    ys.append(wiki_data.is_literary)
    examples.append(wiki_data.body)

## We're actually going to split before converting to features now...
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.75,
    shuffle=True,
    random_state=RANDOM_SEED,
)
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

## Convert to features, train simple model (TFIDF will be explained eventually.)
from sklearn.feature_extraction.text import TfidfVectorizer

# Only learn columns for words in the training data, to be fair.
word_to_column = TfidfVectorizer(
    strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
)
word_to_column.fit(ex_train)

# Test words should surprise us, actually!
X_train = word_to_column.transform(ex_train)
X_vali = word_to_column.transform(ex_vali)
X_test = word_to_column.transform(ex_test)


print("Ready to Learn!")  # Fired up and ready to go
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

models = {
    "SGDClassifier": SGDClassifier(),
    "Perceptron": Perceptron(),
    "LogisticRegression": LogisticRegression(),
    "DTree": DecisionTreeClassifier(),
}

# What was provided
for name, m in models.items():
    m.fit(X_train, y_train)
    print("{}:".format(name))
    print("\tVali-Acc: {:.3}".format(m.score(X_vali, y_vali)))
    if hasattr(m, "decision_function"):
        scores = m.decision_function(X_vali)
    else:
        scores = m.predict_proba(X_vali)[:, 1]
    print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=scores, y_true=y_vali)))

# Bootstrapping time

fitmodels = {}  # dictionary that will hold the fitted models

# Python wants to auto=format this, hence the added
# indentation
for name, model in models.items():  # iterate through our models
    fitmodels[name] = model.fit(
        X_train, y_train
    )  # fit them, and tuck them into the dictionary

# Helper method to make a series of box-plots from the dictionary
# of the fitted models.
simple_boxplot(
    {
        # Python wants to auto=format this, hence the added
        # indentation
        "Logistic Regression": bootstrap_accuracy(
            fitmodels["LogisticRegression"], X_vali, y_vali
        ),
        "Perceptron": bootstrap_accuracy(fitmodels["Perceptron"], X_vali, y_vali),
        "Decision Tree": bootstrap_accuracy(fitmodels["DTree"], X_vali, y_vali),
        "SGDClassifier": bootstrap_accuracy(fitmodels["SGDClassifier"], X_vali, y_vali),
    },
    title="Validation Accuracy",
    xlabel="Model",
    ylabel="Accuracy",
    save="model-cmp.png",
)
"""
Results should be something like:

SGDClassifier:
        Vali-Acc: 0.84
        Vali-AUC: 0.879
Perceptron:
        Vali-Acc: 0.815
        Vali-AUC: 0.844
LogisticRegression:
        Vali-Acc: 0.788
        Vali-AUC: 0.88
DTree:
        Vali-Acc: 0.739
        Vali-AUC: 0.71
"""
TODO("2. Explore why DecisionTrees are not beating linear models. Answer one of:")
TODO("2.A. Is it a bad depth?")
TODO("2.B. Do Random Forests do better?")
TODO(
    "2.C. Is it randomness? Use simple_boxplot and bootstrap_auc/bootstrap_acc to see if the differences are meaningful!"
    # No, it is not randomness. The DecisionTree model(s) are not doing as well as the other models.
)
TODO("2.D. Is it randomness? Control for random_state parameters!")
