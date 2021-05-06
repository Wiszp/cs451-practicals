# Author: Jack English
# CSCI 451 Practical 15
# May 6th, 2021
#%%
import random
from shared import bootstrap_accuracy, simple_boxplot
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import typing as T
from dataclasses import dataclass

#%%

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

(N, D) = X_train.shape
#%% Train up Forest models:

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print("Forest.score = {:.3}".format(forest.score(X_vali, y_vali)))

lr = LogisticRegression()
lr.fit(X_train, y_train)
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
graphs = {
    "RF": bootstrap_accuracy(forest, X_vali, y_vali),
    "SGD": bootstrap_accuracy(sgd, X_vali, y_vali),
    "LR": bootstrap_accuracy(lr, X_vali, y_vali),
}

#%% SVM
from sklearn.svm import SVC as SVMClassifier

configs = []
configs.append({"kernel": "linear"})
configs.append({"kernel": "poly", "degree": 2})
configs.append({"kernel": "poly", "degree": 3})
configs.append({"kernel": "rbf"})
# configs.append({"kernel": "sigmoid"}) # just awful.


@dataclass
class ModelInfo:
    name: str
    accuracy: float
    model: T.Any
    X_vali: T.Optional[np.ndarray] = None


# TODO: C is the most important value for a SVM.
#       1/C is how important the model stays small.
# Bringing the C value down absolutely tanks the accuracy of the k = poly2 and k = poly3 models
# Raising it increases accuracy marginally at first, but then the spread begins to increase.
# Also, as the C value goes up, accuracy and the accuracy ranges become more uniform among the models.
# This is particularly noticeable if you kick it all the way up something like 100.0
# TODO: RBF Kernel is the best; explore its 'gamma' parameter.
# From https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
#  "Intuitively, the gamma parameter defines how far the influence of a single training example reaches,
# with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the
# inverse of the radius of influence of samples selected by the model as support vectors."
# A small gamma value is too restrictive, whereas conversely one that is too high will lead to overfitting

# Based on my experimentation, the auto gamma seemed to do the best (not shocking). 0.001 performed very poorly,
# and 0.1 and 0.2 performed very well. Higher stuff was somewhat of a mess.
for cfg in configs:
    variants: T.List[ModelInfo] = []
    for gamma in [0.2, 0.1, 0.01, 0.001, "auto"]:
        if cfg["kernel"] == "rbf" and gamma != "auto":
            continue
        for class_weights in [None, "balanced"]:
            for c_val in [5.0]:
                svm = SVMClassifier(
                    C=c_val, class_weight=class_weights, gamma=gamma, **cfg
                )
                svm.fit(X_train, y_train)
                name = "k={}{} gamma = {} C={} {}".format(
                    cfg["kernel"],
                    cfg.get("degree", ""),
                    # cfg.get("gamma", "auto"), This is what Professor Foley had
                    gamma,
                    c_val,
                    class_weights or "",
                )
                accuracy = svm.score(X_vali, y_vali)
                print("{}. score= {:.3}".format(name, accuracy))
                variants.append(ModelInfo(name, accuracy, svm))
    best = max(variants, key=lambda x: x.accuracy)
    graphs[best.name] = bootstrap_accuracy(best.model, X_vali, y_vali)


simple_boxplot(
    graphs,
    title="Kernelized Models for Poetry",
    ylabel="Accuracy",
    save="graphs/p15-kernel-cmp.png",
)