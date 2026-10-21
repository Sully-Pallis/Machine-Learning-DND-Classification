# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:04:23 2026

@author: sully
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import localtime, strftime
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("over_one_mil_chars.csv")


df = df.drop(["gold", "name", "background", "race", "subclass_starting",
              "class_other", "subclass_other", "feats", "inventory",
              "date_modified", "notes_len", "Unnamed: 0", "char_id"], axis=1)

X = df.drop("class_starting", axis=1)
y = df["class_starting"]




features = X.columns

"""
for i in range(len(features)):
    for j in range(i+1, len(features)):
        plt.figure()
        plt.scatter(X[features[i]], X[features[j]])
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.title(f"{features[i]} vs {features[j]}")
        plt.show()
 
for feature in features:
    plt.figure()
    plt.hist(X[feature], bins=25)
    plt.xlabel(feature)
    plt.title(feature)
    plt.show()
"""

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_validate, y_train, y_validate = train_test_split(X,y, train_size=0.8, test_size=0.2, shuffle=True, stratify=y)
X_small, X_val_sm, y_small, y_val_sm = train_test_split(X, y, train_size=20000,test_size=20000, shuffle=True, stratify=y)

"""
# decision tree
dt = DecisionTreeClassifier(criterion="entropy", max_depth=20, min_samples_split=50, min_samples_leaf=20)
results = dt.cost_complexity_pruning_path(X_train, y_train)

num = 1
alpha_vals = {}small_test
for alpha in results["ccp_alphas"]:
    if alpha < 0:
        continue
    
    model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha, max_depth=20, min_samples_split=30, min_samples_leaf=20)
    yp = cross_val_predict(model, X_small, y_small, cv=5)
    acc = accuracy_score(y_small, yp)
    pre = precision_score(y_small, yp, average="macro")
    rec = recall_score(y_small, yp, average="macro")
    alpha_vals[alpha] = {"accuracy": acc, "precision": pre, "recall": rec}
    print("iteration of loop" + str(num))
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    num += 1
    
plt.figure()
for a in alpha_vals.keys():
    acc = alpha_vals[a]["accuracy"]
    plt.plot(a, acc, marker="^")
plt.show()


best_alpha = None
best_acc = 0

for alpha, metrics in alpha_vals.items():
    if (metrics["accuracy"] > best_acc):
        best_acc = metrics["accuracy"]
        best_alpha = alpha
"""

# Random Forest 
print("start")
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
rf = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5,
                            min_samples_leaf=2, bootstrap=True)
rf.fit(X_train, y_train)

print("Unique y_validate:", np.unique(y_validate))
print("Unique tree preds:", np.unique(rf.estimators_[0].predict(X_validate)))

rf_accuracy = accuracy_score(y_train, rf.predict(X_train))

tree_accuracies = [accuracy_score(y_train, tree.predict(X_train)) for tree in rf.estimators_]
plt.figure()
plt.scatter(range(len(tree_accuracies)), tree_accuracies, label="individual tree accuracy")
plt.axhline(np.mean(tree_accuracies), color = "red", linestyle="--", label="mean accuracy")
plt.axhline(rf_accuracy, color="purple", linestyle=":", label = "RF accuracy")
plt.legend()
plt.show()
print("Done")
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
