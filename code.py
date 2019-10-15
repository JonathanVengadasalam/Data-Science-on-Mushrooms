
import pandas as pd
raw_data = pd.read_csv("TP_2_datset_mushrooms.csv")
raw_data.shape

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in raw_data.columns:
    raw_data[col] = labelencoder.fit_transform(raw_data[col])
print(raw_data.head())

from sklearn.model_selection import train_test_split
x = raw_data.iloc[:,1:23]
y = raw_data.iloc[:,0]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

lr = LogisticRegression()
lr.fit(xtrain, ytrain)

yprob = lr.predict_proba(xtest)[:,1]
ypred = lr.predict(xtest)

fpr, tpr, thr = roc_curve(ytest, yprob)
roc_auc = auc(fpr, tpr)
print(roc_auc)


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, color="red", label= "AUC = %0.2f" % roc_auc)
plt.plot([0,1], [0,1], linestyle= "--")
plt.legend(loc = "lower right")
plt.axis("tight")
plt.ylabel("True positive rate")
plt.xlabel("False positve rate")
plt.show()


from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=3, kernel="linear")
pca.fit(xtrain)
xprojected = pca.transform(xtrain)
xprojtest = pca.transform(xtest)

In [22]:
from itertools import combinations
from math import ceil

combs = list(combinations(range(len(pca.lambdas_)), 2))
n_line = ceil(len(combs)/3)
fig, axes = plt.subplots(n_line, 3, figsize=(18, 5*n_line))

for ax, comb in zip(axes.ravel(), combs):
    ax.scatter(xprojected[:,comb[0]], xprojected[:,comb[1]], c=ytrain, s=1)
    ax.set_title("Components : {} - {}".format(comb[0],comb[1]))
    

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
fa.fit(xtrain)
xprojected_fa = fa.transform(xtrain)
xprojtest_fa = fa.transform(xtest)

In [20]:
combs = list(combinations(range(fa.components_.shape[0]), 2))
n_line = ceil(len(combs)/3)
fig, axes = plt.subplots(n_line, 3, figsize=(18, 5*n_line))

for ax, comb in zip(axes.ravel(), combs):
    ax.scatter(xprojected_fa[:,comb[0]], xprojected_fa[:,comb[1]], c=ytrain, s=1)
    ax.set_title("Components : {} - {}".format(comb[0],comb[1]))

from sklearn.decomposition import NMF
nmf = NMF(n_components=2)
nmf.fit(xtrain)
xprojected_nmf = nmf.transform(xtrain)
xprojtest_nmf = nmf.transform(xtest)

n [24]:
combs = list(combinations(range(len(nmf.components_)), 2))
n_line = ceil(len(combs)/3)
fig, axes = plt.subplots(n_line, 3, figsize=(18, 5*n_line))

for ax, comb in zip(axes.ravel(), combs):
    ax.scatter(xprojected_nmf[:,comb[0]], xprojected_nmf[:,comb[1]], c=ytrain, s=1)
    ax.set_title("Components : {} - {}".format(comb[0],comb[1]))

from sklearn.manifold import Isomap
ism = Isomap(n_components=2)
ism.fit(xtrain)
xprojected_ism = ism.transform(xtrain)
xprojtest_ism = ism.transform(xtest)


combs = list(combinations(range(ism.embedding_.shape[1]), 2))
n_line = ceil(len(combs)/3)
fig, axes = plt.subplots(n_line, 3, figsize=(18, 5*n_line))

for ax, comb in zip(axes.ravel(), combs):
    ax.scatter(xprojected_ism[:,comb[0]], xprojected_ism[:,comb[1]], c=ytrain, s=1)
    ax.set_title("Components : {} - {}".format(comb[0],comb[1]))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import timeit as tm

start_time = tm.default_timer()
gknn = GridSearchCV(KNeighborsClassifier(), {"n_neighbors":[1,3,5,7]}, cv=5)
gknn.fit(xprojected, ytrain)
time = tm.default_timer() - start_time

print(gknn.best_params_, time)


start_time = tm.default_timer()
ypred = gknn.predict_proba(xprojtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

start_time = tm.default_timer()
gknn2 = GridSearchCV(KNeighborsClassifier(), {"n_neighbors":[1,3,5,7]}, cv=5)
gknn2.fit(xtrain, ytrain)
time = tm.default_timer() - start_time

print(gknn2.best_params_, time)


start_time = tm.default_timer()
ypred = gknn2.predict_proba(xtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

from sklearn.model_selection import GridSearchCV

lr_gs = GridSearchCV(LogisticRegression(), {"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}, cv=5)
lr_gs.fit(xtrain, ytrain)
lr_gs.best_params_

start_time = tm.default_timer()
ypred = lr_gs.predict_proba(xtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

from sklearn.svm import SVC

svc_gs = GridSearchCV(SVC(), {"C":np.logspace(-3,3,10), "kernel":["linear","rbf"]}, cv=5)
svc_gs.fit(xtrain, ytrain)
svc_gs.best_params_

start_time = tm.default_timer()
ypred = svc_gs.predict_proba(xtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

from sklearn.ensemble import RandomForestClassifier
import timeit as tm

rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(xtrain, ytrain)

start_time = tm.default_timer()
ypred = rfc.predict_proba(xtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(rfc, prefit=True, threshold=0.005)
xtrain_features_selected = model.transform(xtrain)
xtrain_features_selected.shape

rfc2 = RandomForestClassifier(n_estimators=20)
rfc2.fit(xprojected_ism, ytrain)

start_time = tm.default_timer()
ypred = rfc2.predict_proba(xprojtest_ism)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)

rfc2 = RandomForestClassifier(n_estimators=20)
rfc2.fit(xprojected, ytrain)

start_time = tm.default_timer()
ypred = rfc2.predict_proba(xprojtest)[:,1]
time = tm.default_timer() - start_time

fpr, tpr, thr = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)
print(roc_auc, time)
