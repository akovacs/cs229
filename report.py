import pandas as pd
from sklearn.feature_extraction import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

N=2000000
M=200000
df = pd.read_table('chats', sep=';', header=None, nrows=N+M+1)
df_test = df[N+1:N+M]
df = df[:N]
p = pd.read_table('profiles_dataset.csv', sep=';', index_col=0)
df = df.join(p, on=3)
df_test = df_test.join(p, on=3)
p.columns = [c + '2' for c in p.columns]
df = df.join(p, on=4)
df_test = df_test.join(p, on=4)

hasher = FeatureHasher()
d = DictVectorizer()
v = CountVectorizer()
v2 = CountVectorizer()

from scipy.sparse import hstack, vstack
import json
v.fit(df.ix[:,1].values + df.ix[:,2].values)
v2.fit(df.ix[:,3].values + df.ix[:,4].values)
#convert conversation to dicts
df.ix[:,12] = df.ix[:,12].map(json.loads)
df.ix[:,13] = df.ix[:,13].map(json.loads)
#convert string gender to 0/1 field
df.ix[:,18] = (df.ix[:,18]=='M').astype(int)
df.ix[:,25] = (df.ix[:,25]=='M').astype(int)
#replace None ages with 0
#TODO is 0 a good choice?
df.ix[:,17][df.ix[:,17]=='None'] = 0
df.ix[:,24][df.ix[:,24]=='None'] = 0
df.ix[:,17] = df.ix[:,17].astype(int)
df.ix[:,24] = df.ix[:,24].astype(int)
df['gender_eq'] = (df.gender==df.gender2)
df['age_diff'] = (df.age.astype(int)-df.age2.astype(int)).abs()

X_train = hstack([v.transform(df.ix[:,1]),
v.transform(df.ix[:,2]),
v2.transform(df.ix[:,3]),
v2.transform(df.ix[:,4]),
df.ix[:,[10,11,17,18,24,25]].astype(int).values,
d.fit_transform(df.ix[:,12]),
d.fit_transform(df.ix[:,13])])

X_train2 = hstack([X_train, df.ix[:,['gender_eq']].astype(int),
df.ix[:,['age_diff']].astype(int)])
#X_train = hasher.transform(df.ix[:,12])
y_train = (df.ix[:,8]!='null').values
#clf.fit(X_train, y_train)
df_test.ix[:,12] = df_test.ix[:,12].map(json.loads)
df_test.ix[:,13] = df_test.ix[:,13].map(json.loads)
df_test.ix[:,18] = df_test.ix[:,18]=='M'
df_test.ix[:,17][df_test.ix[:,17]=='None'] = 0
X_test = hstack([v.transform(df_test.ix[:,1]),
v.transform(df_test.ix[:,2]),
v2.transform(df_test.ix[:,3]),
v2.transform(df_test.ix[:,4]),
df_test.ix[:,[10,11,17,18]].astype(int).values,
hasher.transform(df_test.ix[:,12]),
hasher.transform(df_test.ix[:,13])])
#X_test = hasher.transform(df_test.ix[:,12])
y_test = (df_test.ix[:,8]!='null').values
#yp = clf.predict(X_test)
#print 'precision: ', precision_score(y_test, yp)
#print 'recall: ', recall_score(y_test, yp)

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
params = {'penalty': ['l1', 'l2'],
'n_iter': [20],
'alpha': [1, 0.1, 0.01, 0.001],
'loss': ['log', 'perceptron'],
}

pp = {'alpha': 0.01,
'loss': 'perceptron',
'n_iter': 20,
'penalty': 'l2',
'power_t': 0.5}

clf = SGDClassifier(**pp)
gs = RandomizedSearchCV(clf, params, cv=5, scoring='f1', verbose=2, n_jobs=5,
n_iter=20)
X_train3 = hstack([v.transform(df.ix[:,1]),
v.transform(df.ix[:,2]),])
from sklearn import cross_validation
