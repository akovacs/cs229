#!/usr/bin/python
# Arpad Kovacs <akovacs@stanford.edu>
# Daniel Velkov <dvelkov@stanford.edu>
# Aditya Somani <asomani@stanford.edu>
# CS229 Final Project - Feature Importer

#from collections import Counter
#from pylab import *
#import random

import pandas as pd
from sklearn.feature_extraction import *
from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from scipy.sparse import hstack

N=300000 # Training dataset size
M=30000  # Test dataset size

df = pd.read_table('data/new_chats_dataset.csv', sep=';', header=None, nrows=N+M+1)
df.columns=['chatid','user1','user2','profile1','profile2','start','end','disconnector','reporteduser','reportedreason','numlines1','numlines2','words1','words2']
df_test = df[N+1:N+M+1]
df = df[:N]

profiles = pd.read_table('data/new_profiles_dataset.csv', sep=';', header=None)
profiles_columns=['profile','location','location_flag','age','gender','created','about','screenname']
profiles.columns = [col + '1' for col in profiles_columns]
df = df.merge(profiles, on='profile1')
df_test = df_test.join(profiles, on='profile1')
profiles.columns = [col + '2' for col in profiles_columns]
df = df.merge(profiles, on='profile2')
df_test = df_test.join(profiles, on='profile2')


hasher = FeatureHasher()
d = DictVectorizer()
tfidf = TfidfTransformer()
v = CountVectorizer()
v2 = CountVectorizer()
v3 = CountVectorizer()

def parse(df):
    #convert conversation to dict
    for i in [12,13,'about','about2']:
        df.ix[:,i] = df.ix[:,i].apply(json.loads)
    df['about_empty'] = (df.about=={}).astype(int)
    df['about2_empty'] = (df.about2=={}).astype(int)
    #convert string gender to 0/1 field
    df.ix[:,18] = (df.ix[:,18]=='M').astype(int)
    df.ix[:,25] = (df.ix[:,25]=='M').astype(int)
    #replace None ages with 0
    #TODO is 0 a good choice?
    df.ix[:,17][df.ix[:,17]=='None'] = 0
    df.ix[:,24][df.ix[:,24]=='None'] = 0
    df.ix[:,17] = df.ix[:,17].astype(int)
    df.ix[:,24] = df.ix[:,24].astype(int)
    df['gender_eq'] = (df.gender==df.gender2).astype(int)
    df['age_diff'] = (df.age.astype(int)-df.age2.astype(int)).abs()
    df['u1al'] = df.about.apply(len)
    df['u2al'] = df.about2.apply(len)

    g = df.ix[:,[1,10]].groupby(1)
    avg_conv = g.sum().astype(float).div(g.count().ix[:,10], axis=0)
    avg_conv.columns = ['avg_conv']
    df = df.join(avg_conv, on=1)
    g = df.ix[:,[2,11]].groupby(2)
    avg_conv = g.sum().astype(float).div(g.count().ix[:,11], axis=0)
    avg_conv.columns = ['avg_conv2']
    df = df.join(avg_conv, on=2)

    return df

def extract(df):
    v.fit(df.ix[:,1].values + df.ix[:,2].values)
    v2.fit(df.ix[:,3].values + df.ix[:,4].values)
    Xs = (
        hstack([v.transform(df.ix[:,1]),
            v.transform(df.ix[:,2]),
        ]),
        hstack([v2.transform(df.ix[:,3]),
            v2.transform(df.ix[:,4]),
        ]),
        df.ix[:,[10,11]].astype(int).values,
        df.ix[:,[17,18]].astype(int).values,
        df.ix[:,[24,25]].astype(int).values,
        df.ix[:,['avg_conv','avg_conv2']].values,
        df.ix[:,['about_empty','about2_empty']].values,
        df.ix[:,['gender_eq', 'age_diff']].values,
        df.ix[:,['u1al', 'u2al']].values,
        hstack([d.fit_transform(df.ix[:,12]),
            d.fit_transform(df.ix[:,13]),
        ]),
        hstack([d.fit_transform(df.about),
            d.fit_transform(df.about2),
        ]),
        tfidf.fit_transform(hstack([v3.fit_transform(df.location),
            v3.fit_transform(df.location2)
        ])),
        )
    X = hstack(Xs)
    return X, Xs

def response(df):
    return (df.ix[:,8]!='null').values

df = parse(df)
X, Xs = extract(df)
y_train = response(df)

df_test = parse(df_test)
Xt, Xts = extract(df_test)
y_test = response(df_test)

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
pp = {'alpha': 0.01,
'loss': 'perceptron',
'n_iter': 20,
'penalty': 'l2',
'power_t': 0.5,
'class_weight': 'auto'}
clf = SGDClassifier(**pp)

params={'penalty': ['l1', 'l2'], 'n_iter': [200, 20], 'loss': ['hinge', 'log', 'perceptron'],
    'alpha': [1e-1, 1e-3, 1e-5, 1e-7, 1e-9], 'power_t': [0.3, 0.5, 0.7],
    'class_weight': ['auto']}
gs = RandomizedSearchCV(clf, params, cv=5, scoring='f1', n_jobs=8, n_iter=100,
        verbose=1)
from sklearn import cross_validation
# predict reports
#print cross_validation.cross_val_score(clf, Xs[0], y_train, cv=10,
        #scoring='f1', verbose=2, n_jobs=8)

# predict the 'f' field
df['u1'] = df[1].apply(lambda s: int(s.split(':')[1]))
df['u2'] = df[2].apply(lambda s: int(s.split(':')[1]))
df['p1'] = df[3].apply(lambda s: int(s.split(':')[1]))
df['p2'] = df[4].apply(lambda s: int(s.split(':')[1]))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['f'] = le.fit_transform(df[14])
dd=df.ix[:,['u1','u2','p1','p2','age','gender','location_flag',
    'u1al','age2','gender2','location_flag2','u2al','age_diff',
    'gender_eq','f']].astype(int)
#cross_validation.cross_val_score(clf, dd.ix[:,:14], dd.ix[:,14], cv=10,
        #scoring='f1', n_jobs=5)

# predict user quality
users = pd.read_table('tmp/new_users_dataset.csv', sep=';', header=None,
        index_col=0)
users.columns = ['quality']
dfq = df.join(users, on=1, how='inner')
dfq.quality = (dfq.quality=='clean').astype(int)
Xq, Xqs = extract(dfq)
gs.fit(Xq, dfq.quality.values)
print gs.best_score_
