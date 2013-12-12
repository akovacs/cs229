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

print 'Loading dataframes from CSV'
stopwords_df = pd.read_csv("data/stopwords.txt", header=None, delimiter="\s+->\s+")
stopwords_df.columns=['word','index']
stopwords=set('word:'+stopwords_df['index'])

df = pd.read_table('data/new_chats_dataset.csv', sep=';', header=None, nrows=N+M+1)
df.columns=['chatid','user1','user2','profile1','profile2','start','end','disconnector','reporteduser','reportedreason','numlines1','numlines2','words1','words2','friends']
df_test = df[N+1:N+M+1]
df = df[:N]

profiles = pd.read_table('data/new_profiles_dataset.csv', sep=';', header=None)
profiles_columns=['profile','location','location_flag','age','gender','created','about','screenname']
for user in ('1','2'):
  profiles.columns = [col+user for col in profiles_columns]
  df = df.merge(profiles, on='profile'+user)
  df_test = df_test.merge(profiles, on='profile'+user)


hasher = FeatureHasher()
d = DictVectorizer()
tfidf = TfidfTransformer()
v = CountVectorizer()
v2 = CountVectorizer()
v3 = CountVectorizer()

# convert json string to dict and remove keys corresponding to stopwords
def stripStopWords(jsonString):
    wordsDict = json.loads(jsonString)
    for stopword in set(wordsDict.keys()).intersection(stopwords):
        del wordsDict[stopword]
    return wordsDict

def parse(df):
    print 'Parsing columns'
    #convert conversation to dict, filter out stopwords
    for i in ['words1', 'words2','about1','about2']:
        df.ix[:,i] = df.ix[:,i].apply(stripStopWords)
        #df.ix[:,i] = df.ix[:,i].apply(json.loads)
    df['about1_empty'] = (df.about1=={}).astype(int)
    df['about2_empty'] = (df.about2=={}).astype(int)
    for user in ('1','2'):
      #convert string gender to 0/1 field
      df.ix[:,'gender'+user] = (df.ix[:,'gender'+user]=='M').astype(int)
      #replace None ages with 0
      #TODO is 0 a good choice?
      df.ix[:,'age'+user][df.ix[:,'age'+user]=='None'] = 0
      df.ix[:,'age'+user] = df.ix[:,'age'+user].astype(int)
    df['gender_eq'] = (df.gender1==df.gender2).astype(int)
    df['age_diff'] = (df.age1.astype(int)-df.age2.astype(int)).abs()
    df['u1al'] = df.about1.apply(len)
    df['u2al'] = df.about2.apply(len)

    # compute average conversation length for each user: (sum numlines)/(count conversations)
    for user in ('1','2'):
      g = df.ix[:,['user'+user,'numlines'+user]].groupby('user'+user)
      avg_conv = g.sum().astype(float).div(g.count().ix[:,'numlines'+user], axis=0)
      avg_conv.columns = ['avg_conv'+user]
      df = df.join(avg_conv, on='user'+user)

    # compute number of words that users' about descriptions have in common
    df['about_shared'] = df[['about1','about2']][0:10].apply(lambda row: len(set(row['about1'].keys()).intersection(set(row['about2'].keys()))), axis=1)
    return df

def extract(df):
    print 'Extracting features'
    v.fit(df.ix[:,'user1'].values + df.ix[:,'user2'].values)
    v2.fit(df.ix[:,'profile1'].values + df.ix[:,'profile2'].values)
    Xs = (
        # TODO: 'float' object has no attribute 'lower'???
        hstack([v.transform(df.ix[:,'user1']),
            v.transform(df.ix[:,'user2']),
        ]),
        hstack([v2.transform(df.ix[:,'profile1']),
            v2.transform(df.ix[:,'profile2']),
        ]),
        df.ix[:,['numlines1','numlines2']].astype(int).values,
        # TODO: wait, I think this was off by 1 before?
        df.ix[:,['age1','gender1']].astype(int).values,
        df.ix[:,['age2','gender2']].astype(int).values,
        df.ix[:,['avg_conv1','avg_conv2']].values,
        df.ix[:,['about_empty1','about2_empty']].values,
        df.ix[:,['gender_eq', 'age_diff']].values,
        df.ix[:,['u1al', 'u2al']].values,
        hstack([d.fit_transform(df.ix[:,'words1']),
            d.fit_transform(df.ix[:,'words2']),
        ]),
        hstack([d.fit_transform(df.about1),
            d.fit_transform(df.about2),
        ]),
        tfidf.fit_transform(hstack([v3.fit_transform(df.location1),
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

# predict the friends 'f' field
df['u1'] = df['user1'].apply(lambda s: int(s.split(':')[1]))
df['u2'] = df['user2'].apply(lambda s: int(s.split(':')[1]))
df['p1'] = df['profile1'].apply(lambda s: int(s.split(':')[1]))
df['p2'] = df['profile2'].apply(lambda s: int(s.split(':')[1]))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['f'] = le.fit_transform(df['friends'])
dd=df.ix[:,['u1','u2','p1','p2','age','gender','location_flag',
    'u1al','age2','gender2','location_flag2','u2al','age_diff',
    'gender_eq', 'f']].astype(int)
#cross_validation.cross_val_score(clf, dd.ix[:,:14], dd.ix[:,14], cv=10,
        #scoring='f1', n_jobs=5)

# predict user quality
users = pd.read_table('data/new_users_dataset.csv', sep=';', header=None,
        index_col=0)
users.columns = ['quality']
dfq = df.join(users, on=1, how='inner')
dfq.quality = (dfq.quality=='clean').astype(int)
Xq, Xqs = extract(dfq)
gs.fit(Xq, dfq.quality.values)
print gs.best_score_
