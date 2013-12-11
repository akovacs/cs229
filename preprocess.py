#!/usr/bin/python
# Arpad Kovacs <akovacs@stanford.edu>
# CS224W Final Project - Feature Importer

#from collections import Counter
#from pylab import *
#import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

import sqlite3 as sql
import sanetime

import itertools

# First 10,000 rows from features_f01_rearranged_with_s_converted.csv
# head -n 10000 features_f01_rearranged_with_s_converted.csv > truncated.csv
data = pd.read_table('truncated.csv', sep=',', index_col=0)

#data['agediffAtLeast17'] = (data['agediff'] < 17).astype(int)
def bucketAgeDiff(agediff):
    if agediff < 5:
        return 0
    elif agediff < 10:
        return 1
    elif agediff < 15:
        return 2
    else:
        return 3

data['agebucket'] = data['agediff'].apply(bucketAgeDiff)
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

data.to_csv('agebucket.csv')

