#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

Playing Perudo: what are your odds?

@author: Eric Vansteenberghe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# number of dices
ndices = 30

# how many random draw do you want in your Monte Carlo
tirages = 10**6

# create the draws
experience = np.random.randint(1, 7, size=(ndices,tirages))

# extract the first ten columns for illustration purpose
extracttoshow = experience[:,:10]

# count how many 2 you have in your extract
# divide this by the length of your extract
np.sum(extracttoshow == 2) / 10

# do the same on the full sample to be more precise
# you expect on average to have 5 twos among 30 dices
np.sum(experience == 2) / tirages

# count how many aces and 2's you got per experiment
countnum = np.count_nonzero(experience == 1, axis=0) + np.count_nonzero(experience == 2, axis=0)

# what is the mean of count
# you expect on average to have 5 twos among 30 dices and 5 aces among 30 dices
countnum.mean()

# chance for a calza
plt.hist(countnum, bins=list(range(1,ndices-5)), density=True)

# empirical density P(countnum = k) for k = 1..30, printed as a table
vals = np.bincount(countnum, minlength=31)   # counts for k = 0..30
k = np.arange(1, 31)
empirical_density = 100 *vals[1:31] / tirages

table = pd.DataFrame({"k": k, "empirical_density": empirical_density})
print(table.to_string(index=False, float_format="{:.6f}".format))


# chance for a dudo
dudo = []

for i in range(1,ndices+1):
    dudo.append((countnum < i).sum())

# convert it in odds
dudo = [x / tirages for x in dudo]

# convert it to a data frame
dudo = pd.DataFrame(dudo, index=range(1, ndices+1))
# change column name and plot
dudo.columns = ['dudo chance']
dudo.plot()

# focus on the range [7-15]
dudo.loc[7:15,:].plot()
