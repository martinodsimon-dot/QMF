#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Quantile-Quantile (Q-Q) plots: by hand and with statsmodels

This script accompanies Subsection "Q-Q plots and probability plots" of Section "Python:
Weight and height distributions" of the lecture notes *Quantitative Methods in Finance* by
Eric Vansteenberghe (arXiv:2601.12896, SSRN 5178205), developed over more than ten years of
teaching at Université Paris 1 Panthéon-Sorbonne (Master Finance, Technology & Data).

A Q-Q plot draws the ordered sample x_(1) <= ... <= x_(n) against the quantiles
Q(p) = F^{-1}(p) of a theoretical distribution F, at the probabilities p_i = i/(n+1). If
the points lie close to the 45-degree line, the data are consistent with F.

Pedagogical objectives:
- Understand what a Q-Q plot shows before calling a library function
- Build a Q-Q plot step by step for n = 10 draws from a standard normal law: sort the
  sample, compute the theoretical quantiles with the inverse CDF (norm.ppf), and plot the
  empirical quantiles against the theoretical ones
- See why the probabilities i/n lose the last point (norm.ppf(1) = inf) and i/(n+1) do not

Main topics covered:
- Empirical and theoretical quantiles, inverse cumulative distribution function
- statsmodels.api.qqplot with a 45-degree line; scipy.stats.probplot (probability plots)
- Reproducible random draws with numpy.random.default_rng

Intended audience:
- Economics and finance students with an introductory statistics background

Usage:
- Run cell by cell (the "#%%" markers are recognised by Spyder and VS Code) or as a
  script: python vansteenberghe_qqplots.py
- Set ploton = True to draw the figures
- Q-Q plots of real data (women's weights) against several laws:
  vansteenberghe_height_normal_distrib.py

How to cite:
- Vansteenberghe, E. (2026). Quantitative Methods in Finance. arXiv:2601.12896.
  https://doi.org/10.48550/arXiv.2601.12896

File: vansteenberghe_qqplots.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  # for Q-Q plots
from scipy.stats import norm
from scipy.stats import probplot  # noqa: F401  probability plots: probplot(y, plot=plt)

ploton = False  # True: draw the figures

#%% QQ plots

# number of generated points
Ni = 10
# generate points from a standard normal law
# use a Random Generator so we always have the same samples generated
y = np.random.default_rng(2022).normal(0, 1, Ni)

# plot a historgram
if ploton:
    plt.hist(y)

# Scipy function for Q-Q plot
# against the normal law
# normal law
if ploton:
    sm.qqplot(y, line='45')


# Manual Q-Q plot
# 1) sort values of y
y.sort()
# 2) we have Ni observations
# 2.1
print("for the first observations, the empirical value is", y[0])
print("below this value, we expect the have 10% of the population")
print("if the population follows a standard normal law, the threshold below which 10% of the population can be found is")
print(norm.ppf(.1))
#2.2
print("for the second observations, the empirical value is", y[1])
print("below this value, we expect the have 20% of the population")
print("if the population follows a standard normal law, the threshold below which 20% of the population can be found is")
print(norm.ppf(.2))
#2.3
print("for the third observations, the empirical value is", y[2])
print("below this value, we expect the have 20% of the population")
print("if the population follows a standard normal law, the threshold below which 30% of the population can be found is")
print(norm.ppf(.3))
#2.10
print("for the last and 10th observations, the empirical value is", y[9])
print("below this value, we expect the have 100% of the population")
print("if the population follows a standard normal law, the threshold below which 100% of the population can be found is")
print(norm.ppf(1))
# 3) generate the theoretical quantiles
x = []
for i in range(0,Ni):
    x.append(norm.ppf((i+1)/Ni))
    # to avoid loosing the last point, we can use Ni+1 at the denominator
    #x.append(norm.ppf((i+1)/(Ni+1)))
    
# 4) plot empirical against theoretical quantiles
if ploton:
    plt.plot(x, y, 'r.') # x vs y
    plt.plot(np.arange(-3,3,6/Ni),np.arange(-3,3,6/Ni),'k-') # identity line
    plt.xlabel('theoretical quantiles')
    plt.ylabel('empirical quantiles')
    plt.show()    

