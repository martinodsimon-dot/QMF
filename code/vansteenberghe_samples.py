#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF 2022
Population, samples and absolutely continuous distributions
Maximum Likelihood Estimation

@author: Eric Vansteenberghe

Population, samples: distribution and Central Limit Theorem
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
import scipy.stats as scst
import pandas as pd
from scipy.stats import genextreme
from scipy.optimize import minimize

# To plot, set ploton to True
ploton = False

# We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

#%% I believe the past is representative of the future

# Import historical CAC 40 data
df = pd.read_excel("data/ihc20181130.xlsx", sheet_name="CAC 40®", skiprows=3)
df = df.rename(columns = {'Unnamed: 0':'date'})
df.index = df['date']
df = pd.DataFrame(df['clôture close.1'])
df.columns = ['CAC 40 net return close']
# An index is suppose to be >0, so filter
df = df.loc[df['CAC 40 net return close']>0]

if ploton:
    df.plot()

# compute the daily returns
dx = (df - df.shift(1)) / df.shift(1)
dx = dx.dropna()

if ploton:
    ax = dx.plot()
    fig = ax.get_figure()
    #fig.savefig('cacnetret.pdf')

# Count how many returns are positive and negative
# create an empty data frame
histodx = pd.DataFrame(index=['count'],columns=['neg','pos'])
# store the counts in the data frame
histodx.loc['count','neg'] = dx.loc[dx['CAC 40 net return close']<=0,:].count().values
histodx.loc['count','pos'] = dx.loc[dx['CAC 40 net return close']>0,:].count().values
# convert the data frame to latex to export in the report
histolatex = histodx.to_latex()

# now divide the count by the total number of observations to have an empirical probability
histodx.loc['count','neg'] = dx.loc[dx['CAC 40 net return close']<=0,:].count().values / dx.count().values
histodx.loc['count','pos'] = dx.loc[dx['CAC 40 net return close']>0,:].count().values / dx.count().values
histolatex = histodx.to_latex()

# probability of a daily return of 4.5%?
if ploton:
    dx.hist(bins=20,density=True)

# probability to have daily return falling inside the bucket [0.04,0.05]
100 * dx.loc[(dx['CAC 40 net return close']<=0.05)&(dx['CAC 40 net return close']>=0.04),:].count().values / dx.count().values


#%% Normal distribution

# or use the CAC 40 return data to chose the mean and the standard deviation
mui = dx.mean().values
sigmai = dx.std().values

# probability density function of a normal distribution
def pdfnormal(x, mui, sigmai):
    return (1 / np.sqrt((2*np.pi*sigmai**2))) * np.exp(-((x-mui)**2) / (2*sigmai**2))

xres = 0.0001  # resolution
# define the x-axis range based on CAC 40 data
x = np.arange(dx.min().values[0] /2, dx.max().values[0]/2, xres)
# normal distribution N(mui,sigmai)
pdf = pdfnormal(x, mui, sigmai)

# plot the probability density
if ploton:
    plt.plot(x, pdf, 'b', label="Normal PDF")
    plt.xlabel("Variable value")
    plt.ylabel("Probability of occurence")
    plt.savefig('fig/normal_pdf.pdf')
    plt.show()
    plt.close()

# generate random draws from this distribution
# size of the sample
N = 10**5
X = np.random.normal(mui, sigmai, size=N)
# mean of our random draws
X.mean()  
# standard deviation of our random draws  
X.std()

# our population is uniformly distributed between 0 and 1
if ploton:
    plt.plot(x, pdf, 'r', label="Normal PDF")
    plt.hist(X, bins=100, density=True)
    plt.savefig('fig/normal_pdf_hist.pdf')

# use the CAC 40 empirical data
if ploton:
    plt.plot(x, pdf, 'r', label="Normal PDF")
    plt.hist(dx.values, bins=50, density=True)
    plt.savefig('fig/empirical_normal_CAC.pdf')

     
# Maximum-likelihood estimate of Normal(mu, sigma)

def loglikeNL(coef, seriei):
    mu = coef[0]
    sigma = coef[1]

    # Prevent undefined likelihood at sigma = 0
    if sigma <= 0:
        return np.inf

    n = len(seriei)
    return (
        (n / 2) * np.log(2 * np.pi * sigma**2)
        + np.sum((seriei - mu)**2) / (2 * sigma**2)
    )

thetastart = np.array([1.0, 1.0])

# Fit simulated/theoretical data
MLAR1 = minimize(loglikeNL, thetastart, args=(X,), method="Nelder-Mead")

print("estimated theta", MLAR1.x)
print("true variables", [np.mean(X), np.std(X)])
print("success of ML estimation", MLAR1.success)

# Fit empirical data
MLAR1 = minimize(loglikeNL, thetastart, args=(dx.values,), method="Nelder-Mead")

print("estimated theta", MLAR1.x)
print("success of ML estimation", MLAR1.success)


#%% fit a normal distribution on the CAC 40 returns
mui, stdi = scst.norm.fit(dx.values)
print('our ML estimates', MLAR1.x, 'scst estimates', mui, stdi)

# generate random draw from the normal law
dxnorm = pd.DataFrame(np.random.normal(mui, stdi, size=10**8))
dxnorm.columns = ['normal']

# probability to have return inside the bucket [0.04,0.05]
print( 100 * dxnorm.loc[(dxnorm['normal']<=0.05)&(dxnorm['normal']>=0.04),:].count().values / dxnorm.count().values)
# which is almost equivalent to integrating the pdf from 0.04 to 0.05 around the value at 0.045:
print(100 * 0.01 * scst.norm.pdf(0.045, loc=mui, scale=stdi))

if ploton:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dx.plot(kind='kde', ax=ax)
    dxnorm.plot(kind='kde', ax=ax, color='red')

#%%  cumulative distribution function

xres = 0.0001  # resolution
# define the x-axis range based on CAC 40 data
x = np.arange(dx.min().values[0]/2, dx.max().values[0]/2, xres)
# normal law cdf
cdf = scst.norm.cdf(x, loc=mui, scale=stdi)

# use the CAC 40 empirical data
if ploton:
    plt.plot(x, cdf, 'r', label="Normal CDF")
    plt.hist(dx.values, bins=50, density=True, cumulative=True)
    plt.savefig('fig/empirical_normal_CAC_cdf.pdf')

# empirical "probability" that the daily return is lower or equal to 4.5 %
100 * dx.loc[(dx['CAC 40 net return close']<=0.045),:].count().values / dx.count().values
# probability to have return lower or equal to 4.5 %
100 * dxnorm.loc[(dxnorm['normal']<=0.045),:].count().values / dxnorm.count().values
# cdf of the normal law at 4.5%
scst.norm.cdf(0.045, loc=mui, scale=stdi)

#%% Expectation of the absolute values of a random variable normaly distributed
    
# chose values for the mean and standard deviation
mui = 0
sigmai = 3.26

# generate random draws from this distribution
# size of the sample
N = 10**7
X = np.random.normal(mui, sigmai, size=N)
Xabs = np.abs(X)

print('mean of random variable', np.mean(X))
print('mean of absolute values of random variable', np.mean(Xabs))
print('expected value of absolute values of random variable', sigmai * np.sqrt(2/np.pi))

del cdf, dxnorm, histodx, histolatex, mui, N, pdf, sigmai, stdi, thetastart, x, X, Xabs, xres

#%% Introduction to exponential distribution

# Focus on the negative returns
alphai = -( 1 / dx.loc[dx['CAC 40 net return close']<0,:].mean().values )

# probability density function of an exponential distribution
def pdfexp(x, alphai):
    return alphai * np.exp(-alphai * x)

xres = 0.0001  # resolution
# define the x-axis range based on CAC 40 data
x = np.arange(-dx.loc[dx['CAC 40 net return close']<0,:].max().values[0], -dx.loc[dx['CAC 40 net return close']<0,:].min().values[0], xres)
# normal distribution N(mui,sigmai)
pdfexp = pdfexp(x, alphai)

# plot the probability density
if ploton:
    plt.plot(x, pdfexp, 'b', label="Exponential PDF")
    plt.xlabel("Variable value")
    plt.ylabel("Probability of occurence")
    plt.savefig('exp_pdf.pdf')
    plt.show()
    plt.close()

# use the CAC 40 empirical data
if ploton:
    plt.plot(x, pdfexp, 'r', label="Exponential PDF")
    plt.hist(-dx.loc[dx['CAC 40 net return close']<0,:].values, bins=50, density=True)
    plt.savefig('empirical_exponential_CAC.pdf')

del alphai, pdfexp, x, xres

#%% Application
# first chose your alpha
alphai = 1.6
N = 10
sample = np.random.exponential(scale=1/alphai, size=N)
if ploton:
    plt.hist(sample)

# apply our MLE to find alpha
alphahat = 1/np.mean(sample)

#%% Generalized extreme value distribution, standardized

xilist = list(range(0,10))


# probability density function of a Generalized extreme value distribution, standardized (mu = 0 and sigma = 1)
def pdfsGEV(x,xi):
    if xi == 0:
        return np.exp(-np.exp(-x)-x)      
    else:
        return ((1 + xi*x)**(-1-1/xi)) * np.exp( -(1+xi*x)**(-1/xi) )

if ploton:
    # we want to plot the normal distribution N(0,1)
    jet= plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0,1,10)))
    dx = 0.001  # resolution
    x = np.arange(0, 5, dx)  
    for xi in xilist:
        pdfi = pdfsGEV(x,xi)
        plt.axis("tight")
        plt.plot(x, pdfi, 'g', label="Standardized GEV PDF, xi = %.0f"%xi,color=next(colors))
    plt.legend(loc="best")
    plt.xlabel("Return value")
    plt.ylabel("Probability of occurence")
    plt.show()
    plt.close()


if ploton:
    # generate random draws from this distribution
    # size of the sample
    Nlist = list(10**np.array([0,1,2,3,4,5,6,7,8]))
    # NB: scipy notation, c = - xi
    xi = 1.5
    c = - xi
    meanlist = []
    for Ni in Nlist:
        X = genextreme.rvs(c, size=Ni)
        meanlist.append(np.mean(X))
    plt.loglog(Nlist,meanlist,label="Mean as a function of sample size")
    plt.legend(loc="best")
    plt.xlabel("Sample size")
    plt.ylabel("Sample mean")
    plt.savefig('fig/sGEVmean.pdf')
    plt.show()
    plt.close()



#%% Gamma distribution

# create the gamma distribution
def gammadist(alphai,ri,x):
    return ( alphai**ri / gamma(ri) ) * x**(ri-1)*np.exp(-alphai*x)

# set the parameters
alphai = 100
rlist = [0.5,1,1.5,3]
# create the x-axis
xlist = np.arange(0.001,0.1,0.001)

# store the results as a function of the shape parameters
rlist = pd.DataFrame(columns=rlist,index=xlist)
for ri in rlist:
    rlist.loc[:,ri] = gammadist(alphai,ri,xlist)

if ploton:
    ax = rlist.plot(title="Gamma distribution as a function of shape parameter, rate param = 100")
    fig = ax.get_figure()
    fig.savefig("fig/gammadistillu.pdf")

# now fix gamma and vary the rate para
ri = 1.5
alphalist = [0.5,1,10,100,200]

# store the results as a function of the rate parameters
alphalist = pd.DataFrame(columns=alphalist,index=xlist)
for alphai in alphalist:
    alphalist.loc[:,alphai] = gammadist(alphai,ri,xlist)

if ploton:
    ax = alphalist.plot(title="Gamma distribution as a function of rate parameter, shape param = 1.5")
    fig = ax.get_figure()
    fig.savefig("fig/gammadistillu2.pdf")

# Generate a population from a gamma distribution
# size of sample
N = 10**5
ri = 3
alphai = 15
X = np.random.gamma(scale = 1/alphai, shape = ri, size = N)
if ploton:
    plt.hist(X,bins=100,normed=True)
    
print('mean of random variable', np.mean(X))
print('expected value', ri / alphai)

    
print('variance of random variable', np.var(X))
print('expected value', ri / alphai**2)

