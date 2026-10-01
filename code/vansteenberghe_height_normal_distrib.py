#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Python: Weight and height distributions (normality tests, Q-Q plots, distribution fitting)

This script accompanies Section "Python: Weight and height distributions" of the lecture
notes *Quantitative Methods in Finance* by Eric Vansteenberghe (arXiv:2601.12896,
SSRN 5178205), developed over more than ten years of teaching at Université Paris 1
Panthéon-Sorbonne (Master Finance, Technology & Data).

Heights look normally distributed, but what about weights? The script confronts the normal
law with two anthropometric variables from the Demographic and Health Surveys: women's
height, which is close to normal, and women's weight, which is right-skewed. It then tests
normality formally and searches for a better-fitting distribution.

Pedagogical objectives:
- Describe a sample before modelling it: summary statistics, histograms, outliers, boxplot
- Recognise a right-skewed distribution (mean above the median) and reduce skewness with logs
- Test H0: the sample is i.i.d. N(mu, sigma^2), and see what each test compares with the
  normal law: the whole CDF, the skewness and kurtosis, or the frequencies in bins
- Simulate the critical values of the test statistics under H0 by Monte Carlo
- Select a parametric distribution with Q-Q plots and Kolmogorov-Smirnov p-values
- Illustrate the Central Limit Theorem with the means of samples of heights

Main topics covered:
- Pearson correlation test (rho = 0 does not imply independence), one-sample t-test
- Normality tests: Jarque-Bera, Kolmogorov-Smirnov (one- and two-sample), Lilliefors,
  Anderson-Darling, chi-square goodness of fit with equiprobable bins
- Q-Q plots against the normal, logistic, exponential, log-normal, Weibull and Pareto laws
- Automatic search over the scipy.stats distributions (contributed by João H. Dimas);
  the Fitter package (optional)
- Central Limit Theorem; bivariate kernel density estimate of f(weight, height)

Data (not distributed in this repository):
- data/replication_final.dta: Demographic and Health Surveys (DHS) microdata, collected
  with the help of Professor Dean Spears. One row per birth, pooling the DHS of India
  (2005-06, NFHS-3) and of 25 sub-Saharan African countries (2003-2010).
- Variables used: v002 household number, v012 age of the woman, v024 region (state in
  India), v437 her weight in 0.1 kg, v438 her height in 0.1 cm (i.e. in mm). After
  dropping duplicates and implausible values, the lecture notes use n = 15,494 women.
- The DHS Program terms of use do not allow the microdata to be redistributed: request
  access at https://dhsprogram.com/data/, download the surveys in Stata format and save
  the five variables above as data/replication_final.dta. With another extract, the
  results will differ from those reported in the lecture notes.

Intended audience:
- Master students in economics and finance (statistics and econometrics)

Usage:
- Run cell by cell (the "#%%" markers are recognised by Spyder and VS Code) or as a
  script: python vansteenberghe_height_normal_distrib.py. The working directory is set to
  the repository root: the first folder, going up from this script's folder, with data/.
- Set ploton = True to draw the figures and export those of the lecture notes to fig/.
- Set fitteruse = True to run the Fitter package (pip install fitter).
- Requirements: numpy, pandas (with jinja2, needed by DataFrame.to_latex), scipy,
  statsmodels and matplotlib.

Related research by the author:
- Skewness as an economic signal: Vansteenberghe, E. (2024). Uncertain and Asymmetric
  Forecasts. arXiv:2411.05938 — builds an asymmetry indicator from the median and the
  skewness of subjective probability distributions.

How to cite:
- Vansteenberghe, E. (2026). Quantitative Methods in Finance. arXiv:2601.12896.
  https://doi.org/10.48550/arXiv.2601.12896

File: vansteenberghe_height_normal_distrib.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""

import os
# We set the working directory
os.chdir('/Users/skimeur/Mon Drive/QMF')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  # module namespace: t-test and loop over the scipy.stats distributions
from scipy import stats
from scipy.stats import (gaussian_kde, skew, kurtosis, pearsonr,
                         pareto, lognorm, weibull_min, expon, logistic)
import statsmodels.api as sm  # for Q-Q plots
from statsmodels.stats.diagnostic import lilliefors

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ploton = False  # True: draw the figures and export those of the lecture notes to fig/

# work from the repository root: the first folder, from this script's folder upwards,
# that contains data/ (or replace ROOT by the path of your own folder)
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "data").is_dir()), HERE)
os.chdir(ROOT)
if ploton:
    os.makedirs("fig", exist_ok=True)

DATA_FILE = os.path.join("data", "replication_final.dta")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{ROOT / DATA_FILE} not found: the DHS microdata cannot be "
                            "redistributed, see 'Data' in the header of this script.")

df = pd.read_stata(DATA_FILE, convert_categoricals=False,
                   columns=["v002", "v012", "v024", "v437", "v438"])

# v002 household number
# v012 current age (respondent)
# v024 region (state in India)
# v437 women's weight in .1 of kg
# v438 women's height in .1 of cm (i.e. in mm)

statdesc = df.loc[:,["v012","v437","v438"]].describe().round(1)
statdesclatex = statdesc.to_latex()

ages = list(df.v012.unique())
ages.sort()


# keep only unique observations
df = df.drop_duplicates(subset=["v002","v012"])

if ploton:
    df.v437.hist(bins=50) # v437 women's weight in .1 of kg
    df.v438.hist(bins=50)

# we seem to have unexpected outliers
# in this study, we are not focusing on the tail hence we drop outliers
df = df.loc[(df.v437<1600)&(df.v438<2000)&(df.v438>1200),:]

if ploton:
    df.v437.hist(bins=50) # weight
    df.v438.hist(bins=50) # height
    
# intuition of a right skewness with a boxplot showing the median and the mean
if ploton:
    df.boxplot(column=['v437'], sym='', showmeans=True)
    
#%% Are the rv statiscially independent?

# H_0: rho = 0 (no linear association)
# H_1: rho != 0

r, p_value = pearsonr(
    df["v437"], df["v438"], alternative="two-sided"
)

print(f"Pearson correlation: r = {r:.3f}")
print(f"p-value: {p_value:.3e}")

# Non-rejection does not establish statistical independence:
# nonlinear dependence may exist even when rho = 0.



#%% Weight: a right-skewed distribution

if ploton:
    ax = df.v437.hist(bins=50, density=True)
    fig = ax.get_figure()
    fig.savefig('fig/Indian_weight.pdf')

# take the log of weights: the log function reduces skewness
df['logv437'] = np.log(df.v437)
print("Skewness of the weights", df.v437.skew())
print("Skewness of log of the weights", df.logv437.skew())

# visual inspection: are the log weights normally distributed?
# probability density function of a normal distribution
def pdfnormal(x, mui, sigmai):
    return (1 / np.sqrt((2*np.pi*sigmai**2))) * np.exp(-((x-mui)**2) / (2*sigmai**2))

xres = 0.0001  # resolution
# define the x-axis range based on the log weights
xgrid = np.arange(df.logv437.min(), df.logv437.max(), xres)
# normal distribution N(mui,sigmai)
pdf = pdfnormal(xgrid, df.logv437.mean(), df.logv437.std())

# plot the probability density
if ploton:
    plt.hist(df.logv437, bins=100, density=True)
    plt.plot(xgrid, pdf, 'b', label="Normal PDF")
    plt.xlabel("log weight")
    plt.ylabel("Probability of occurence")
    plt.savefig('fig/Indian_logweight.pdf')
    plt.show()
    plt.close()


#%% Is the mean what we think it is? (a bit useless here, just for illustration)
# T-test: is the mean return statistically different from mu?
# Two-sided test
# H0: the expected value of this sample made of (presumably) independent observations is equal to mu=df.logv437.mean()
scipy.stats.stats.ttest_1samp(df.logv437, df.logv437.mean())


#%% Normality tests: common set-up
# H0: the sample is i.i.d. N(mu, sigma^2), with mu and sigma^2 unknown
# the tests differ in what they compare with the normal law:
#   the whole CDF: 1) Kolmogorov-Smirnov, 2) Lilliefors, 3) Anderson-Darling
#   skewness and kurtosis: 4) Jarque-Bera
#   frequencies in bins: 5) chi-square goodness of fit
# we run each test on the weights, then all of them on the weights, log weights and heights

rng = np.random.default_rng(2026)  # reproducible random draws
x = df.v437.to_numpy()  # weights, in .1 of kg
n = len(x)
mu_hat, sigma_hat = x.mean(), x.std(ddof=1)

#%% 1) Jarque-Bera test: skewness and kurtosis
# JB = n/6 S^2 + n/24 (K-3)^2, asymptotically chi2 with 2 degrees of freedom under H0
# NB: kurtosis() returns the EXCESS kurtosis K-3 by default, hence fisher=False
JBstat = (n / 6) * skew(x)**2 + (n / 24) * (kurtosis(x, fisher=False) - 3)**2

# compare it to a chi 2 at 95% with two degrees of freedom
if JBstat > stats.chi2.ppf(q=0.95, df=2):
    print('we reject normality')
else:
    print('normality of the distribution not rejected')

print(stats.jarque_bera(x))  # same statistic, with its p-value
del JBstat


#%% 2) Kolmogorov-Smirnov (KS) test
# one-sample KS, H0: the sample comes from a FULLY SPECIFIED continuous distribution F0
# statistic: D_n = max_x |F_n(x) - F0(x)|, with F_n the empirical CDF
# kstest(x, stats.norm.cdf) would not be correct: it tests the STANDARD normal N(0,1)
# plugging in the estimated mu and sigma, the p-value is too large (F0 is fitted to the data):
# the Lilliefors test (2) corrects this
print(stats.kstest(x, 'norm'))

# two-sample KS, H0: both samples come from the same distribution
# compare the weights with a sample drawn from the fitted normal distribution
sampletheoretical = rng.normal(mu_hat, sigma_hat, n)
print(stats.ks_2samp(x, sampletheoretical))
# we reject the null hypothesis (but this is against one randomly generated sample, think about MC:
# for the heights, only about half of the draws reject at the 1% level)

# show the cumulative distribution functions
if ploton:
    fig, ax = plt.subplots(ncols=1)
    ax1 = ax.twinx()
    ax.hist(x, 100, alpha=0.2, cumulative=True, color='blue')
    ax1.hist(x, 100, alpha=0.2, density=True, color='blue')
    ax.hist(sampletheoretical, 100, alpha=0.2, cumulative=True, color='red')
    ax1.hist(sampletheoretical, 100, alpha=0.2, density=True, color='red')
    plt.title('PDF and CDF, normal law in red')
    fig.savefig('fig/cdf_hist_cacnrom.pdf')

if ploton:
    fig, ax = plt.subplots(ncols=1)
    ax.hist(x, 100, alpha=0.2, cumulative=True, color='blue')
    ax.hist(sampletheoretical, 100, alpha=0.2, cumulative=True, color='red')
    plt.title('CDF, normal law in red')
    fig.savefig('fig/cdf_cacnrom.pdf')


#%% 3) Lilliefors test: KS with estimated mean and variance
print(lilliefors(x, dist='norm'))  # (statistic, p-value)
# we reject the null hypothesis, our sample doesn't come from a normally distributed population


#%% 4) Anderson-Darling test: more weight on deviations in the tails
print(stats.anderson(x, dist='norm'))  # reject H0 at level alpha if statistic > critical value
# (with SciPy >= 1.17, add method='interpolate' to also get a p-value)
# KS tests a fixed distribution, Lilliefors adapts KS to normality with estimated μ,σ,
# and Anderson–Darling also uses estimated μ,σ but gives more weight to tail deviations.


#%% 5) Chi-square goodness-of-fit test
# nb_bins bins, equiprobable under the fitted normal: each bin expects n/nb_bins observations
nb_bins = 50
bin_edges = stats.norm.ppf(np.linspace(0, 1, nb_bins + 1), loc=mu_hat, scale=sigma_hat)
obs_counts = np.histogram(x, bins=bin_edges)[0]
exp_counts = np.full(nb_bins, n / nb_bins)

# degrees of freedom: nb_bins - 1 - 2, as we estimated mu and sigma
chi2, p = stats.chisquare(f_obs=obs_counts, f_exp=exp_counts, ddof=2)
print("chi2:", chi2, "p-value:", p)

# ---- table for inspection/plot: each bin is located at its median under the fitted normal ----
dfchisq = pd.DataFrame({
    'bin_mid': stats.norm.ppf((np.arange(nb_bins) + 0.5) / nb_bins, loc=mu_hat, scale=sigma_hat),
    'observed': obs_counts,
    'expected': exp_counts
})

# ---- plot ----
if ploton:
    ax = dfchisq.set_index('bin_mid')[['observed','expected']].plot()
    ax.set_xlabel('weight (bin median under the fitted normal)')
    ax.set_ylabel('count')
    ax.set_title('Chi-square GOF: observed vs expected')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('fig/chisquaretest.pdf')


#%% Summary: the tests on the weights, the log weights and the heights
def normality_stats(sample, nb_bins=50):
    """Statistics of the normality tests, H0: the sample is drawn from a normal distribution."""
    s = np.asarray(sample, dtype=float)
    m, mu, sigma = len(s), s.mean(), s.std(ddof=1)
    edges = stats.norm.ppf(np.linspace(0, 1, nb_bins + 1), loc=mu, scale=sigma)
    return pd.Series({
        'D_n (Kolmogorov-Smirnov, Lilliefors)': stats.kstest(s, 'norm').statistic,
        'A^2 (Anderson-Darling)': stats.anderson(s, dist='norm').statistic,
        'JB (Jarque-Bera)': stats.jarque_bera(s).statistic,
        'chi2 (goodness of fit, 50 bins)': stats.chisquare(np.histogram(s, bins=edges)[0],
                                                           np.full(nb_bins, m / nb_bins)).statistic,
    })

summary = pd.DataFrame({'weight': normality_stats(df.v437),
                        'log weight': normality_stats(df.logv437),
                        'height': normality_stats(df.v438)})

# 5% critical values: as Lilliefors did, we simulate the statistics under H0 (think about MC)
# under H0 they do not depend on mu and sigma, so we can draw from N(0,1)
sims = pd.DataFrame([normality_stats(rng.standard_normal(len(df))) for _ in range(1000)])
summary['5% critical value'] = sims.quantile(0.95)
print(summary.round(4))

# with a fully specified F0, the KS critical value would be larger: the naive KS test is too lenient
print("5% KS critical value for a fully specified F0:", stats.kstwobign.ppf(0.95) / np.sqrt(len(df)))
# asymptotic critical values of JB and chi2: compare with the simulated ones
print("chi2(2) and chi2(47) 95% quantiles:", stats.chi2.ppf(0.95, 2), stats.chi2.ppf(0.95, 47))


#%% Q-Q plots

# normal law
if ploton:
    sm.qqplot((df.v437-np.mean(df.v437))/np.std(df.v437), line='45')
    
# logistic distribution on the log
if ploton:
    sm.qqplot(df.logv437, dist=logistic, line='r')

# exponential law
if ploton:
    sm.qqplot(df.loc[df.v437>df.v437.median(),'v437'], line='r', dist=expon)

# log-normal law
# you have to indicate the shape parameter in the distargs arguments
if ploton:
    shapelgn = lognorm.fit(df.v437)[0]
    sm.qqplot(df.v437, line='r', dist=lognorm, distargs=(shapelgn, ))

    
# Weibull law
# nota bene: in the sparams, you need to choose the shape parameter
if ploton:
    c = weibull_min.fit(df.v437)[0]
    sm.qqplot(np.log(df.v437), dist=weibull_min, distargs=(c, ), line='r')


# Pareto law
# nota bene: in the sparams, you need to define the scale parameter, here it is u=median
if ploton:
    sm.qqplot(np.log(df.v437), dist=pareto, distargs=(df.v437.median(), ), line='r')


#%% Fitter approach
fitteruse = False
if fitteruse:
    from fitter import Fitter
    f = Fitter(df.v437)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    f.summary()
    # exponnorm has a good outcome


#%% Exercise: try to find a better fit with another law (this is alreadty don in the Fitter appraoch)
    
# work of João H. Dimas
# This code tests all probability distributions from scipy.stats and finds the best fit to rcac.
# It was ran the first time to automatically add the ones with errors to ignoreList. Then, I inserted them directly into the code to save processing time. 
# The ignored are the distributions that requires different parameters, so they can't be tested automatically.
# The list of distributions to ignore is larger than the ones that work. IOne could just iterate the working ones, but the code shows the original idea.
ignoreList = ['ksone','alpha','beta','betaprime','bradford','burr','burr12','fisk','chi','chi2','cosine','dgamma','dweibull','exponnorm','exponweib','exponpow','fatiguelife','foldcauchy','f','foldnorm','frechet_r','weibull_min','frechet_l','weibull_max','genlogistic','genpareto','genexpon','genextreme','gamma','erlang','gengamma','genhalflogistic','gompertz','gausshyper','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','levy_stable','loggamma','loglaplace','lognorm','mielke','kappa4','kappa3','nakagami','ncx2','ncf','t','nct','pareto','lomax','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rice','recipinvgauss','semicircular','skewnorm','trapz','triang','truncexpon','truncnorm','tukeylambda','vonmises','vonmises_line','wrapcauchy','gennorm','halfgennorm','argus','kstwobign']

maxPValue = 0  
bestDist = None
bestSample = None  

for attr, value in scipy.stats.__dict__.items():
    try:
        if not attr in ignoreList and hasattr(value, "fit") and hasattr(value, "rvs") and hasattr(value, "name"):
            localMaxPValue = 0
            dist = getattr(scipy.stats, attr)
            print("Testing distribution: {}".format(dist.name))
            loc, scale = dist.fit(df.logv437)  
            
            # Since the sample is small, we run 100 times to find the highest p-value (ideally we would store the p-values, and analyse their distribution to choose the best fitting).
            for x in range(100):
                newseries = dist.rvs(loc=loc,scale=scale,size=len(df))
                statistic, pvalue = scipy.stats.ks_2samp(df.logv437,newseries)
                if pvalue > maxPValue:
                    maxPValue = pvalue
                    bestDist = dist
                    bestSample = newseries
                if pvalue > localMaxPValue:
                    localMaxPValue = pvalue
            print("Best p-value for {}: {:.4f}".format(dist.name, localMaxPValue))
    except Exception as ex:
        print("Error with distribution {}".format(attr))
        ignoreList.append(attr)
            
print("\nBest distribution: {}\nMax. p-value: {:.4f}".format(bestDist.name, maxPValue)) 
# Winner is Hyperbolic Secant, with second place going to Laplace distribution.
# Laplace was one first guess because of the tent shape.

# Get the "best" found sample and plot it
df['bestSample'] = bestSample

if ploton:
    plt.hist(df.logv437, 100, alpha=0.5, density=True, label='Weight distribution')
    plt.hist(df.bestSample, 100, alpha=0.5, density=True, label='Logistic distribution')
    plt.legend(loc='upper right')
    plt.title('Weight distribution versus {} PDF'.format(bestDist.name))
    plt.savefig('fig/distrib_fit_loop.pdf')
    plt.show()

# list the content of a library:
#scipy.stats.__dict__.items()

del ignoreList, maxPValue, bestDist, bestSample, loc, scale, newseries, statistic, pvalue, localMaxPValue, attr, x




#%% Height

if ploton:
    ax = df.v438.hist(bins=50, density=True)
    fig = ax.get_figure()
    fig.savefig('fig/Indian_height.pdf')

# are the weights normally distributed?
# Kolmogorov-Smirnov two-sided test
stats.ks_2samp(df.v438, np.random.normal(df.v438.mean(), df.v438.std(), len(df)))
# we do not always reject the null hypothesis at the 1 percent threshold
# Lilliefors test
lilliefors(df.v438, dist='norm') # we reject the null hypothesis, 
# our sample doesn't come from a normally distributed population

# CLT exercise

# now we consider that X is our population
# we will create nrs random samples from this population with nb individuals
nrs = 10**4
nb = 10**2
samples = []
for x in range(nrs):
    samples.append(np.random.choice(df.v438, size=nb))

samplesMeans = [np.mean(s) for s in samples]

# Lilliefors test for normality
lilliefors(samplesMeans, dist='norm')

if ploton:
    plt.hist(samplesMeans, bins=200, density=True, label="Samples means")
    plt.title("Central Limit Theorem")
    plt.legend(loc='upper right')
    plt.show()
    

barZ = [(s-np.mean(df.v438))/np.math.sqrt(np.std(df.v438, ddof=1)**2/nb) for s in samplesMeans]

# Lilliefors test for normality
lilliefors(barZ, dist='norm')

if ploton:
    plt.hist(barZ, bins=200, density=True, label="Samples means")
    plt.title("Central Limit Theorem")
    plt.legend(loc='upper right')
    plt.show()

#%% Non-parametric kernel denity plot
# Nonparametric density estimation of f(x, y)

# Define x and y as the variables to plot
x = df['v437']  # Women's weight in .1 of kg
y = df['v438']  # Women's height in cm

# Create a 2D grid over which to evaluate the KDE
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

# Stack the grid to use as input for KDE
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)

# Evaluate the KDE on the grid
f = np.reshape(kernel(positions).T, xx.shape)

# Plot the results
if ploton:
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(xx, yy, f, levels=50, cmap='viridis')
    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel('Density')
    ax.set_xlabel('Weight (v437)')
    ax.set_ylabel('Height (v438)')
    ax.set_title('Nonparametric Density Plot of f(Weight, Height)')
    plt.savefig('fig/nonparametric_density_f_xy.pdf')
    plt.show()


# 3D Plot of f(x, y)
if ploton:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(xx, yy, f, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Add labels and title
    ax.set_xlabel('Weight (v437)')
    ax.set_ylabel('Height (v438)')
    ax.set_zlabel('Density')
    ax.set_title('3D Nonparametric Density Plot of f(Weight, Height)')
    
    # Optional: Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=30)
    
    
    
    