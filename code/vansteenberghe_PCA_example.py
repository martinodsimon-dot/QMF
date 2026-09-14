#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Principal component analysis (PCA) of the daily returns of Euronext biotech stocks,
factor model and shock generation

This script accompanies Section "Principal Component Analysis" of the lecture notes
*Quantitative Methods in Finance* by Eric Vansteenberghe (arXiv:2601.12896,
SSRN 5178205), developed over more than ten years of teaching at Université Paris 1
Panthéon-Sorbonne (Master Finance, Technology & Data). The linear algebra it uses
(matrix product and transpose, covariance matrix of a data matrix, quadratic forms,
eigenvalues, spectral theorem, Lagrange multipliers) is collected in the companion
primer *Mathematics for Finance: A Self-Contained Primer*.

Data: daily closing prices downloaded from Euronext (one CSV file per company, in the
export format of the Euronext website: ';' separator, decimal comma, French column
names) for biotech and pharmaceutical companies listed on Euronext, in data/biotech/.

Pedagogical objectives:
- Build a panel of daily returns from many files and standardise it (centre-reduit)
- Compute the sample covariance matrix of the standardised returns (their correlation
  matrix) and diagonalise it: eigenvalues and eigenvectors, by hand with numpy.linalg
- Understand the principal components as the uncorrelated linear combinations of the
  returns with the largest variances (the eigenvectors are the weights, the
  eigenvalues are the variances), and check every identity of the notes numerically
- Compare with scikit-learn's PCA and decide how many components to keep (share of
  variance explained, scree plot, Kaiser criterion)
- Regress each stock on the first components: the coefficients are the loadings
- Simulate shocks on a portfolio from the factor model (normal factors and residuals)

Main topics covered:
- pandas: reading Euronext CSV files, concat, pct_change, standardisation, cov
- numpy.linalg.eigh, matrix products, reconstruction checks with numpy.allclose
- sklearn.decomposition.PCA, statsmodels OLS, Monte Carlo simulation

Intended audience:
- Economics and finance students who know the basics of pandas and linear algebra

Usage:
- Run cell by cell (the "# %%" markers are recognised by Spyder and VS Code) or as a
  script: python vansteenberghe_PCA_example.py
- The CSV files are expected in DATA_DIR (see the configuration block); set
  SAVE_FIGURES = True to export the figures of the lecture notes to FIG_DIR

File: vansteenberghe_PCA_example.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Folder containing the Euronext CSV files (one per company). The default assumes that
# python is run from the root of the QMF folder (the one containing code/ and data/);
# otherwise write the absolute path.
DATA_DIR = os.path.join("data", "biotech")

# Sample window and selection rule: we keep the companies whose price series covers at
# least MIN_COVERAGE of the trading days of the window. The window is the period common
# to the two Euronext downloads of the folder (September 2017 and January 2018).
START, END = "2013-01-17", "2017-09-06"
MIN_COVERAGE = 0.99

N_COMPONENTS_KEPT = 2       # components used in the factor model (F_1 and F_2)
N_SIMULATIONS = 20          # number of simulated shocks (S in the notes)
HOLDING_PER_STOCK = 1e6     # euros invested in each stock (A_i in the notes)
RANDOM_SEED = 0

SHOW_PLOTS = True      # display the figures on screen
SAVE_FIGURES = False   # export the figures of the lecture notes as PDF files
FIG_DIR = "fig"        # export folder, relative to the current working directory


def finish_figure(filename):
    """Export the current figure if SAVE_FIGURES, display it if SHOW_PLOTS, then close it.

    Parameters
    ----------
    filename : str
        Name of the PDF file written in FIG_DIR when SAVE_FIGURES is True.
    """
    if SAVE_FIGURES:
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def banner(title):
    """Print a title line, to separate the outputs of the script when run as a whole."""
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# Loading the Euronext files
# --------------------------------------------------------------------------- #
def locate_folder(folder=DATA_DIR):
    """Return the data folder, looking at `folder`, then one level up.

    The second attempt covers the case where python is run from the code/ folder rather
    than from the root of the repository.
    """
    for candidate in (folder, os.path.join("..", folder)):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"folder '{folder}' not found; set DATA_DIR to the folder "
                            "containing the Euronext CSV files")


def read_euronext_prices(path):
    """Read one Euronext CSV export and return the closing price as a Series indexed by date.

    The Euronext export uses ';' as separator, a decimal comma, dates written
    day/month/year, French column names ('Fermeture' is the closing price) and a
    trailing ';' that creates an empty last column. Every one of these quirks must be
    told to pandas; the series is named after the file (the company).
    """
    prices = pd.read_csv(path, sep=";", decimal=",", usecols=["Date", "Fermeture"])
    prices.index = pd.to_datetime(prices["Date"], format="%d/%m/%Y")
    series = prices["Fermeture"].astype(float).sort_index()
    series.name = os.path.splitext(os.path.basename(path))[0]
    return series


def build_price_panel(folder):
    """Read every CSV file of the folder into one DataFrame (dates x companies)."""
    folder = locate_folder(folder)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    series = [read_euronext_prices(os.path.join(folder, f)) for f in files]
    return pd.concat(series, axis=1, sort=True)


def select_complete_stocks(prices, start, end, min_coverage=MIN_COVERAGE):
    """Restrict the panel to the window and to the stocks with an almost complete history.

    Returns
    -------
    pandas.DataFrame
        Prices over [start, end] for the selected stocks, without any missing value
        (the few remaining holes, non-trading days for one stock, are dropped).
    """
    window = prices.loc[start:end].dropna(how="all")
    coverage = window.notna().mean()
    kept = coverage.index[coverage >= min_coverage]
    return window[kept].dropna(how="any")


# --------------------------------------------------------------------------- #
# PCA by hand
# --------------------------------------------------------------------------- #
def standardise(returns):
    """Centre and reduce each column: (R - mean) / std, the R^cr of the notes.

    Returns
    -------
    standardised : pandas.DataFrame
        Columns with mean 0 and standard deviation 1 (the matrix X of the notes).
    means, stds : pandas.Series
        The statistics used, needed to convert simulated standardised returns back to
        returns.
    """
    means, stds = returns.mean(), returns.std(ddof=1)
    return (returns - means) / stds, means, stds


def pca_by_hand(X):
    """Principal components of the columns of X from the eigendecomposition of X'X/(T-1).

    With X the T x N matrix of standardised returns, the sample covariance matrix
    Sigma = X'X/(T-1) is symmetric positive semidefinite. Its eigenvectors q_1, ..., q_N
    (orthonormal) are the weights of the principal components, its eigenvalues
    lambda_1 >= ... >= lambda_N their variances, and the components (scores) are the
    columns of F = X Q.

    The sign of an eigenvector is arbitrary (q and -q solve the same problem): we use
    the convention of scikit-learn, the entry of largest absolute value of each
    eigenvector is made positive, so that both computations can be compared exactly.

    Returns
    -------
    eigenvalues : pandas.Series
        lambda_1 >= ... >= lambda_N, indexed 'PC1', ..., 'PCN'.
    loadings : pandas.DataFrame
        The matrix Q (stocks in rows, components in columns).
    scores : pandas.DataFrame
        The matrix F = X Q (dates in rows, components in columns).
    """
    Sigma = X.cov(ddof=1)                                   # N x N, symmetric
    eigenvalues, Q = np.linalg.eigh(Sigma.values)           # eigh: symmetric matrices
    order = np.argsort(eigenvalues)[::-1]                   # eigh returns ascending
    eigenvalues, Q = eigenvalues[order], Q[:, order]
    signs = np.sign(Q[np.argmax(np.abs(Q), axis=0), np.arange(Q.shape[1])])
    Q = Q * signs
    names = [f"PC{j + 1}" for j in range(len(eigenvalues))]
    eigenvalues = pd.Series(eigenvalues, index=names, name="eigenvalue")
    loadings = pd.DataFrame(Q, index=X.columns, columns=names)
    scores = pd.DataFrame(X.values @ Q, index=X.index, columns=names)
    return eigenvalues, loadings, scores


def explained_variance_table(eigenvalues):
    """Share of the total variance explained by each component, and cumulated share."""
    share = eigenvalues / eigenvalues.sum()
    return pd.DataFrame({"eigenvalue": eigenvalues, "share": share,
                         "cumulated share": share.cumsum()})


def factor_regression(X, scores, n_components=N_COMPONENTS_KEPT):
    """Regress each standardised return on the first components, without intercept.

    R^cr_{i,t} = alpha_i F_{1,t} + beta_i F_{2,t} + epsilon_{i,t} (equation of the notes).
    The components are orthogonal, so the OLS coefficients are the loadings q_{i1}, q_{i2}
    exactly, and R^2_i = (lambda_1 q_{i1}^2 + lambda_2 q_{i2}^2) / Var(R^cr_i).

    Returns
    -------
    pandas.DataFrame
        One row per stock: the coefficients, the residual standard deviation
        sigma_i^epsilon and the R^2 of the regression.
    """
    factors = scores.iloc[:, :n_components]
    rows = {}
    for stock in X.columns:
        fit = sm.OLS(X[stock], factors).fit()
        rows[stock] = {**{f"coef {c}": fit.params[c] for c in factors.columns},
                       "sigma_eps": np.sqrt(fit.ssr / (fit.nobs - n_components)),
                       "R2": fit.rsquared}
    return pd.DataFrame(rows).T


def simulate_shocks(eigenvalues, regression, means, stds, n_simulations=N_SIMULATIONS,
                    holding=HOLDING_PER_STOCK, seed=RANDOM_SEED):
    """Simulate returns from the factor model and the resulting portfolio gains and losses.

    The four steps of the notes, for s = 1, ..., n_simulations:
    1. draw the factors F^s_j ~ N(0, lambda_j) (the variance of component j is its
       eigenvalue) and the residuals epsilon^s_i ~ N(0, sigma_i^epsilon);
    2. standardised return of each stock: R^cr,s_i = alpha_i F^s_1 + beta_i F^s_2 + eps^s_i;
    3. actual return: R^s_i = R^cr,s_i * std_i + mean_i;
    4. portfolio shock: sum_i R^s_i A_i, with A_i the amount held in stock i.

    Returns
    -------
    returns : pandas.DataFrame
        Simulated returns (simulations in rows, stocks in columns).
    portfolio : pandas.Series
        Simulated gain or loss of the portfolio, in euros.
    """
    rng = np.random.default_rng(seed)
    coef_columns = [c for c in regression.columns if c.startswith("coef ")]
    n_components = len(coef_columns)
    factors = rng.normal(0.0, np.sqrt(eigenvalues.iloc[:n_components].values),
                         size=(n_simulations, n_components))
    residuals = rng.normal(0.0, regression["sigma_eps"].values.astype(float),
                           size=(n_simulations, len(regression)))
    standardised = factors @ regression[coef_columns].values.T.astype(float) + residuals
    returns = pd.DataFrame(standardised * stds.values + means.values,
                           columns=regression.index, index=range(1, n_simulations + 1))
    portfolio = (returns * holding).sum(axis=1)
    return returns, portfolio


# %% ------------------------------------------------------------------------ #
# 1. Prices, returns and standardisation
# --------------------------------------------------------------------------- #
banner("1. Data")
prices_all = build_price_panel(DATA_DIR)
print(f"{prices_all.shape[1]} companies read, {prices_all.index.min().date()} to "
      f"{prices_all.index.max().date()}")
prices = select_complete_stocks(prices_all, START, END)
print(f"kept {prices.shape[1]} stocks with a complete history over {START} - {END}, "
      f"{len(prices)} trading days:")
print(", ".join(prices.columns))

returns = prices.pct_change().dropna(how="any")             # R_{i,t} = P_t / P_{t-1} - 1
X, means, stds = standardise(returns)                       # R^cr_{i,t}
T, N = X.shape
print(f"\nT = {T} daily returns, N = {N} stocks; standardised returns: mean "
      f"{X.mean().abs().max():.1e}, std {X.std().mean():.3f}")

# %% ------------------------------------------------------------------------ #
# 2. The covariance matrix of the standardised returns (their correlation matrix)
# --------------------------------------------------------------------------- #
banner("2. Covariance matrix")
Sigma = X.cov(ddof=1)
assert np.allclose(Sigma.values, X.values.T @ X.values / (T - 1))   # Sigma = X'X/(T-1)
assert np.allclose(Sigma.values, Sigma.values.T)                     # symmetric
assert np.allclose(np.diag(Sigma.values), 1.0)                       # standardised data
off_diagonal = Sigma.values[np.triu_indices(N, k=1)]
print(f"trace(Sigma) = {np.trace(Sigma.values):.1f} = N; mean pairwise correlation "
      f"{off_diagonal.mean():.3f} (min {off_diagonal.min():.3f}, max {off_diagonal.max():.3f})")

# %% ------------------------------------------------------------------------ #
# 3. PCA by hand: eigenvalues, eigenvectors, scores
# --------------------------------------------------------------------------- #
banner("3. PCA by hand")
eigenvalues, Q, F = pca_by_hand(X)
# Every identity of the notes, checked numerically
assert np.allclose(Q.values.T @ Q.values, np.eye(N))                 # Q'Q = I
assert np.allclose(Q.values @ np.diag(eigenvalues) @ Q.values.T, Sigma.values)  # spectral
assert np.allclose(F.cov(ddof=1).values, np.diag(eigenvalues))        # Var(F) = Lambda
assert np.isclose(eigenvalues.sum(), N)                               # total variance
print("the components are uncorrelated and their variances are the eigenvalues")
table = explained_variance_table(eigenvalues)
print(table.head(6).round(3))
print(f"the first component explains {table.loc['PC1', 'share']:.1%} of the variance, "
      f"the first two {table.loc['PC2', 'cumulated share']:.1%}")
print("\nweights of the first component (all positive: a sector factor):")
print(Q["PC1"].sort_values().round(3).to_string())
# The first component is close to the average standardised return of the sector
equal_weighted = X.mean(axis=1)
print(f"\ncorrelation between PC1 and the equal-weighted average of the standardised "
      f"returns: {np.corrcoef(F['PC1'], equal_weighted)[0, 1]:.3f}")

# %% ------------------------------------------------------------------------ #
# 4. Same thing with scikit-learn
# --------------------------------------------------------------------------- #
banner("4. scikit-learn PCA")
pca = PCA(n_components=N).fit(X.values)
assert np.allclose(pca.explained_variance_, eigenvalues.values)      # same eigenvalues
assert np.allclose(pca.components_.T, Q.values)                       # same eigenvectors
assert np.allclose(pca.transform(X.values), F.values)                 # same scores
print("scikit-learn returns the same eigenvalues (explained_variance_), the same "
      "eigenvectors (components_, one per row) and the same scores (transform)")

# %% ------------------------------------------------------------------------ #
# 5. How many components? Scree plot, Kaiser criterion, share of variance
# --------------------------------------------------------------------------- #
banner("5. How many components to keep")
n_kaiser = int((eigenvalues > 1).sum())
print(f"Kaiser criterion (eigenvalue > 1, more variance than one standardised stock): "
      f"{n_kaiser} components")
print("rule of thumb of the notes (one component if it explains more than 70%): "
      f"PC1 explains {table.loc['PC1', 'share']:.1%}")
n_for_70 = int((table["cumulated share"] < 0.70).sum() + 1)
print(f"components needed to explain 70% of the variance: {n_for_70} of {N}")

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
left.bar(range(1, N + 1), eigenvalues.values, color="0.6")
left.axhline(1, color="black", linestyle="--", linewidth=1, label="Kaiser threshold")
left.set_xlabel("principal component")
left.set_ylabel("eigenvalue (variance of the component)")
left.set_title("scree plot")
left.legend(frameon=False)
right.plot(range(1, N + 1), table["cumulated share"], "k.-")
right.axhline(0.70, color="black", linestyle="--", linewidth=1, label="70% of the variance")
right.set_xlabel("number of components kept")
right.set_ylabel("cumulated share of the variance")
right.set_ylim(0, 1)
right.legend(frameon=False, loc="lower right")
plt.tight_layout()
finish_figure("PCA_variance.pdf")

plt.figure(figsize=(6.5, 5))
Q["PC1"].sort_values().plot.barh(color="0.6")
plt.xlabel("weight in the first principal component")
plt.tight_layout()
finish_figure("PCA_loadings.pdf")

# %% ------------------------------------------------------------------------ #
# 6. Factor model: regress each stock on the first two components
# --------------------------------------------------------------------------- #
banner("6. Factor regressions")
regression = factor_regression(X, F, N_COMPONENTS_KEPT)
assert np.allclose(regression["coef PC1"].astype(float), Q["PC1"])   # coefficients =
assert np.allclose(regression["coef PC2"].astype(float), Q["PC2"])   # loadings
print("the OLS coefficients equal the weights (loadings) of the components")
print(regression.round(3).sort_values("R2", ascending=False).to_string())
cumulated = table.loc[f"PC{N_COMPONENTS_KEPT}", "cumulated share"]
print(f"\naverage R2 with {N_COMPONENTS_KEPT} components: {regression['R2'].mean():.3f} "
      f"(= share of variance explained by them, {cumulated:.3f})")

# %% ------------------------------------------------------------------------ #
# 7. Shock generation under the normality assumption
# --------------------------------------------------------------------------- #
banner("7. Shock generation")
simulated_returns, portfolio = simulate_shocks(eigenvalues, regression, means, stds)
print(f"portfolio: {N} stocks, {HOLDING_PER_STOCK:,.0f} euros each; {N_SIMULATIONS} "
      "simulated daily shocks (euros):")
print(portfolio.round(0).to_string())
print(f"mean {portfolio.mean():,.0f}, standard deviation {portfolio.std():,.0f}, "
      f"worst {portfolio.min():,.0f}")

# A larger number of simulations gives the distribution of the portfolio shock, from
# which a Value-at-Risk can be read (section on risk measures of the notes)
_, portfolio_large = simulate_shocks(eigenvalues, regression, means, stds,
                                     n_simulations=10_000)
var_99 = -portfolio_large.quantile(0.01)
print(f"\nwith 10,000 simulations: 1% quantile of the daily portfolio shock "
      f"{-var_99:,.0f} euros, i.e. a one-day 99% Value-at-Risk of {var_99:,.0f} euros "
      f"({var_99 / (N * HOLDING_PER_STOCK):.2%} of the portfolio)")

plt.figure(figsize=(6.5, 3.8))
plt.hist(portfolio_large / 1e3, bins=60, color="0.6", density=True)
plt.axvline(-var_99 / 1e3, color="black", linestyle="--", label="1% quantile (99% VaR)")
plt.xlabel("simulated one-day portfolio gain or loss (thousand euros)")
plt.ylabel("density")
plt.legend(frameon=False)
finish_figure("PCA_portfolio_shocks.pdf")

# %% ------------------------------------------------------------------------ #
# 8. Exercise: PCA on the EBA stress test scenarios (see the notes)
# --------------------------------------------------------------------------- #
# Apply the functions above to the macro-financial scenarios of the EBA stress tests:
# build a data matrix with one row per date and one column per variable (or per
# country), standardise, diagonalise the covariance matrix, and find the most important
# factor behind the scenarios. The pipeline is the same: standardise -> cov -> eigh.
