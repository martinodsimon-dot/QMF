#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Naive Bayes classifiers: from Bayes' rule to a Gaussian naive Bayes classifier of the
sex of a worker from the wage, the age and the education

This script accompanies Section "Naive Bayes Classifiers" of the lecture notes
*Quantitative Methods in Finance* by Eric Vansteenberghe (arXiv:2601.12896,
SSRN 5178205), developed over more than ten years of teaching at Université Paris 1
Panthéon-Sorbonne (Master Finance, Technology & Data). The probability tools it uses
(conditional probability, law of total probability, Bayes' theorem, conditional
independence, the Gaussian density) are collected in the companion primer
*Mathematics for Finance: A Self-Contained Primer*.

Data: the sample of 534 workers used for the wage equation of the lecture notes
(Section "Multivariate regressions and tests (wages)"), in the spirit of Mincer (1974),
Schooling, Experience and Earnings:
- http://www.economicswebinstitute.org/data/wagesmicrodata.xls
We keep the sex, the hourly wage (in log), the age and the years of education.

Pedagogical objectives:
- State the question as a comparison of posterior probabilities,
  P(male | wage, age, education) versus P(female | wage, age, education)
- Turn it around with Bayes' rule: posterior = prior x likelihood / evidence
- See why the joint likelihood cannot be estimated cell by cell (curse of
  dimensionality) and what the "naive" conditional-independence assumption buys
- Estimate the class-conditional Gaussian densities, classify one observation by hand,
  then the whole sample: confusion matrix, accuracy, precision, recall, F1
- Compare with scikit-learn's GaussianNB and with the trivial classifier
- Exercise: drop the naive assumption with a multivariate Gaussian likelihood
  (quadratic discriminant analysis) — a skeleton is provided, the solution is at the end

Main topics covered:
- pandas groupby statistics by class, the Gaussian density written by hand
- Bayes' rule with densities, maximum a posteriori classification
- Confusion matrix and the usual classification metrics
- numpy.linalg (determinant, inverse) for the multivariate Gaussian density

Intended audience:
- Economics and finance students who know the basics of pandas and probability

Usage:
- Run cell by cell (the "# %%" markers are recognised by Spyder and VS Code) or as a
  script: python ML_wage_naive_bayes.py
- The data file wagesmicrodata.xls is expected in DATA_DIR (see the configuration
  block); reading .xls files requires the xlrd package (pip install xlrd)
- Set RUN_SOLUTION = True to run the solution of the exercise of the last cell

File: ML_wage_naive_bayes.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""

import os
# We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Folder containing wagesmicrodata.xls. The default assumes that python is run from the
# root of the QMF folder (the one containing code/ and data/); otherwise write the
# absolute path (see the section "Indicating where your file is located" of the notes).
DATA_DIR = "data"
DATA_FILE = "wagesmicrodata.xls"

FEATURES = ["WAGE", "AGE", "EDUCATION"]   # the observed variables (W, A, E in the notes)
TARGET = "SEX"                            # the class to predict: 1 = male, 0 = female
TARGET_ROW = 6                            # the observation classified by hand in the notes

SHOW_PLOTS = True      # display the figures on screen
SAVE_FIGURES = False   # export the figures of the lecture notes as PDF files
FIG_DIR = "fig"        # export folder, relative to the current working directory
RUN_SOLUTION = False   # run the solution of the exercise (try it yourself first!)


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
# Loading
# --------------------------------------------------------------------------- #
def locate(filename, data_dir=DATA_DIR):
    """Return the path of a data file, looking in data_dir, then one level up.

    The second attempt covers the case where python is run from the code/ folder rather
    than from the root of the repository.
    """
    for candidate in (os.path.join(data_dir, filename),
                      os.path.join("..", data_dir, filename)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"{filename} not found in '{data_dir}' or '../{data_dir}'. Set DATA_DIR to the "
        "folder containing the wage data.")


def load_wage_data(filename=DATA_FILE):
    """Read the wage micro-data and return the columns SEX, WAGE (log), AGE, EDUCATION.

    Two quirks of the file must be known (open it in Excel first!): the first row of the
    'Data' sheet holds the description of each variable, not data, and the sex is coded
    1 for a woman and 0 for a man. We drop the description row, recode the sex as
    1 = male (so that the positive class of the confusion matrix is 'male', as in the
    notes) and take the logarithm of the hourly wage, as in the wage equation of the
    notes.

    Returns
    -------
    pandas.DataFrame
        534 rows, columns SEX (1 = male), WAGE (log of the hourly wage in dollars),
        AGE (years) and EDUCATION (years of schooling).
    """
    raw = pd.read_excel(locate(filename), sheet_name="Data", header=0, index_col=0)
    df = raw.iloc[1:].reset_index(drop=True).astype(float)   # drop the description row
    df["SEX"] = (1 - df["SEX"]).astype(int)                    # 1 = male
    df["WAGE"] = np.log(df["WAGE"])                            # log hourly wage
    return df[[TARGET] + FEATURES]


# --------------------------------------------------------------------------- #
# Naive Bayes with Gaussian likelihoods
# --------------------------------------------------------------------------- #
def gaussian_pdf(x, mu, sigma):
    """Density of the normal distribution N(mu, sigma^2) at x (the formula of the notes).

    Parameters
    ----------
    x : float or array-like
        Point(s) at which the density is evaluated.
    mu, sigma : float
        Mean and standard deviation.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def class_statistics(df, features=FEATURES, target=TARGET):
    """Prior probability, mean and standard deviation of each feature, for each class.

    The naive Bayes classifier has only these parameters: for each class s and each
    feature, one mean and one standard deviation (the sample standard deviation, with
    n - 1 in the denominator; scikit-learn uses n, which changes the third decimal).

    Returns
    -------
    priors : pandas.Series
        P(S = s), the share of each class in the sample, indexed by class.
    means, stds : pandas.DataFrame
        One row per class, one column per feature.
    """
    priors = df[target].value_counts(normalize=True).sort_index()
    means = df.groupby(target)[features].mean()
    stds = df.groupby(target)[features].std(ddof=1)
    return priors, means, stds


def naive_bayes_posterior(X, priors, means, stds):
    """Posterior probability of each class under the naive Gaussian assumption.

    For each observation x = (x_1, ..., x_p) and each class s,
        numerator_s(x) = P(S = s) * prod_j f(x_j | S = s),
    where f(. | S = s) is the N(mean_{s,j}, std_{s,j}^2) density: the product replaces
    the joint density of the features by the product of the one-dimensional densities
    (conditional independence given the class). The posterior is
        P(S = s | x) = numerator_s(x) / sum_{s'} numerator_{s'}(x),
    and the denominator (the evidence) is the same for every class: it only rescales.

    Parameters
    ----------
    X : pandas.DataFrame
        Observations, one column per feature (same columns as `means`).
    priors, means, stds
        Output of class_statistics.

    Returns
    -------
    pandas.DataFrame
        One column per class with the posterior probabilities (rows sum to one).
    """
    numerators = pd.DataFrame(index=X.index, columns=priors.index, dtype=float)
    for s in priors.index:
        likelihood = np.ones(len(X))
        for feature in means.columns:
            density = gaussian_pdf(X[feature], means.loc[s, feature], stds.loc[s, feature])
            likelihood = likelihood * density
        numerators[s] = priors[s] * likelihood
    return numerators.div(numerators.sum(axis=1), axis=0)


def predict_map(posterior):
    """Maximum a posteriori decision: the class with the largest posterior probability."""
    return posterior.idxmax(axis=1).astype(int)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred):
    """2x2 confusion matrix with the actual class in rows and the predicted class in columns.

    Returns
    -------
    pandas.DataFrame
        Rows 'Actual 0', 'Actual 1'; columns 'Predicted 0', 'Predicted 1'.
    """
    table = pd.crosstab(y_true, y_pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    table.index = ["Actual 0", "Actual 1"]
    table.columns = ["Predicted 0", "Predicted 1"]
    return table


def classification_metrics(cm):
    """Accuracy, precision, recall and F1 score from a 2x2 confusion matrix.

    With the positive class 1 (male): TP = actual 1 predicted 1, FP = actual 0 predicted
    1, FN = actual 1 predicted 0, TN = actual 0 predicted 0,
        accuracy = (TP + TN) / n,  precision = TP / (TP + FP),  recall = TP / (TP + FN),
        F1 = 2 * precision * recall / (precision + recall).
    The F1 score ignores the true negatives: a classifier that predicts the positive
    class for everybody has a recall of 1 and a decent F1 score when the positive class
    is the majority. Always compare with this trivial classifier.

    Returns
    -------
    dict
        The four metrics as floats.
    """
    tn, fp = cm.iloc[0, 0], cm.iloc[0, 1]
    fn, tp = cm.iloc[1, 0], cm.iloc[1, 1]
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"accuracy": (tp + tn) / cm.values.sum(), "precision": precision,
            "recall": recall, "F1": f1}


def report(name, y_true, y_pred):
    """Print the confusion matrix and the metrics of a classifier, and return the metrics."""
    cm = confusion_matrix(y_true, y_pred)
    metrics = classification_metrics(cm)
    print(f"\n{name}")
    print(cm)
    print("  ".join(f"{k} = {v:.3f}" for k, v in metrics.items()))
    return metrics


# %% ------------------------------------------------------------------------ #
# 1. The data and the question
# --------------------------------------------------------------------------- #
banner("1. Data: sex, log wage, age and education of 534 workers")
df = load_wage_data()
X, y = df[FEATURES], df[TARGET].astype(int)
print(df.head(7))
# The table of the notes hides the sex of observation TARGET_ROW; the question is
# whether P(S = 1 | W, A, E) > P(S = 0 | W, A, E) for this worker.
x_new = X.loc[TARGET_ROW]
print(f"\nobservation {TARGET_ROW} to classify: {x_new.round(3).to_dict()}"
      f"  (hourly wage {np.exp(x_new['WAGE']):.2f} dollars)")

# %% ------------------------------------------------------------------------ #
# 2. Bayes' rule: prior, likelihood, evidence, posterior
# --------------------------------------------------------------------------- #
banner("2. Bayes' rule")
priors, means, stds = class_statistics(df)
print("prior probabilities P(S = s), s = 0 (female), 1 (male):")
print(priors.round(4))
print(f"\n{int(y.sum())} men and {int((1 - y).sum())} women; the prior of 'male' is the "
      f"share of men, {priors[1]:.3f}")
# Bayes' rule: P(S = s | x) = P(S = s) f(x | S = s) / f(x). The likelihood f(x | S = s)
# is the density of the three features given the sex. Estimating it cell by cell would
# require a three-dimensional histogram: with 10 bins per variable, 1,000 cells to fill
# with 289 men and 245 women (the curse of dimensionality). The naive assumption of
# conditional independence replaces it by a product of three one-dimensional densities,
# each estimated from 2 numbers (a mean and a standard deviation) per class.
print("\nclass-conditional means:")
print(means.round(3))
print("class-conditional standard deviations:")
print(stds.round(3))
print("parameters to estimate: 2 classes x 3 features x 2 = 12, plus the prior")

# %% ------------------------------------------------------------------------ #
# 3. Classifying one observation by hand
# --------------------------------------------------------------------------- #
banner(f"3. Observation {TARGET_ROW} by hand")
numerators = {}
for s in priors.index:
    densities = {f: gaussian_pdf(x_new[f], means.loc[s, f], stds.loc[s, f]) for f in FEATURES}
    likelihood = np.prod(list(densities.values()))
    numerators[s] = priors[s] * likelihood
    print(f"class {s}: " + ", ".join(f"f({f}) = {d:.4f}" for f, d in densities.items())
          + f"; product = {likelihood:.3e}; prior x product = {numerators[s]:.3e}")
evidence = sum(numerators.values())
posterior_male = numerators[1] / evidence
print(f"evidence f(x) = {evidence:.3e}")
print(f"posterior P(male | x) = {posterior_male:.3f}, P(female | x) = {1 - posterior_male:.3f}")
print(f"posterior odds male/female = {numerators[1] / numerators[0]:.3f} "
      f"= prior odds {priors[1] / priors[0]:.3f} x likelihood ratio "
      f"{(numerators[1] / priors[1]) / (numerators[0] / priors[0]):.3f}")
print(f"decision: {'male' if posterior_male > 0.5 else 'female'};"
      f" actual sex: {'male' if y[TARGET_ROW] == 1 else 'female'}")
# The same computation with the function used below on the whole sample
posterior = naive_bayes_posterior(X, priors, means, stds)
assert np.isclose(posterior.loc[TARGET_ROW, 1], posterior_male)

# %% ------------------------------------------------------------------------ #
# 4. The whole sample: confusion matrix and metrics
# --------------------------------------------------------------------------- #
banner("4. Naive Bayes on the whole sample")
y_pred = predict_map(posterior)
metrics_nb = report("Gaussian naive Bayes (by hand, in-sample)", y, y_pred)
# The F1 score of 0.65 must be read next to the trivial classifier: predicting 'male'
# for everybody has an accuracy of 0.54 (the share of men) and an F1 score of 0.70,
# because F1 ignores the true negatives. The gain of the classifier is visible in the
# accuracy (0.60 against 0.54) and in the balance between the two kinds of errors.
metrics_trivial = report("Trivial classifier: everybody is male", y, pd.Series(1, index=y.index))

# scikit-learn's GaussianNB does the same computation, with the biased standard
# deviation (n instead of n - 1 in the denominator): one observation changes side
from sklearn.naive_bayes import GaussianNB  # noqa: E402  (imported here on purpose)
sklearn_pred = pd.Series(GaussianNB().fit(X, y).predict(X), index=y.index)
report("scikit-learn GaussianNB (in-sample)", y, sklearn_pred)
print("\nobservations classified differently by hand and by scikit-learn:",
      int((sklearn_pred != y_pred).sum()))

# Figure: the two class-conditional Gaussian densities of the log wage against the
# histograms by sex. The overlap is why the classifier makes so many errors.
grid = np.linspace(df["WAGE"].min() - 0.3, df["WAGE"].max() + 0.3, 300)
bins = np.linspace(df["WAGE"].min(), df["WAGE"].max(), 26)
plt.hist(df.loc[y == 1, "WAGE"], bins=bins, density=True, color="0.75", label="men (histogram)")
plt.hist(df.loc[y == 0, "WAGE"], bins=bins, density=True, histtype="step", color="black",
         linestyle="--", label="women (histogram)")
for s, label, style in [(0, "women", "--"), (1, "men", "-")]:
    plt.plot(grid, gaussian_pdf(grid, means.loc[s, "WAGE"], stds.loc[s, "WAGE"]), style,
             color="black", linewidth=2,
             label=f"{label}: N({means.loc[s, 'WAGE']:.2f}, {stds.loc[s, 'WAGE']:.2f}$^2$)")
plt.xlabel("log hourly wage")
plt.ylabel("density")
plt.legend(frameon=False, loc="upper right", fontsize=8)
finish_figure("naive_bayes_wage_densities.pdf")

plt.matshow(confusion_matrix(y, y_pred).values, cmap="Greys")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
finish_figure("naive_bayes_confusion.pdf")

# %% ------------------------------------------------------------------------ #
# 5. Exercise: relaxing the naive assumption
# --------------------------------------------------------------------------- #
banner("5. Exercise: relaxing the naive assumption")
# Within each sex, the features are not independent: the log wage and the education
# are positively correlated, the age and the education negatively. The naive Bayes
# classifier ignores these correlations.
for s, label in [(0, "women"), (1, "men")]:
    print(f"\ncorrelation matrix of the features among {label}:")
    print(X[y == s].corr().round(3))
# The exercise (see the notes): replace the product of one-dimensional Gaussian
# densities by ONE three-dimensional Gaussian density per class,
#     f(x | S = s) = exp(-(x - mu_s)' Sigma_s^{-1} (x - mu_s) / 2)
#                    / sqrt((2 pi)^p det(Sigma_s)),
# with mu_s the vector of class means and Sigma_s the class covariance matrix (p = 3).
# This is quadratic discriminant analysis (QDA). Steps:
#   1. estimate mu_s with X[y == s].mean() and Sigma_s with X[y == s].cov() (ddof=1);
#   2. write multivariate_gaussian_pdf(X, mu, Sigma) with numpy.linalg.inv and
#      numpy.linalg.det (the quadratic form (x - mu)' Sigma^{-1} (x - mu) is computed
#      row by row: numpy.einsum("ij,jk,ik->i", D, Sigma_inv, D) with D = X - mu);
#   3. posterior = prior x density, normalised across classes; predict the class with
#      the largest posterior; confusion matrix and metrics with the functions above;
#   4. compare with scikit-learn's QuadraticDiscriminantAnalysis, and with the
#      version that imposes a common covariance matrix to both classes (linear
#      discriminant analysis, LDA);
#   5. count the parameters: 3 means + 6 covariance entries per class, against
#      3 means + 3 standard deviations for naive Bayes. More parameters fit the sample
#      better; use a train/test split or cross-validation to check the gain out of
#      sample (Section "Train and test sets" of the notes).
# You should find that the number of men classified as women falls from 94 to 64.


def multivariate_gaussian_pdf(X, mu, Sigma):
    """Density of the multivariate normal N(mu, Sigma) at each row of X.

    To be completed by the student (see the steps above); the solution is at the end
    of the file. Parameters: X (n x p array or DataFrame), mu (length p), Sigma (p x p).
    Returns an array of n densities.
    """
    raise NotImplementedError("write the multivariate Gaussian density (exercise)")


# %% ------------------------------------------------------------------------ #
# 6. Solution of the exercise (set RUN_SOLUTION = True after trying)
# --------------------------------------------------------------------------- #
def multivariate_gaussian_pdf_solution(X, mu, Sigma):
    """Density of the multivariate normal N(mu, Sigma) at each row of X (solution).

    f(x) = exp(-q(x) / 2) / sqrt((2 pi)^p det(Sigma)),  q(x) = (x - mu)' Sigma^{-1} (x - mu).
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    p = len(mu)
    deviations = X - mu                                     # n x p
    Sigma_inv = np.linalg.inv(Sigma)
    quadratic_form = np.einsum("ij,jk,ik->i", deviations, Sigma_inv, deviations)
    return np.exp(-quadratic_form / 2) / np.sqrt((2 * np.pi) ** p * np.linalg.det(Sigma))


def gaussian_classifier_posterior(X, y, pooled_covariance=False):
    """Posterior probabilities with one multivariate Gaussian density per class.

    Parameters
    ----------
    X : pandas.DataFrame
        Features (n x p).
    y : pandas.Series
        Classes (0/1) used to estimate the priors, means and covariance matrices.
    pooled_covariance : bool
        If True, the same covariance matrix (weighted average of the class covariance
        matrices) is used for both classes: linear discriminant analysis (LDA). If
        False, each class has its own matrix: quadratic discriminant analysis (QDA).

    Returns
    -------
    pandas.DataFrame
        One column per class with the posterior probabilities.
    """
    classes = sorted(y.unique())
    covariances = {s: X[y == s].cov(ddof=1) for s in classes}
    if pooled_covariance:
        n = {s: int((y == s).sum()) for s in classes}
        pooled = sum((n[s] - 1) * covariances[s] for s in classes) / (len(y) - len(classes))
        covariances = {s: pooled for s in classes}
    numerators = pd.DataFrame(index=X.index, columns=classes, dtype=float)
    for s in classes:
        density = multivariate_gaussian_pdf_solution(X, X[y == s].mean(), covariances[s])
        numerators[s] = (y == s).mean() * density
    return numerators.div(numerators.sum(axis=1), axis=0)


if RUN_SOLUTION:
    banner("6. Solution: multivariate Gaussian likelihoods (QDA and LDA)")
    for s in [0, 1]:
        print(f"\ncovariance matrix of the features, class {s}:")
        print(X[y == s].cov().round(3))
    posterior_qda = gaussian_classifier_posterior(X, y)
    print(f"\nobservation {TARGET_ROW}: P(male | x) = {posterior_qda.loc[TARGET_ROW, 1]:.3f}"
          f" with the multivariate density, {posterior_male:.3f} with the naive one")
    report("QDA: one covariance matrix per class (by hand, in-sample)", y,
           predict_map(posterior_qda))
    posterior_lda = gaussian_classifier_posterior(X, y, pooled_covariance=True)
    report("LDA: common covariance matrix (by hand, in-sample)", y, predict_map(posterior_lda))

    from sklearn.discriminant_analysis import (  # noqa: E402
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    report("scikit-learn QuadraticDiscriminantAnalysis", y,
           pd.Series(QuadraticDiscriminantAnalysis().fit(X, y).predict(X), index=y.index))
    report("scikit-learn LinearDiscriminantAnalysis", y,
           pd.Series(LinearDiscriminantAnalysis().fit(X, y).predict(X), index=y.index))

    # Out of sample: 5-fold cross-validation (each observation is predicted by a model
    # estimated on the other four fifths of the sample)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, model in [("naive Bayes", GaussianNB()),
                        ("QDA", QuadraticDiscriminantAnalysis()),
                        ("LDA", LinearDiscriminantAnalysis())]:
        report(f"{name}, 5-fold cross-validated predictions", y,
               pd.Series(cross_val_predict(model, X, y, cv=folds), index=y.index))
