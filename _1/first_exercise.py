import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import pandas as pd
import seaborn as sns


def normalize(X):
    return np.divide((X - X.mean(axis=0)), X.std(axis=0))

def get_spectrum(X):
    # compute the covariance of the data
    C = np.matmul(X.T, X)/(X.shape[0]) # D x D
    # compute eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors = np.linalg.eigh(C)
    # arrange eigenvalues and eigenvectors following the decreasing order of the eigenvalues
    sorting_idx = np.argsort(-evalues)
    evalues = evalues[sorting_idx]
    evectors = evectors[:, sorting_idx]
    return evalues, evectors

def plot_log_spectrum(evalues):
    ticks = np.arange(1,len(evalues)+1)
    plt.figure(figsize=(16,9))
    plt.scatter(ticks, np.log(evalues), marker='o', color='orange')
    plt.xticks(ticks);
    plt.ylabel("Log-eigenvalues")
    plt.xlabel("Eigenvalue index")
    plt.show()

def get_PC(X, num_components, plot=False):
    evalues, evectors = get_spectrum(X)
    if(plot):
        plot_log_spectrum(evalues)
    # compute projection matrix
    A = evectors[:, :num_components]
    # compute principal components
    Y = X @ A
    return Y

def plot_PC(data, hue):
    plt.figure(figsize=(12,8))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=hue);
    plt.xlabel('First Coordinate')
    plt.ylabel('Second Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1));
    plt.show()
