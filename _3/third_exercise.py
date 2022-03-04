import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import seaborn as sns
import random


# compute pair-wise euclidean distance
def sq_euclidean_distance(x, y):
    return np.sum((x-y)**2, axis=-1)

# compute distance between points and cluster centroids
def centroid_distance(X, C):
    return sq_euclidean_distance(X[:, np.newaxis, :], C[np.newaxis, :, :])

# assign point to the same cluster of the nearest cluster centroid
def assign(D):
    return np.argsort(D)[:, 0]  # z

# update cluster centroids
def update_centroids(X, z):
    partition = [X[np.where(z==i), :].squeeze(0) for i in np.unique(z)]
    # average position of points belonging to the cluster
    C = np.array([np.mean(i, axis=0) for i in partition])
    return C

# evaluate objective function: within clusters sums of squares
def objective_f(X, C, z):
    partition = [X[np.where(z==i), :].squeeze(0) for i in np.unique(z)]
    sum_intra_c = [(sq_euclidean_distance(partition[i][:, np.newaxis, :],
                    C[np.newaxis, i, :]).sum())
                    for i in range(len(C))]
    return sum(sum_intra_c)

def k_means_iter(X, z, update_fn):
    C = update_fn(X, z)
    D = centroid_distance(X, C)
    z = assign(D)
    return C, D, z

def k_means(X, C_init, update_fn, k=15):
    D = centroid_distance(X, C_init)
    z_prev = assign(D)
    C, D, z = k_means_iter(X, z_prev, update_fn)
    while (not np.array_equal(z, z_prev)):
        z_prev = z
        C, D, z = k_means_iter(X, z_prev, update_fn)
    obj = objective_f(X, C, z)
    return obj, z, C

def k_means_repeated(X, init_fn, update_fn, k=15, repeats=1):
    obj = []
    min_obj = None
    best_assignation = None
    best_centers = None
    for i in range(repeats):
        C_init = init_fn(X, k)
        obj_i, z, C = k_means(X, C_init, update_fn, k)
        obj.append(obj_i)
        if i==0 or obj_i < min_obj:
            min_obj = obj_i
            best_assignation = z
            best_centers = C
    obj = np.array(obj)
    return np.mean(obj), min_obj, best_assignation, best_centers

def random_init(X, k=15):
    init_idx = np.random.choice(X.shape[0], k, replace=False)
    C_init = X[init_idx, :]
    return C_init

def k_plus_plus(X, k=15):
    init_idx = np.random.randint(X.shape[0])
    C = np.array([X[init_idx, :]])
    for i in range(1,k):
        # distance of every point from already computed centroids
        D = centroid_distance(X, C)
        # min distance of every point from centroids
        d = np.min(D, axis=1)
        # prob of points being next centroid
        p = d/np.sum(d)
        # cumulative probability distribution
        cumulative_p = np.cumsum(p)
        r=random.random()
        for j,p in enumerate(cumulative_p):
            if r<p:
                l=j
                break
        C = np.append(C, np.array([X[l, :]]), axis=0)
    return C

def scree_plot(values, min_k, max_k, lag):
    ticks = np.arange(min_k, max_k+1, lag)
    plt.figure(figsize=(16,9))
    plt.scatter(np.arange(len(values)), values, marker='o', color='orange')
    plt.xticks(ticks=np.arange(len(ticks)), labels=ticks);
    plt.ylabel("Best value of the objective function")
    plt.xlabel("Number of clusters k")
    plt.show()

def plot_assignation(X, z, C):
    plt.figure(figsize=(15,10))
    plt.scatter(X[:, 0], X[:, 1], c=z)
    plt.scatter(C[:, 0], C[:, 1], c='red', marker='x')
    plt.show()
