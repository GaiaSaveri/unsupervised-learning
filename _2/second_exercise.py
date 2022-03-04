import numpy as np
from sklearn.metrics import mutual_info_score, accuracy_score

# linear prediction function: z = Wx + b (logit scores)
# with W = [#labels, #features], X = [#features, 1], B = [#labels, 1] --> z = [#labels, 1]
def linear_layer(X, W, b):
    Z = np.array([np.empty([W.shape[0]]) for j in range(X.shape[0])]) # z = [#data, #labels]
    for i in range(X.shape[0]):
        Z[i] = (np.matmul(W, (X[i].T)).reshape(-1, 1) + b).reshape(-1)
    return Z

# softmax layer: softmax(z_i)
def softmax_layer(Z):
    prob = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        max_num = np.max(Z[i, :], keepdims=True)  # prevent overflow of exponential
        num = np.exp(Z[i, :] - max_num)
        den = np.sum(num)
        prob[i] = num/den
    return prob

# predict the most likely label for each data point
def prediction_layer(prob):
    predictions = np.array([np.argmax(i) for i in prob])
    return predictions

# stacking layers together
def multinomial_logistic_regression(X, W, b):
    Z = linear_layer(X, W, b)
    P = softmax_layer(Z)
    y_hat = prediction_layer(P)
    return P, y_hat

# loss function used to optimize parameters: cross entropy loss
def cross_entropy_loss(P, labels):
    n_data = P.shape[0]
    loss = 0
    for prob, label in zip(P, labels):
        if(prob[label] == 0):
            prob[label] = 1e-10 # avoid division by 0
        # assuming labels are 1-hot encoded
        loss += -np.log(prob[label])
    loss /= n_data
    return loss

# optimizing with stochastic gradient descent
def training(X, labels, learning_rate=0.05, epochs=500):
    # random initialization of weights abd biases:
    n_features = X.shape[1]
    n_labels = len(np.unique(labels))
    n_data = X.shape[0]
    W = np.random.rand(n_labels, n_features)
    b = np.random.rand(n_labels, 1)
    losses = np.array([])
    # training loop
    for i in range(epochs):
        prob, _ = multinomial_logistic_regression(X, W, b)
        loss = cross_entropy_loss(prob, labels)
        losses = np.append(losses, loss)
        # gradients
        prob[np.arange(n_data), labels] -= 1
        dLdW = np.matmul(prob.T, X)
        dLdb = np.sum(prob, axis=0).reshape(-1, 1)
        # update W, b
        W -= (learning_rate * dLdW)
        b -= (learning_rate * dLdb)
    return W, b, losses

# quantifying the quality of the prediction
def mutual_information(x, y):
    MI = 0.0
    x_classes = np.unique(x)
    y_classes = np.unique(y)
    px = np.array([len(x[x==x_class])/float(len(x)) for x_class in x_classes]) # p(x)
    py = np.array([len(y[y==y_class])/float(len(y)) for y_class in y_classes]) # p(y)
    for i in range(len(x_classes)):
        if px[i] == 0:
            continue
        sy = y[x==x_classes[i]]
        if len(sy) == 0:
            continue
        pxy = np.array([len(sy[sy==y_class])/float(len(y)) for y_class in y_classes]) # p(x, y)
        frac = pxy[py>0.]/(py[py>0.] * px[i])
        MI += np.sum(pxy[pxy>0.]*np.log2(frac[frac>0.]))
    return MI

# testing the quality of the prediction
def testing(X, labels, W, b):
    P, y_hat = multinomial_logistic_regression(X, W, b)
    print("Accuracy: ", accuracy_score(labels, y_hat, normalize=True))
    print("Mutual information score: ", mutual_information(labels, y_hat))
    return mutual_information(labels, y_hat)
