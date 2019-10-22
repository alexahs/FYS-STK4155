import numpy as np


class LogisticRegression:

    def __init__(self):
        return None



    def sigmoid(self, X, theta):
        term = np.exp(X @ theta)
        return term/(1 + term)



    def SGD(self, n_epochs=1000, size_minibatch=10):

        n = len(self.X[0])

        n_batches = int(size_minibatch/n)

        theta = np.random.randn(n)

        t0 = 1
        t1 = 10
        learning_rate = lambda t: t0/(t + t1)

        for epoch in range(n_epochs):
            for i in range(n_batches):
                random_indices = np.random.randint(size_minibatch, size=size_minibatch)
                Xi = self.X[random_indices,:]
                yi = self.y[random_indices]
                gradients = Xi.T @ (self.sigmoid(Xi, theta) - yi)
                eta = learning_rate(epoch*n_batches + i)
                theta -= eta*gradients


        self.theta = theta


    def fit(self, X, y, n_epochs=10, size_minibatch=10):
        self.X = X
        self.y = y
        self.SGD(n_epochs, size_minibatch)


    def predict(self, X, confidence=False):
        z = X @ self.theta
        if confidence:
            return z
        else:
            return (z > 0).astype(np.int)
