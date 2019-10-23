import numpy as np


class LogisticRegression:

    def __init__(self, n_epochs=1000, size_minibatch=128):
        self.n_epochs = n_epochs
        self.size_minibatch = size_minibatch


    def sigmoid(self, X, theta):
        term = np.exp(X @ theta)
        return term/(1 + term)



    def SGD(self):

        n = len(self.X[0])

        n_batches = int(self.size_minibatch/n)

        theta = np.random.rand(self.X.shape[1])

        t0 = 1
        t1 = 10
        learning_rate = lambda t: t0/(t + t1)




        for epoch in range(self.n_epochs):
            for i in range(n_batches):
                random_indices = np.random.randint(self.size_minibatch, size=self.size_minibatch)
                Xi = self.X[random_indices,:]
                yi = self.y[random_indices]
                gradients = Xi.T @ (self.sigmoid(Xi, theta) - yi)
                eta = learning_rate(epoch*n_batches + i)
                theta = theta - eta*gradients


        self.theta = theta


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.SGD()


    def predict(self, X, confidence=False):
        z = X @ self.theta
        if confidence:
            return z
        else:
            return (z > 0).astype(np.int)



    def set_n_epoch(self, n):
        self.n_epochs = n

    def set_size_minibatch(self, n):
        self.size_minibatch = n
