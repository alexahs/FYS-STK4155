import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from LogisticRegression import *


def load_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    return X, y

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    print(y)

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def test_log_reg():

    #generate data
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)


    #classify with sklearn
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)

    pred_func_skl = lambda x: clf.predict(x)
    plot_decision_boundary(pred_func_skl, X, y)


    #classify with own method
    model = LogisticRegression()
    model.fit(X, y)
    pred_func_own = lambda x: model.predict(x)

    plot_decision_boundary(pred_func_own, X, y)








if __name__ == '__main__':
    test_log_reg()
