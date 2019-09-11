import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression



class RegressionMethods:
    def __init__(self, X, z):
        self.X = X
        self.z = z


    def ols(self):
        XT = self.X.T
        self.beta = np.linalg.pinv(XT.dot(self.X)).dot(XT).dot(self.z)

        return self.beta


    def ridge(self, lambda_):
        XT = self.X.T
        p = np.shape(self.X)[1]
        L = np.identity(p)*lambda_
        self.beta = np.linalg.pinv(XT.dot(self.X) + L).dot(XT).dot(self.z)

        return self.beta


    def lasso(self, lambda_):
        clf = skl.Lasso(alpha = lambda_)
        clf.fit(self.X, self.z)
        self.beta = clf.coef_

        return self.beta


    def call_solver(self, method = 'ols', lambda_ = 0.1):
        if method == 'ols':
            return self.ols()
        elif method == 'ridge':
            return self.ridge(lambda_)
        elif method == 'lasso':
            return self.lasso(lambda_)

        return None



if __name__ == '__main__':
    # model = RegressionMethods()
    print("hello world")
