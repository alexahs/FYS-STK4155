import numpy as np
import sklearn.linear_model as skl
# from sklearn.preprocessing import PolynomialFeatures



class RegressionMethods:
    def __init__(self, method = 'ols', alpha = 0):
        self.method = method
        self.alpha = alpha



    def ols(self):
        XT = self.X.T
        self.beta = np.linalg.pinv(XT.dot(self.X)).dot(XT).dot(self.z)



    def ridge(self):

        # self.X -= np.mean(self.X, axis = 0)
        # self.z -= np.mean(self.z)
        #
        # col_var = np.var(self.X, axis = 0)
        #
        # for i in range(1, len(self.X[0,:])):
        #     self.X[i,:] /= col_var[i]

        XT = self.X.T
        p = np.shape(self.X)[1]
        L = np.identity(p)*self.alpha

        self.beta = np.linalg.pinv(XT.dot(self.X) + L).dot(XT).dot(self.z)


    def lasso(self):
        clf = skl.Lasso(alpha = alpha, fit_intercept=True, normalize=True).fit(self.X, self.z)
        self.beta = clf.coef_


    def fit(self, X, z):

        self.X = X

        if len(np.shape(z)) > 1:
            self.z = np.ravel(z)
        else:
            self.z = z


        if self.method == 'ols':
            self.ols()
        elif self.method == 'ridge':
            self.ridge()
        elif self.method == 'lasso':
            self.lasso()


    def predict(self, X):
        self.z_tilde = X @ self.beta


        return self.z_tilde

        # if self.method == 'ridge':
        #     return self.z_tilde + np.mean(self.z_tilde)
        # else:
        #     return self.z_tilde




if __name__ == '__main__':
    print("hello world")
