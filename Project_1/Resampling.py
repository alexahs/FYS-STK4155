import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class Resampling:

    def __init__(self, X, z):
        self.X = X
        self.z = z


    def train_test(self, model, test_size = 0.2):

        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size = test_size)

        model.fit(X_train, z_train)
        z_pred = model.predict(X_test)

        error = np.mean((z_test - z_pred)**2)
        bias = np.mean((z_test - np.mean(z_pred))**2)
        variance = np.var(z_test - z_pred)
        r2 = r2_score(z_test, z_pred)

        return error, bias, variance, r2


    def k_fold_CV(self, model, k = 5, center = False):

        kfold = KFold(n_splits = k, shuffle=True)

        error = np.zeros(k)
        bias = np.zeros(k)
        variance = np.zeros(k)
        r2 = np.zeros(k)


        i = 0
        for train_inds, test_inds in kfold.split(self.X):
            X_train = self.X[train_inds]
            X_test = self.X[test_inds]
            z_train = self.z[train_inds]
            z_test = self.z[test_inds]

            if center:
                X_train_mean = np.mean(X_train, axis=0)
                z_train_mean = np.mean(z_train)
                X_train -= X_train_mean
                X_test -= X_train_mean
                z_train -= z_train_mean

            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)

            if center:
                z_pred += z_train_mean


            error[i] = np.mean((z_test - z_pred)**2)
            bias[i] = np.mean((z_test - np.mean(z_pred))**2)
            variance[i] = np.var(z_test - z_pred)
            r2[i] = r2_score(z_test, z_pred)
            i += 1

        return np.mean(error), np.mean(bias), np.mean(variance), np.mean(r2)


# if __name__ == '__main__':
