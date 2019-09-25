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
        variance = np.var(z_pred)
        r2 = r2_score(z_test, z_pred)

        return error, bias, variance, r2


    def bootstrap(self, model, n_bootstraps = 100, test_size = 0.2):
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size = test_size)
        sampleSize = X_train.shape[0]

        # mse = np.zeros(n_bootstraps)
        # variance = np.zeros(n_bootstraps)
        # r2 = np.zeros(n_bootstraps)


        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        # z_train_preds = np.empty((z_train.shape[0], n_bootstraps))



        for i in range(n_bootstraps):
            indices = np.random.randint(0, sampleSize, sampleSize)
            X_, z_ = X_train[indices], z_train[indices]
            model.fit(X_, z_)

            z_pred[:,i] = model.predict(X_test)
            # z_train_preds[:,i] = model.predict(X_)


        """
        error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
        """
        z_test = z_test.reshape((len(z_test), 1))
        z_train = z_train.reshape((len(z_train), 1))



        # error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True) )
        error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True))
        bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )


        print('Error:', error)
        print('Bias^2:', bias)
        print('Var:', variance)
        print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))



        return error, bias, variance





    def k_fold_CV(self, model, k = 5, center = False):

        kfold = KFold(n_splits = k, shuffle=True)

        error = np.zeros(k)
        bias = np.zeros(k)
        variance = np.zeros(k)
        r2 = np.zeros(k)
        train_error = np.zeros(k)


        i = 0
        for train_inds, test_inds in kfold.split(self.X):
            X_train = self.X[train_inds]
            X_test = self.X[test_inds]
            z_train = self.z[train_inds]
            z_test = self.z[test_inds]


            if center:
                X_train = X_train[1:]
                X_train_mean = np.mean(X_train, axis=0)
                z_train_mean = np.mean(z_train)
                X_train_std = np.sqrt(np.var(X_train, axis=0))
                z_train_std = np.sqrt(np.var(z_train))

                X_train -= X_train_mean
                X_test -= X_train_mean
                z_train -= z_train_mean

                X_train /= X_train_std
                X_test /= X_test_std
                z_train /= z_train_std

            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)
            z_pred_train = model.predict(X_train)

            if center:
                z_pred *= z_train_std
                z_pred += z_train_mean
                z_pred_train *= z_train_std
                z_pred_train += z_train_mean




            error[i] = np.mean((z_test - z_pred)**2)
            bias[i] = np.mean((z_test - np.mean(z_pred))**2)
            variance[i] = np.var(z_pred)
            r2[i] = r2_score(z_test, z_pred)
            train_error[i] = mean_squared_error(z_pred_train, z_train)
            i += 1


        return np.mean(error), np.mean(bias), np.mean(variance), np.mean(r2), np.mean(train_error)


# if __name__ == '__main__':
