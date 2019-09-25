def bootstrap(self, model, n_bootstraps = 50, test_size = 0.2):
    X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size = test_size)
    sampleSize = X_train.shape[0]


    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_train_pred = np.empty((z_train.shape[0], n_bootstraps))
    z_train_boot = np.empty((z_train.shape[0], n_bootstraps))



    for i in range(n_bootstraps):
        # print('bootstrap iter:', i)
        indices = np.random.randint(0, sampleSize, sampleSize)
        X_, z_ = X_train[indices], z_train[indices]
        model.fit(X_, z_)

        z_train_boot[:,i] = z_

        z_pred[:,i] = model.predict(X_test)
        z_train_pred[:,i] = model.predict(X_)



    z_test = z_test.reshape((len(z_test), 1))
    # z_train_pred = z_train_pred.reshape((len(z_train_pred), 1))
    # print('shape z train boot',z_train_boot.shape)


    error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True))
    error_train = np.mean( np.mean((z_train_pred - z_train_boot)**2, axis=1, keepdims=True))
    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )


    return error, bias, variance, error_train
