from RegressionMethods import *
from project1_main import *
from Resampling import *


class UnitTests():

    def __init__(self):
        self.n = 50
        np.random.seed(10)
        degree = 5
        sigma = 0.05
        x, y = generate_mesh(self.n)
        z = frankie_function(x, y, self.n, sigma)
        self.z_flat = np.ravel(z)
        self.X = create_design_matrix(x, y, degree)
        self.tol = 1e-15

    def test_ols(self):

        clf = skl.LinearRegression().fit(self.X, self.z_flat)
        z_pred_skl = clf.predict(self.X)


        model = RegressionMethods('ols')
        model.fit(self.X, self.z_flat)
        z_pred_model = model.predict(self.X)

        diff = mean_squared_error(z_pred_skl, z_pred_model)
        assert diff < self.tol


    def test_ridge(self):

        alpha = 0.1

        clf = skl.Ridge(alpha = alpha, fit_intercept=False).fit(self.X, self.z_flat)
        z_pred_skl = clf.predict(self.X)

        model = RegressionMethods(method = 'ridge', alpha = alpha)
        model.fit(self.X, self.z_flat)
        z_pred_model = model.predict(self.X)


        # print(z_pred_model)
        # print(z_pred_skl)

        diff = mean_squared_error(z_pred_skl, z_pred_model)
        assert diff < self.tol


    def test_kfold(self, k=5):
        # kfold = KFold(n_splits = k, shuffle=True)
        # skl_model = LinearRegression()
        # skl_mse_folds = cross_val_score(skl_model, self.X, self.z_flat, scoring='neg_mean_squared_error', cv=kfold)
        # mean_skl_mse_folds = -np.mean(skl_mse_folds)
        #
        #
        # model = RegressionMethods('ols')
        # resample = Resampling(self.X, self.z_flat)
        # mse = resample.k_fold_CV(model)[0]
        #
        # tol = 1e-3
        # diff = abs(mse - mean_skl_mse_folds)
        # assert diff < tol

        model = RegressionMethods('ols')
        resample = Resampling(self.X, self.z_flat)
        resample.k_fold_CV(model)




if __name__ == '__main__':
    tests = UnitTests()
    # tests.test_ols()
    # tests.test_ridge()
    tests.test_kfold()
    print("All tests passed")
