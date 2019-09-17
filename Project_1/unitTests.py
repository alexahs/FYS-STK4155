from RegressionMethods import *
from project1_main import *



class UnitTests():

    def __init__(self):
        self.n = 40
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

        alpha = 0.01

        clf = skl.Ridge(alpha = alpha).fit(self.X, self.z_flat)
        z_pred_skl = clf.predict(self.X)


        model = RegressionMethods(method = 'ridge', alpha = alpha)
        model.fit(self.X, self.z_flat)
        z_pred_model = model.predict(self.X)


        # print(z_pred_model)
        # print(z_pred_skl)

        diff = mean_squared_error(z_pred_skl, z_pred_model)

        assert diff < self.tol


if __name__ == '__main__':
    tests = UnitTests()
    tests.test_ols()
    tests.test_ridge()
    print("All tests passed")
