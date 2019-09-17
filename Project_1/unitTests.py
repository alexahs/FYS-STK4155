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
        z_tilde_skl = clf.predict(self.X)

        model = RegressionMethods(self.X, self.z_flat)
        beta = model.call_solver('ols', 0)
        z_tilde = self.X @ beta



        diff = mean_squared_error(z_tilde, z_tilde_skl)

        assert diff < self.tol


    def test_ridge(self):

        self.X -= np.mean(self.X, axis=0)
        print(np.shape(np.mean(self.X, axis=0)))
        self.z_flat -= np.mean(self.z_flat)



        clf = skl.Ridge(alpha = 0.1).fit(self.X, self.z_flat)
        z_tilde_skl = clf.predict(self.X)


        model = RegressionMethods(self.X, self.z_flat)
        beta = model.call_solver('ridge', 0.1)
        z_tilde = self.X @ beta

        diff = mean_squared_error(z_tilde, z_tilde_skl)

        print(z_tilde)
        print(z_tilde_skl)
        assert diff < self.tol


if __name__ == '__main__':
    tests = UnitTests()
    tests.test_ols()
    tests.test_ridge()
    print("All tests passed")
