import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures



np.random.seed(1337)


class Frankie():

    def __init__(self, n, degree=2, noiseCoeff=0):
        self.n = n
        self.degree = degree
        self.noiseCoeff = noiseCoeff
        self.x, self.y = self.generate()
        self.X = self.create_design_matrix()


    def frankie_function(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4



    def generate(self):
        x = np.linspace(0, 1, self.n)
        y = np.linspace(0, 1, self.n)
        return x, y


    def create_design_matrix(self):
        X = np.column_stack((self.x, self.y))
        poly = PolynomialFeatures(self.degree)
        return poly.fit_transform(X)


    def plot_frankie(self):
        xx, yy = np.meshgrid(self.x, self.y)
        z = self.frankie_function(xx, yy)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()



    def OLS(self):
        eps = self.noiseCoeff*np.random.randn(self.n)
        print(eps)
        self.z = self.frankie_function(self.x, self.y) + eps
        reg = skl.LinearRegression().fit(self.X, self.z)
        self.beta = reg.coef_
        self.z_tilde = reg.predict(self.X)


    def confidence_metrics(self):
        self.MSE = mean_squared_error(self.z, self.z_tilde)
        self.R2 = r2_score(self.z, self.z_tilde)






model = Frankie(10, 5, 0.1)

model.OLS()
model.confidence_metrics()
print(model.beta)
print(model.MSE)
print(model.R2)
# model.plot_frankie()



# x = np.linspace(0, 9, 10)
# y = np.linspace(10, 19, 10)
# #
# X = np.column_stack((x, y))
# poly = PolynomialFeatures(2)
# y = poly.fit_transform(X)
# # print(y)
# print(y)

# print(poly.get_feature_names())
