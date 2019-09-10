import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


n = 20
degree = 5

def frankie_function(x, y, eps = 0.05):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + eps*np.random.randn(n)




def generate_xy(n, start = 0, stop = 1):
    x = np.linspace(start, stop, n)
    y = np.linspace(start, stop, n)
    return x, y


def create_design_matrix(x, y, deg, method = 0):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((deg+1)*(deg+2)/2)
    X = np.ones((N,p))

    for i in range(1,deg+1):
    	q = int((i)*(i+1)/2)
    	for k in range(i+1):
    		X[:,q+k] = x**(i-k) * y**k

    if method == 0:
        return X
    if method == 1:
        return X[:,1:]


def OLS(X, z):
    XT = X.T
    beta = np.linalg.pinv(XT.dot(X)).dot(XT).dot(z)
    return beta



x, y = generate_xy(n)
x, y = np.meshgrid(x, y)

z = frankie_function(x, y)



# print(x)

z_flat = np.ravel(z)
X = create_design_matrix(x, y, degree)


# X_train, X_test, Y_train, Y_test = ...


beta = OLS(X, z_flat)



z_tilde = X @ beta



MSE = mean_squared_error(z_flat, z_tilde)
R2 = r2_score(z_flat, z_tilde)
# print(np.sqrt((z - z)**2))
print(MSE)
print(R2)
