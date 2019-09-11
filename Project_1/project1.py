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



n = 100
degree = 5

def frankie_function(x, y, eps = 0.05):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + eps*np.random.randn(n)




def generate_xy(n, start = 0, stop = 1):
    # x = np.linspace(start, stop, n)
    # y = np.linspace(start, stop, n)
    x = np.random.rand(n)
    y = np.random.rand(n)
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




def k_fold(X, z, k = 5):
    R2_scores = np.zeros(k)
    MSE_scores = np.zeros(k)

    kfold = KFold(n_splits = k)

    i = 0
    for train_inds, test_inds in kfold.split(X):


        X_train = X[train_inds]
        X_test = X[test_inds]

        z_train = z[train_inds]
        z_test = z[test_inds]

        beta = OLS(X_train, z_train)
        z_test_predict = X_test @ beta

        MSE_scores[i] = mean_squared_error(z_test_predict, z_test)
        R2_scores[i] = r2_score(z_test_predict, z_test)

        i += 1


    return R2_scores, MSE_scores


x, y = generate_xy(n)
x, y = np.meshgrid(x, y)


z = frankie_function(x, y)

z_flat = np.ravel(z)


X = create_design_matrix(x, y, degree)


KF = k_fold(X, z_flat)

print("k-fold CV model performance")
print("R2 scores:  ", KF[0])
print("MSE scores: ", KF[1])

# X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)
#
# beta = OLS(X_train, z_train)
#
#
# z_train_predict = X_train @ beta
# print("Model performance for training set")
# print("MSE:     ", mean_squared_error(z_train_predict, z_train))
# print("R2 score:", r2_score(z_train_predict, z_train), "\n")
#
#
# z_test_predict = X_test @ beta
# print("Model performance for test set")
# print("MSE:     ", mean_squared_error(z_test_predict, z_test))
# print("R2 score:", r2_score(z_test_predict, z_test), "\n")
