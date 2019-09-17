from RegressionMethods import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge


def frankie_function(x, y, n, sigma = 0, mu = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(mu, sigma, n)


def generate_mesh(n, random_pts = 0):
    if random_pts == 0:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
    if random_pts == 1:
        x = np.random.rand(n)
        y = np.random.rand(n)
    return np.meshgrid(x, y)


def create_design_matrix(x, y, deg):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((deg + 1)*(deg + 2)/2)
    X = np.ones((N,p))

    for i in range(1, deg + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X



def plot_mesh(x, y, z, n):
    z = np.reshape(z, (n, n))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



def k_fold(X, z, solver_name, lambda_ = 0, k=5, normalize = True):
    R2_scores = np.zeros(k)
    MSE_scores = np.zeros(k)
    kfold = KFold(n_splits = k, shuffle=True)


    i = 0

    for train_inds, test_inds in kfold.split(X):



        X_train = X[train_inds]
        X_test = X[test_inds]

        z_train = z[train_inds]
        z_test = z[test_inds]


        # if normalize:
        #     X_train_mean = np.mean(X_train, axis=0)
        #

        #normalisere test data med variansen til treningsdata (og gjennomsnitt)
        #

        model = RegressionMethods(X_train, z_train)

        beta = model.call_solver(solver_name, lambda_)
        z_test_predict = X_test @ beta

        MSE_scores[i] = mean_squared_error(z_test_predict, z_test)
        R2_scores[i] = r2_score(z_test_predict, z_test)

        i += 1


    return np.mean(R2_scores), np.mean(MSE_scores)


def get_ridge_confidence_metrics(X, z, lambda_start = -3, lambda_stop = 3, n_lambdas = 100, k = 5):

    R2_scores = np.zeros(n_lambdas)
    MSE_scores = np.zeros(n_lambdas)
    lambdas = np.logspace(lambda_start, lambda_stop, n_lambdas)


    for lamb, i in zip(lambdas, range(len(lambdas))):
        vals = k_fold(X, z, 'ridge', lambda_ = lamb)
        R2_scores[i], MSE_scores[i] = vals


    return R2_scores, MSE_scores, lambdas


# def get_ols_confidence_metrics(X, z, k = 5):
#     complexity = np.linspace(1, 6, 6)

def main():

    np.random.seed(100)

    n = 100

    deg = 5
    sigma = 0.05
    x, y = generate_mesh(n)
    z = frankie_function(x, y, n, sigma)
    z_flat = np.ravel(z)

    # complexity = np.linspace(1, 6, 6)




    X = create_design_matrix(x, y, deg)
    model = RegressionMethods(X, z_flat)
    beta = model.call_solver('lasso', 0.0001)

    # print(beta)

    # _, MSEs, lambdas = get_ridge_confidence_metrics(X, z_flat)

    # n_lambdas = 10
    # lambdas = np.logspace(-3, 2, n_lambdas)



    # plt.plot(lambdas, MSEs)
    # plt.xlabel("lambda")
    # plt.ylabel("MSE")
    # plt.show()


    # error_scores = pd.DataFrame(columns=['degree', 'lambda', 'MSE', 'R2'])


    # for lmb in lambdas:
    #
    #     for deg in complexity:
    #         print(deg)
    #         X = create_design_matrix(x, y, int(deg))
    #
    #         R2, MSE = k_fold(X, z_flat, 'ridge', lmb)
    #
    #         error_scores = error_scores.append({'degree': int(deg), 'lambda': lmb, 'MSE': MSE, 'R2': R2}, ignore_index=True)
    #     print(lmb)


    # error_scores.to_csv('k_fold_error_metrics.csv')




if __name__ == '__main__':
    main()
