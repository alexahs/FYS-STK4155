import numpy as np
import pandas as pd
from RegressionMethods import *
from Resampling import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from imageio import imread


def load_terrain(filename):
    terrain = imread('data/' + filename)
    dims = np.shape(terrain)
    terrain = terrain[0: dims[0] // 2, 0: dims[1] //2 ]
    new_dims = np.shape(terrain)
    return terrain, new_dims[0]

def show_terrain(filename):
    terrain = imread('data/' + filename)
    plt.figure()
    plt.title('Terrain')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


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



def ols_model_complexity_analysis(x, y, z, max_deg=10, save_to_file = False):

    if save_to_file:
        filename = 'results/' + 'OLS_error_scores'
        error_scores = pd.DataFrame(columns=['degree', 'mse', 'bias', 'variance', 'r2', 'mse_train'])

    model = RegressionMethods('ols')
    degrees = np.linspace(1, max_deg, max_deg)
    mse = np.zeros(max_deg)
    bias = np.zeros(max_deg)
    variance = np.zeros(max_deg)
    r2 = np.zeros(max_deg)
    mse_train = np.zeros(max_deg)

    i = 0
    for deg in degrees:
        X = create_design_matrix(x, y, int(deg))
        print('degree: ', deg)
        resample = Resampling(X, z)

        mse[i], bias[i], variance[i], r2[i], mse_train[i] = resample.k_fold_CV(model)

        if save_to_file:
            error_scores = error_scores.append({'degree': degrees[i],
                                                'mse': mse[i],
                                                'bias': bias[i],
                                                'variance': variance[i],
                                                'r2': r2[i],
                                                'mse_train': mse_train[i]}, ignore_index=True)
        #end if
        i += 1
    #end for


    plt.plot(degrees, mse, label='test set')
    plt.plot(degrees, mse_train, label='training set')
    plt.legend()
    plt.xlabel('Model complexity [deg]')
    plt.ylabel('Mean Squared Error')
    plt.show()


    if save_to_file:
        error_scores.to_csv(filename + '.csv')
        print(error_scores)
        print(error_scores['mse'].min())





def ridge_lasso_complexity_analysis(x, y, z, model_name, k = 5, max_deg=10, save_to_file = True):

    n_lambdas = 10
    model = RegressionMethods(model_name)

    lambdas = np.logspace(-3, 3, n_lambdas)
    degrees = np.linspace(1, max_deg, max_deg)
    # mse = np.zeros(max_deg)
    # bias = np.zeros(max_deg)
    # variance = np.zeros(max_deg)
    # r2 = np.zeros(max_deg)
    # mse_train = np.zeros(max_deg)

    if save_to_file:
        filename = 'results/' + model_name + '_error_scores'
        error_scores = pd.DataFrame(columns=['degree', 'lambda', 'mse', 'bias', 'variance', 'mse_train'])


    i = 0
    for deg in degrees:
        j = 0
        X = create_design_matrix(x, y, int(deg))
        print('degree: ', deg)
        resample = Resampling(X, z)
        for lamb in lambdas:
            model.set_alpha(lamb)

            mse, bias, variance, r2, mse_train = resample.k_fold_CV(model)


            if save_to_file:
                error_scores = error_scores.append({'degree': deg,
                                                    'lambda': lamb,
                                                    'mse': mse,
                                                    'bias': bias,
                                                    'variance': variance,
                                                    'r2': r2,
                                                    'mse_train': mse_train}, ignore_index=True)
            #end if
            j+=1
        #end for lambdas
        # print('Error:', mse[i])
        # print('Bias^2:', bias[i])
        # print('Var:', variance[i])
        # print('{} >= {} + {} = {}'.format(mse[i], bias[i], variance[i], bias[i]+variance[i]))
        i+=1
    #end for degrees



    if save_to_file:
        error_scores.to_csv(filename + '.csv')
        print(error_scores)
        print(error_scores['mse'].min())
