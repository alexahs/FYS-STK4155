from RegressionMethods import *
from Resampling import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.colors as colors
from imageio import imread


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







# def plot_param_heatmap(filename_csv, filename_npy):
#
#
#     # data = pd.read_csv(data_filename)
#     csv_data = pd.read_csv(filename_csv)
#     # npy_data = np.load(filename_npy)
#
#     lambdas = csv_data['lambda'].unique()
#     degrees = csv_data['degree'].unique()




    # print(data)

    # im = plt.imshow(npy_data, cmap='PuBu_r', interpolation='nearest', norm=colors.LogNorm(vmin=npy_data.min(), vmax=npy_data.max()))
    # plt.colorbar(im)
    # plt.xlabel("degree")
    # plt.ylabel("lambda")
    # plt.show()

def ols_model_complexity_analysis(x, y, z, max_deg=20, save_to_file = False):


    if save_to_file:
        filename = 'results/' + 'OLS_error_scores'
        error_scores = dp.DataFrame(columns=['degree', 'mse', 'bias', 'variance', 'training error'])

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

        i += 1

    #end for degrees


    print(mse)
    print(mse_train)

    plt.plot(degrees, mse, label='mse test')
    plt.plot(degrees, mse_train, label='mse train')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()


def analyze_model_params(x, y, z, model, k = 5, max_deg=10, save_to_file = False):

    degrees = np.linspace(1, max_deg, max_deg)

    if model.method == 'ols':
        lambdas = [0]
    else:
        n_lambdas = 10
        lambdas = np.logspace(-3, 3, n_lambdas)



    if save_to_file:
        filename = 'results/' + 'errorScores_' + model.method
        error_scores = pd.DataFrame(columns=['degree', 'lambda', 'mse', 'bias', 'variance', 'training error'])

    # if model.method != 'ols':
    #     mse_grid = np.zeros((len(degrees), n_lambdas))

    i = 0
    for deg in degrees:
        j = 0
        X = create_design_matrix(x, y, int(deg))
        print(deg)
        resample = Resampling(X, z)
        for lamb in lambdas:
            # print(lamb)
            model.set_alpha(lamb)

            error, bias, variance, r2, train_error = resample.k_fold_CV(model)

            # if model.method != 'ols':
            #     mse_grid[i, j] = error


            if save_to_file:
                error_scores = error_scores.append({'degree': deg,
                                                    'lambda': lamb,
                                                    'mse': error,
                                                    'bias': bias,
                                                    'variance': variance,
                                                    'r2': r2,
                                                    'training error': train_error}, ignore_index=True)
            #end if
            j+=1
        #end for lambdas
        i+=1
    #end for degrees



    error_scores.to_csv(filename + '.csv')
    # if model.method != 'ols':
    #     np.save("mse_" + model.method, mse_grid)
    # print(mse_grid)


def show_terrain():
    terrain1 = imread('data/terrain2.tif')
    print(np.shape(terrain1))
    print(np.max(terrain1))
    print(np.min(terrain1))
    # Show the terrain
    plt.figure()
    plt.title('Terrain')
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():

    np.random.seed(10)
    n = 100
    deg = 5
    sigma = 0.1
    x, y = generate_mesh(n)
    z = frankie_function(x, y, n, sigma)
    z_flat = np.ravel(z)
    # X = create_design_matrix(x, y, deg)

    # model = RegressionMethods('ridge')



    ols_model_complexity_analysis(x, y, z_flat)


    # analyze_model_params(x, y, z_flat, model, save_to_file=True)
    # plot_param_heatmap('results/errorScores_ols.csv', 'mse_ridge.npy')

    # show_terrain()
    # resample = Resampling(X, z_flat)
    # mse, bias, variance, r2, train_error = resample.k_fold_CV(model)
    # #
    # #
    # print('Error:', mse)
    # print('Bias^2:', bias)
    # print('Var:', variance)
    # print('{} >= {} + {} = {}'.format(mse, bias, variance, bias+variance))






if __name__ == '__main__':
    main()
