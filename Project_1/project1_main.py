from RegressionMethods import *
from Resampling import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
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



def plot_param_heatmap(filename_csv, filename_npy):


    # data = pd.read_csv(data_filename)
    csv_data = pd.read_csv(filename_csv)
    npy_data = np.load(filename_npy)

    lambdas = csv_data['lambda'].unique()
    degrees = csv_data['degree'].unique()



    # print(data)

    im = plt.imshow(npy_data, cmap='PuBu_r', interpolation='nearest', norm=colors.LogNorm(vmin=npy_data.min(), vmax=npy_data.max()))
    plt.colorbar(im)
    plt.xlabel("degree")
    plt.ylabel("lambda")
    plt.show()


def analyze_model_params(x, y, z, model, k = 5, max_deg=10):



    degrees = np.linspace(1, max_deg, max_deg)


    if model.method == 'ols':
        lambdas = [0]
    else:
        n_lambdas = 20
        lambdas = np.logspace(-3, 3, n_lambdas)

    filename = 'errorScores_' + model.method

    error_scores = pd.DataFrame(columns=['degree', 'lambda', 'mse', 'bias', 'variance'])

    if model.method != 'ols':
        mse_grid = np.zeros((len(degrees), n_lambdas))

    i = 0

    for deg in degrees:
        j = 0
        X = create_design_matrix(x, y, int(deg))
        print(deg)
        resample = Resampling(X, z)
        for lamb in lambdas:
            # print(lamb)
            model.set_alpha(lamb)

            error, bias, variance, r2 = resample.k_fold_CV(model)



            if model.method != 'ols':
                mse_grid[i, j] = error

            j+=1

            error_scores = error_scores.append({'degree': deg,
                                                'lambda': lamb,
                                                'mse': error,
                                                'bias': bias,
                                                'variance': variance,
                                                'r2': r2}, ignore_index=True)


        i+=1



    error_scores.to_csv(filename + '.csv')
    if model.method != 'ols':
        np.save("mse_" + model.method, mse_grid)
    print(mse_grid)


def show_terrain():
    terrain1 = imread('terrain2.tif')
    # Show the terrain
    plt.figure()
    plt.title('Terrain')
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():

    np.random.seed(100)
    n = 20
    deg = 5
    sigma = 0.05
    x, y = generate_mesh(n)
    z = frankie_function(x, y, n, sigma)
    z_flat = np.ravel(z)
    X = create_design_matrix(x, y, deg)

    model = RegressionMethods('ridge')


    # analyze_model_params(x, y, z_flat, model)
    # plot_param_heatmap('errorScores_ridge.csv', 'mse_ridge.npy')

    show_terrain()
    # resample = Resampling(X, z_flat)
    # mse, bias, variance = resample.k_fold_CV(model)
    #
    #
    # print('Error:', mse)
    # print('Bias^2:', bias)
    # print('Var:', variance)
    # print('{} >= {} + {} = {}'.format(mse, bias, variance, bias+variance))






if __name__ == '__main__':
    main()
