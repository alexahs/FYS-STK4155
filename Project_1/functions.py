import numpy as np
import pandas as pd
from RegressionMethods import *
from Resampling import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from imageio import imread
import seaborn as sns


def load_terrain(filename):
    slice = 2
    terrain = imread('data/' + filename)
    dims = np.shape(terrain)
    if dims[0] != dims[1]:
        terrain = terrain[0:dims[1], :]
        dims = terrain.shape
    # if dims[0] < 450:
    #     slice = 4
    terrain = terrain[0:dims[0]//2, 0:dims[1]//2]
    terrain = terrain[0:-1:slice, 0:-1:slice]
    dims = np.shape(terrain)
    print(filename, 'loaded.', dims[0],'x',dims[1])
    return terrain*0.001, dims[0]

def show_terrain(terrain_data):
    plt.figure()
    terrain1 = imread(terrain_data)
    plt.title('Terrain')
    plt.imshow(terrain_data, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def generate_mesh(n, random_pts = 0):
    if random_pts == 0:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
    if random_pts == 1:
        x = np.random.rand(n)
        y = np.random.rand(n)
    return np.meshgrid(x, y)


def frankie_function(x, y, n, sigma = 0, mu = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(mu, sigma, n)


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
    surf = ax.plot_surface(x, y, z, cmap=cm.terrain,
                           linewidth=0, antialiased=False)

    # ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_model(x, y, z, model, deg):
    X = create_design_matrix(x, y, deg)
    z_flat = np.ravel(z)
    model.fit(X, z_flat)
    z_pred = model.predict(X).reshape(len(x), len(y))

    # z_pred = z_pred[0:-1:10]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z_pred,
        cmap=cm.terrain, alpha=0.9)

    # surface2 = ax.plot_surface(x, y, z, cmap=cm.binary, alpha=0.5)

    plot_mesh(x, y, z, len(x))

    x = x[0:-1:10]
    y = y[0:-1:10]
    z = z[0:-1:10]
    # ax.scatter(x, y, z, alpha=0.5, c='#808080',  s=0.5)

    # ax.legend()

    plt.show()
