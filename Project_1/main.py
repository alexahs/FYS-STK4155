from RegressionMethods import *
from Resampling import *
from functions import *
from analysis import *
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








def main():

    np.random.seed(100)
    n = 100
    deg = 5
    sigma = 0.2

    ### frankie data
    # x, y = generate_mesh(n)
    # z = frankie_function(x, y, n, sigma)
    # z_flat = np.ravel(z)
    ###

    ### terrain data
    terrain_data, n = load_terrain('norway1.tif')
    z_flat = np.ravel(terrain_data)
    x, y = generate_mesh(n)
    ###

    # plot_mesh(x, y, z, n)

    # model = RegressionMethods('ols')
    # model.fit(X, z_flat)
    # model.predict(X)
    # betas = model.beta
    #
    # confidence_interval_ols(X, z, betas)




    # plot_mesh(x, y, terrain_data, n)

    # show_terrain('SRTM_data_Norway_1.tif')






    model_degree_analysis(x, y, z_flat, 'ols', min_deg=6, max_deg = 15, alpha = 10**-10)
    # ridge_lasso_complexity_analysis(x, y, z_flat, 'ridge',min_deg=5, max_deg=14)



if __name__ == '__main__':
    main()
