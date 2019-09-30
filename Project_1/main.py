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
    n = 20
    deg = 5
    sigma = 0.2

    ### frankie data
    x, y = generate_mesh(n)
    z = frankie_function(x, y, n, sigma)
    z_flat = np.ravel(z)
    # X = create_design_matrix(x, y, deg)
    ###

    # plot_mesh(x, y, z, n)

    # model = RegressionMethods('ols')
    # model.fit(X, z_flat)
    # model.predict(X)
    # betas = model.beta
    #
    # confidence_interval_ols(X, z, betas)



    ### terrain data
    # terrain_data, n = load_terrain('terrain2.tif')
    # z_flat = np.ravel(terrain_data)
    # x, y = generate_mesh(n)
    # X = create_design_matrix(x, y, deg)
    ###

    # plot_mesh(x, y, terrain_data, n)

    # show_terrain(terrain_data)



    # resample = Resampling(X, z_flat)
    # error, bias, variance, mse_train = resample.bootstrap(model)
    # print('Error:', error)
    # print('Bias^2:', bias)
    # print('Var:', variance)
    # print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))





    # model_degree_analysis(x, y, z_flat, 'ols', max_deg = 10)
    ridge_lasso_complexity_analysis(x, y, z_flat, 'ridge', max_deg=10)


    # resample = Resampling(X, z_flat)
    # mse, bias, variance, r2, train_error = resample.k_fold_CV(model)






if __name__ == '__main__':
    main()
