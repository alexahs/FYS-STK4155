from RegressionMethods import *
from Resampling import *
from functions import *
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

    np.random.seed(10)
    n = 100
    deg = 5
    sigma = 0.5

    x, y = generate_mesh(n)
    z = frankie_function(x, y, n, sigma)
    z_flat = np.ravel(z)
    X = create_design_matrix(x, y, deg)

    ### terrain data
    terrain_data, n = load_terrain('terrain1.tif')
    z_flat = np.ravel(terrain_data)
    x, y = generate_mesh(n)
    ###



    # ols_model_complexity_analysis(x, y, z_flat, save_to_file=True)
    ridge_lasso_complexity_analysis(x, y, z_flat, 'ridge', save_to_file=True)


    # resample = Resampling(X, z_flat)
    # mse, bias, variance, r2, train_error = resample.k_fold_CV(model)






if __name__ == '__main__':
    main()
