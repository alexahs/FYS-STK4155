from RegressionMethods import *
from Resampling import *
from functions import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors






def ols_model_complexity_analysis(x, y, z, max_deg=20, n_bootstraps = 100):

    filename = 'results/' + 'OLS_error_scores'
    error_scores = pd.DataFrame(columns=['degree', 'mse', 'bias', 'variance', 'r2', 'mse_train'])

    model = RegressionMethods('ols')
    degrees = np.linspace(1, max_deg, max_deg)
    mse = np.zeros(max_deg)
    bias = np.zeros(max_deg)
    variance = np.zeros(max_deg)
    # r2 = np.zeros(max_deg)
    mse_train = np.zeros(max_deg)

    i = 0
    for deg in degrees:
        X = create_design_matrix(x, y, int(deg))
        print('degree: ', deg)
        resample = Resampling(X, z)

        mse[i], bias[i], variance[i], mse_train[i] = resample.bootstrap(model, n_bootstraps)
        # mse[i], bias[i], variance[i], mse_train[i] = resample.k_fold_CV(model)

        error_scores = error_scores.append({'degree': degrees[i],
                                            'mse': mse[i],
                                            'bias': bias[i],
                                            'variance': variance[i],
                                            # 'r2': r2[i],
                                            'mse_train': mse_train[i]}, ignore_index=True)
        i += 1
    #end for


    plt.plot(degrees, mse, label='test set')
    plt.plot(degrees, mse_train, label='training set')
    plt.legend()
    plt.xlabel('Model complexity [deg]')
    plt.ylabel('Mean Squared Error')
    plt.show()

    plt.plot(degrees, mse, label='mse')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='variance')
    plt.legend()
    plt.show()


    error_scores.to_csv(filename + '.csv')
    print(error_scores)
    print(error_scores['mse'].min())





def ridge_lasso_complexity_analysis(x, y, z, model_name, k = 5, max_deg=10):

    n_lambdas = 10
    model = RegressionMethods(model_name)

    lambdas = np.linspace(-10, 2, n_lambdas)
    degrees = np.linspace(1, max_deg, max_deg)
    d, l = np.meshgrid(degrees, lambdas)

    filename = 'results/' + 'error_scores_' + model_name
    error_scores = pd.DataFrame(columns=['degree', 'lambda', 'mse', 'bias', 'variance', 'mse_train'])



    min_mse = 1e100
    min_lambda = 0
    min_degree = 0

    i = 0
    for deg in degrees:
        j = 0
        X = create_design_matrix(x, y, int(deg))
        print('degree: ', deg)
        resample = Resampling(X, z)
        for lamb in lambdas:
            model.set_alpha(10**lamb)

            mse, bias, variance, mse_train = resample.bootstrap(model, n_bootstraps=10)


            error_scores = error_scores.append({'degree': deg,
                                                'log lambda': lamb,
                                                'mse': mse,
                                                'bias': bias,
                                                'variance': variance,
                                                # 'r2': r2,
                                                'mse_train': mse_train}, ignore_index=True)
            if mse < min_mse:
                min_mse = mse
                min_lambda = lamb
                min_degree = deg


            j+=1
        #end for lambdas
        i+=1
    #end for degrees

    print('min mse:', min_mse)
    print('degree:', min_degree)
    print('lambda:', min_lambda)


    error_scores.to_csv(filename + '.csv')
    # print(error_scores)


    mse_table = pd.pivot_table(error_scores, values='mse', index=['degree'], columns='log lambda')
    mse_values = mse_table.values
    mse_values = mse_values.T

    # print(mse_table)
    # print(mse_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(min_degree, min_lambda, min_mse, color='r',
                        label='Min. MSE = %0.2f, ' %min_mse +
                         r'$\log_{10}(\lambda)$ = %0.2f, ' %min_lambda +
                         'degree = %d' %min_degree)
    surface = ax.plot_surface(d, l, mse_values,
        cmap=mpl.cm.coolwarm, alpha=0.5)

    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel(r"$\log_{10}(\lambda)$")
    ax.set_zlabel("Mean Squared Error")

    ax.legend()

    plt.show()
