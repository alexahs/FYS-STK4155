from RegressionMethods import *
from Resampling import *
from functions import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.patches import Rectangle
from tqdm import tqdm
import time
plt.style.use('ggplot')





def model_degree_analysis(x, y, z, model_name, min_deg=1, max_deg=10, n_bootstraps = 100, alpha = 0):

    dat_filename = 'results/' + 'error_scores_deg_analysis_' + model_name
    fig_filename = 'figures/' + 'deg_analysis_' + model_name
    error_scores = pd.DataFrame(columns=['degree', 'mse', 'bias', 'variance', 'r2', 'mse_train'])

    model = RegressionMethods(model_name, alpha=alpha)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)
    nDegs = len(degrees)
    mse = np.zeros(nDegs)
    bias = np.zeros(nDegs)
    variance = np.zeros(nDegs)
    r2 = np.zeros(nDegs)
    mse_train = np.zeros(nDegs)



    min_mse = 1e100
    min_r2 = 0
    min_deg = 0
    i = 0
    for deg in tqdm(degrees):
        X = create_design_matrix(x, y, int(deg))
        # print('degree: ', deg)
        resample = Resampling(X, z)

        mse[i], bias[i], variance[i], mse_train[i], r2[i] = resample.bootstrap(model, n_bootstraps)
        # mse[i], bias[i], variance[i], mse_train[i] = resample.k_fold_CV(model)

        error_scores = error_scores.append({'degree': degrees[i],
                                            'mse': mse[i],
                                            'bias': bias[i],
                                            'variance': variance[i],
                                            'r2': r2[i],
                                            'mse_train': mse_train[i]}, ignore_index=True)
        if mse[i] < min_mse:
            min_mse = mse[i]
            min_r2 = r2[i]
            min_deg = deg

        i += 1
    #end for


    ID = '006'

    plt.plot(degrees, mse, label='test set')
    plt.plot(degrees, mse_train, label='training set')
    # plt.plot(min_deg, min_mse, 'ro', label='Min. MSE = %0.3f, ' %min_mse +
    #                             'degree = %d' %min_deg)
    plt.legend()
    plt.xlabel('Model complexity [degree]')
    plt.ylabel('Mean Squared Error')
    plt.savefig(fig_filename + '_test_train_' + ID + '.pdf')
    plt.show()

    plt.plot(degrees, mse, label='mse')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='variance')
    plt.xlabel('Complexity')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(fig_filename + '_bias_variance_' + ID + '.pdf')
    plt.show()


    error_scores.to_csv(dat_filename + '.csv')
    # print(error_scores)
    # print(error_scores['mse'].min())
    print('min mse:', min_mse)
    print('r2:', min_r2)
    print('deg:', min_deg)



def ridge_lasso_complexity_analysis(x, y, z, model_name, min_deg=1, max_deg=10):

    n_lambdas = 13
    model = RegressionMethods(model_name)



    lambdas = np.linspace(-10, 2, n_lambdas)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)

    dat_filename = 'results/' + 'error_scores_' + model_name
    fig_filename = 'figures/' + 'min_mse_meatmap_' + model_name
    error_scores = pd.DataFrame(columns=['degree', 'lambda', 'mse', 'bias', 'variance', 'mse_train'])


    min_mse = 1e100
    min_lambda = 0
    min_degree = 0
    min_r2 = 0

    i = 0
    for deg in tqdm(degrees):
        j = 0
        X = create_design_matrix(x, y, int(deg))
        # print('degree: ', deg)
        resample = Resampling(X, z)
        for lamb in tqdm(lambdas):
            model.set_alpha(10**lamb)

            mse, bias, variance, mse_train, r2 = resample.bootstrap(model, n_bootstraps=10)


            error_scores = error_scores.append({'degree': deg,
                                                'log lambda': lamb,
                                                'mse': mse,
                                                'bias': bias,
                                                'variance': variance,
                                                'r2': r2,
                                                'mse_train': mse_train}, ignore_index=True)
            if mse < min_mse:
                min_mse = mse
                min_lambda = lamb
                min_degree = deg
                min_r2 = r2


            j+=1
        #end for lambdas
        i+=1
    #end for degrees

    print('min mse:', min_mse)
    print('min r2:', min_r2)
    print('degree:', min_degree)
    print('lambda:', min_lambda)


    error_scores.to_csv(dat_filename + '.csv')


    mse_table = pd.pivot_table(error_scores, values='mse', index=['degree'], columns='log lambda')
    idx_i = np.where(mse_table == min_mse)[0]
    idx_j = np.where(mse_table == min_mse)[1]


    fig = plt.figure()
    ax = sns.heatmap(mse_table, annot=True, fmt='.2g', cbar=True, linewidths=1, linecolor='white',
                            cbar_kws={'label': 'Mean Squared Error'})
    ax.add_patch(Rectangle((idx_j, idx_i), 1, 1, fill=False, edgecolor='red', lw=3))

    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel("Complexity")
    ax.set_ylim(len(degrees), 0)
    # plt.savefig('test.pdf')
    plt.show()



def confidence_interval_ols(X, z, betas):
    cov = np.var(z)*np.linalg.pinv(X.T.dot(X))

    std_betas = np.sqrt(np.diag(cov))

    CI = 1.96*std_betas

    # print(np.sort(CI))
    # print(np.var(z))

    plt.xticks(np.arange(0, len(betas), step=1))

    plt.errorbar(range(len(betas)), betas, CI, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96\dot \sigma$')
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.grid()
    plt.show()



"""
mse_values = mse_table.values
mse_values = mse_values.T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(min_degree, min_lambda, min_mse, color='r',
                    label='Min. MSE = %0.3f, ' %min_mse +
                     r'$\log_{10}(\lambda)$ = %0.2f, ' %min_lambda +
                     'degree = %d' %min_degree)
surface = ax.plot_surface(d, l, mse_values,
    cmap=mpl.cm.coolwarm, alpha=0.5)

ax.set_xlabel("Complexity")
ax.set_ylabel(r"$\log_{10}(\lambda)$")
ax.set_zlabel("Mean Squared Error")

ax.legend()
ID = '003'
plt.savefig(fig_filename + '_' + ID + '.pdf')

plt.show()
"""
