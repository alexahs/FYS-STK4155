import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from LogisticRegression import *


def load_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    print(df.columns)

    # print(df.loc[:, df.columns == 'LIMIT_BAL'].values)



    outlier_gender1 = np.where(X[:,1] < 1)[0]
    outlier_gender2 = np.where(X[:,1] > 2)[0]

    outlier_education1 = np.where(X[:,2] < 1)[0]
    outlier_education2 = np.where(X[:,2] > 4)[0]

    outlier_marital1 = np.where(X[:,3] < 1)[0]
    outlier_marital2 = np.where(X[:,3] > 3)[0]

    inds = np.concatenate((outlier_gender1,
                           outlier_gender2,
                           outlier_education1,
                           outlier_education2,
                           outlier_marital1,
                           outlier_marital2))


    outlier_rows = np.unique(inds)

    X = np.delete(X, outlier_rows, axis=0)

    # print(X[:, 4])
    # plt.hist(X[:, 4])
    # plt.show()

    onehotencoder = OneHotEncoder(categories="auto")
    sc = StandardScaler()
    scale_inds = np.concatenate((np.array([0, 4]), np.array(range(11, 23))))
    onehot_inds = [1, 2, 3]

    X = ColumnTransformer([("StandardScaler", sc, scale_inds),],
                            remainder="passthrough").fit_transform(X)


    print(X.shape)

    X = ColumnTransformer([("", onehotencoder, [1]),],
                            remainder="passthrough").fit_transform(X)


    # print(X[:, 4])
    # plt.hist(X[:, 4])
    # plt.show()

    # X = ColumnTransformer([("onehot", onehotencoder, [1, 2, 3]),],
    #                          remainder="passthrough").fit_transform(X)



    # print(X[:, 9])
    # plt.plot(X[:, 9])
    # plt.show()

    print(X.shape)

    for i in range(X.shape[1]):
        plt.hist(X[:, i], label=str(i))
        plt.legend()
        plt.show()

    return X, y






if __name__ == '__main__':
    filename = 'data/default_of_credit_card_clients.xls'
    X, y = load_CC_data(filename)
