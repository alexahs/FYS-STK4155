import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from LogisticRegression import *


def load_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


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
    y = np.delete(y, outlier_rows, axis=0)


    onehotencoder = OneHotEncoder(categories="auto")
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, [1, 2, 3])])


    X = preprocessor.fit_transform(X)

    # y = onehotencoder.fit_transform(y)


    return X, np.ravel(y)


def scale_data_split(X, y, test_size = 0.2, which_dataset = 'creditcard'):



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    sc = StandardScaler()
    scaler = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('standardscaler', sc, list(range(9, X_train.shape[1])))])


    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    # filename = 'data/default_of_credit_card_clients.xls'
    # X, y = load_CC_data(filename)



    pass
