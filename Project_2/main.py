import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from functions import *
from LogisticRegression import *



def main():
    filename = 'data/default_of_credit_card_clients.xls'

    X, y = load_CC_data(filename)

    X_train, X_test, y_train, y_test = scale_data_split(X, y)



    model = LogisticRegression(n_epochs=1000, size_minibatch=512)
    model.fit(X_train, y_train)
    pred_model = model.predict(X_test)

    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X_train, y_train)
    # pred_skl = clf.predict(X_test)

    accuracy_model = accuracy_score(pred_model, y_test)
    # accuracy_skl = accuracy_score(pred_skl, y_test)

    print('accuracy model:', accuracy_model)
    # print('accuracy skl:', accuracy_skl)





if __name__ == '__main__':
    main()
