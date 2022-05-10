from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

#The function get_model_parameters returns the parameters of xgboost model
def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params

#Sets the parameters of a sklean LogisticRegression model
def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    print(params)
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

#Initializes the model parameters that the Flower server will ask for
def set_initial_params(model: LogisticRegression):

    n_classes = 2  # MNIST has 2 classes
    n_features = 29  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data() -> Dataset:
    data=pd.read_csv('X_train.csv')
    X=data.drop("loan_default",1)
    y=data.loan_default
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
    return (x_train, y_train), (x_test, y_test)

#save the data
def save_data(model):
    filename = 'finalized_model.cbm'
    pickle.dump(model, open(filename, 'wb'))
    print('**************hellopickle!****************')
    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)
    return 0

#Shuffles data and its label
def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

#Splits datasets into a number of partitions
def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )