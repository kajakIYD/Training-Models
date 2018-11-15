# Linear regression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time


def generate_noised_data(a, b, variance_x, variance_y, N):
    X = variance_x * np.random.rand(N, 1)
    Y = a * X + b + variance_y * np.random.randn(N, 1)
    return X, Y


def generate_true_data(a, b, X):
    Y = a * X + b
    return Y


def generate_Y_pred_lin(params_vec, X):
    return params_vec[0] * X + params_vec[1]
    


def plot_true_vs_predicted_values(X, Y_true, Y_pred, Y_noised, title):
    plt.figure()
    plt.plot(X, Y_true)
    plt.plot(X, Y_pred)
    plt.scatter(X, Y_noised)
    plt.title(title)
    plt.show()


def plot_items_of_dictionary(dictionary, title):
    lists = sorted(dictionary.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.show()
    
    
def swap(params):
    params[0], params[1] = params[1], params[0]
    return params


a = 2
b = 5
variance_x = 5 # Describe how long is range along x-axis
variance_y = 1 # How much Your data values are noised

times_dict = dict()
theta_dict = dict()
scikit_linear_regression_dict = dict()
mean_squared_error_dict = dict()

N_vector = [10, 100, 1000, 10000, 100000]

for N in N_vector:
    X, Y = generate_noised_data(a, b, variance_x, variance_y, N)
    Y_true = generate_true_data(a, b, X)
    
    # The Normal Equation - optimal solution in the meaning of MSE

    start_time = time.time()
    
    X_b = np.c_[np.ones((N, 1)), X] # this line adds one column "before" whole X content
    # I.e:
    #      |2|          |1 2|
    # x =  |2| ->  x =  |1 2|
    #      |2|          |1 2|

    theta_dict['NEq_' + str(N)] = swap(list(np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)))
    times_dict['NEq_' + str(N)] = time.time() - start_time
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, Y)
    scikit_linear_regression_dict[str(N)] = [lin_reg.coef_, lin_reg.intercept_]
    
    Y_pred = generate_Y_pred_lin(theta_dict['NEq_' + str(N)], X)
    plot_true_vs_predicted_values(X, Y_true, Y_pred, Y, "Linear regression")
    
    mean_squared_error_dict['NEq_' + str(N)] = mean_squared_error(Y_true, Y)
    
print(theta_dict)
print(times_dict)
print(scikit_linear_regression_dict)
print(mean_squared_error_dict)

dictionaries = [times_dict, mean_squared_error_dict]
titles = ['times', 'mean_squared_error_dict']

for dictionary, title in zip(dictionaries, titles):
    plot_items_of_dictionary(dictionary, title)

    
