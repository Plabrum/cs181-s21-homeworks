#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    # TODO: your code here
    return y_df

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    # TODO: your code here
    return y_df


def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    y_pred_list = []
    
    for value in df.itertuples():
        x = np.array([value[1], value[2]])
        alt_data = list(df[["x1", "x2", "y"]][:value[0]].append( df[["x1", "x2","y"]][value[0]+1:], ignore_index=True).to_records(index=False))
        y_pred_list.append(kernelized_regression(alpha, x, alt_data))
        
    return pd.Series(y_pred_list)

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    # TODO: your code here
    y_pred_list = []
    
    for value in df.itertuples():
        x = np.array([value[1], value[2]])
        alt_data = list(df[["x1", "x2", "y"]][:value[0]].append( df[["x1", "x2","y"]][value[0]+1:], ignore_index=True).to_records(index=False))
        y_pred_list.append(knn_regression(k, x, alt_data))
        
    return pd.Series(y_pred_list)


def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

def knn_regression(k, test_x, alt_data):
    W = alpha * np.array([[1., 0.], [0., 1.]])
    def kernel(x, x_prime, W):
        z = x - x_prime
        return np.exp(-1*((z.T)@(W@z)))
    
    # construct an ordered of distance and y values list
    nearest_list = []
    for value in alt_data:
        distance = kernel(test_x, np.array([value[0], value[1]]), W)
        nearest_list.append((distance, value[2]))

    # sort the list
    nearest_list.sort(key=(lambda fs: fs[0]), reverse=True)
    # take only the top k values
    k_nearest = nearest_list[:k]
    y_hat = (1/k)*sum([y for _,y in k_nearest])
    return y_hat

def kernelized_regression(alpha, test_x, alt_data):
    W = alpha * np.array([[1., 0.], [0., 1.]])
    def kernel(x, x_prime, W):
        z = x - x_prime
        return np.exp(-1*((z.T)@(W@z)))
    
    top_list =[]
    bottom_list = []
    for value in alt_data:
        distance = kernel(test_x, np.array([value[0], value[1]]), W)
        top_list.append(distance*value[2])
        bottom_list.append(distance)

    y_hat = sum(top_list)/sum(bottom_list)
    return y_hat

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
