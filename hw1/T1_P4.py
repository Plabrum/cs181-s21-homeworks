#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

def plot_data():
    # Plot the data.
    plt.figure(1)
    plt.plot(years, republican_counts, 'o')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.figure(2)
    plt.plot(years, sunspot_counts, 'o')
    plt.xlabel("Year")
    plt.ylabel("Number of Sunspots")
    plt.figure(3)
    plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# Nothing fancy for outputs.
Y = republican_counts
# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
# This adds an x_0 so that we can have an offset w_0
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))


# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false

def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    basis_x_list = []
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
     
    def apply_trans(basis_func, j_list):
        for j in j_list:
            def part(x):
                return basis_func(x,j)
            basis_x =[]
            for i in xx:
                basis_x.append(part(i))
            basis_x_list.append((j, np.array(basis_x)))

    # Perform basis functions
    if part == "a":

        j_list = list(range(1,6,1))
        def basis_func(x,j):
            return pow(x,j)

        apply_trans(basis_func, j_list)
        
    else:
        basis_x_list.append((1, [xx]))

    # elif part == "b":
    #     basis_func = exp()
    #     j_list = np.arange(1960,2015,5)

    return basis_x_list# Nothing fancy for outputs.

def plot_q1_pt(figure, part):
    # Create a single a figure
    plt.figure(figure)
    # Add data points
    plt.plot(years, republican_counts, 'o')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Republicans vs Year with "+ part +" basis function applied")

    # Perform basis transformation
    X_new_basis = make_basis(years, part, is_years=True)
    
    # Add a new plot for each of the j basis cases
    for j, j_basis in X_new_basis:
        
        # X_matrix = np.vstack((np.ones(years.shape), years)).T
        # This does the linear regression
        add_ones = np.vstack((np.ones(j_basis.shape), j_basis)).T
        w = find_weights(add_ones,Y)
        # This creates a series of predicted y_hat values to draw the best-fit line
        grid_Yhat  = np.dot(grid_X.T, w)
        plt.plot(grid_years, grid_Yhat, '-', label=str(j))

def plot_q1():
    # basis_list = ["a", "b", "c", "d"]
    basis_list=["a"]
    for basis in enumerate(basis_list):
        print("computing basis: ", basis[1])
        plot_q1_pt(*basis)

# TODO: plot and report sum of squared error for each basis
L2_loss = 0

plot_q1()
plt.show()