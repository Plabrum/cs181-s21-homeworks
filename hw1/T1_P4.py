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

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false

# Nothing fancy for outputs.
Y = republican_counts
# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
# Sunspot pre_processing
cut_sunspot_counts = sunspot_counts[:13]
grid_sunspot_counts = np.linspace(min(cut_sunspot_counts), max(cut_sunspot_counts), 200)
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
    basis_x_list = [np.ones(xx.shape)]
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
     
    def apply_trans(basis_func, j_list):
        for j in j_list:
            # print("doing in j list: ", j)
            def part(x):
                return basis_func(x,j)
            basis_x =[]
            for i in xx:
                basis_x.append(part(i))
            basis_x_list.append(np.array(basis_x))

    # Perform basis functions
    if part == "a":

        j_list = list(range(1,6,1))
        def basis_func(x,j):
            return pow(x,j)

        apply_trans(basis_func, j_list)
    
    if part == "b":

        j_list = list(range(1960,2015,5))
        def basis_func(x,j):
            return np.exp((-1*((x-j)**2))/25)

        apply_trans(basis_func, j_list)
   
    if part == "c":

        j_list = list(range(1,6,1))
        def basis_func(x,j):
            return np.cos(x/j)

        apply_trans(basis_func, j_list)

    if part == "d":

        j_list = list(range(1,26,1))
        def basis_func(x,j):
            return np.cos(x/j)

        apply_trans(basis_func, j_list)

    basis_x_array = np.array(basis_x_list).T
#     print("array shape of part "+part+": ", basis_x_array.shape)
    return basis_x_array

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
    X_new_grid = make_basis(grid_years, part, is_years=True)
    # This does the linear regression
    w = find_weights(X_new_basis,Y)
    # This creates a series of predicted y_hat values to draw the best-fit line
    grid_Yhat  = np.dot(X_new_grid, w)
    plt.plot(grid_years, grid_Yhat, '-')
    
    # TODO: plot and report sum of squared error for each basis
    print("L2 Loss: ", sum((Y - np.dot(X_new_basis,w))**2))

def plot_q1():
# basis_list = ["a", "b", "c", "d"]
    basis_list=["a", "b", "c", "d"]
    for basis in enumerate(basis_list):
        print("computing basis: ", basis[1])
        plot_q1_pt(*basis)

def plot_q2_pt(figure, part):
    # Create a single a figure
    plt.figure(figure)
    # Add data points
    plt.plot(years, republican_counts, 'o')
    plt.xlabel("Sunspot Count")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Republicans vs Sunspots with "+ part +" basis function applied")

    # Perform basis transformation
    X_new_basis = make_basis(sunspot_counts, part, is_years=False)
    X_new_grid = make_basis(grid_sunspot_counts, part, is_years=False)
    # This does the linear regression
    w = find_weights(X_new_basis,Y)
    # This creates a series of predicted y_hat values to draw the best-fit line
    grid_Yhat  = np.dot(X_new_grid, w)
    plt.plot(grid_years, grid_Yhat, '-')
    
    # TODO: plot and report sum of squared error for each basis
    print("L2 Loss: ", sum((Y - np.dot(X_new_basis,w))**2))

def plot_q2():
    basis_list=["a", "c", "d"]
    for basis in enumerate(basis_list):
        print("computing basis: ", basis[1])
        plot_q2_pt(*basis)

plot_q1()
plt.show()
plot_q2()
plt.show()