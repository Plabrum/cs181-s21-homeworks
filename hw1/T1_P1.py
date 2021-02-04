import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])


# def mahalanobis(a,d):


def kernel(x, x_prime, W):
    #  a and b are 2d vectors stored in a np array
    z = x - x_prime
    ztwz = (z.T)@(W@z)
    distance = np.exp(-1*ztwz)
    return distance

def kernelized_regression(test_x, W, alt_data):

    top_list =[]
    bottom_list = []
    for value in alt_data:
        # current x and y values
        x_n = np.array([value[0], value[1]])
        y_n = value[2]
        # Use our mahalanobis distance kernel to gauge distance to other x vals
        distance = kernel(test_x, x_n, W)
        # add the 
        top_list.append(distance*y_n)
        bottom_list.append(distance)

    y_hat = sum(top_list)/sum(bottom_list)

    return y_hat

def compute_loss(W):

    # losses for each y:
    losses = []
    for i, value in enumerate(data):
        
        # Maybe only search over data not including the test case?
        alt_data = data[:i] + data[i+1:]

        y = value[2]
        # construct a 2d vector for each set of x1 and x2 set of values
        test_x = np.array([value[0], value[1]])
        # This is the predicted y value given the above x vector
        y_hat = kernelized_regression(test_x, W, alt_data)
        # Ths is the residual comparing the predicted y value with the actual y value
        residual = y - y_hat
        # Add square the residual and add it to the loss list
        losses.append(pow((residual), 2))
    
    # Sum the loss list and scale by 1/2
    loss = sum(losses)
    return loss

# Need to remove the x_star from the test set


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))