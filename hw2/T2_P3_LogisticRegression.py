import numpy as np
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    def __basis1(self, x): 
        return np.hstack((np.ones((x.shape[0],1)), x))

    def __softmax(self, activation):
        return np.array([(np.exp(k)/sum(np.exp(activation))) for k in activation])

    def __loss(self, y, y_pred):
        return -1*sum(sum(y*np.log(y_pred)))

    '''
    
    Fitting process:
    1. Convert the raw x (27x2) and y(27,1) into x_pred(27,3) with a basis applied and y_hot(27,3)
    2. Make a series of y_predicted (27, 3)
    3. Calculate the loss accrued by the predictions, loss = cross entropy error + l2 regularization
    4. Add the loss to the loss_list for use in the prediction 
    5. Calculate the gradient of the loss
    6. Update the weights according to the new gradients

    '''

    def __gradient(self, x, y, y_pred):
        # print("Type x: ", type(x))
        # print("Shape x: ", x.shape)
        # print("Type y: ", type(y))
        # print("Shape y: ", y.shape)
        # print("Type y_pred: ", type(y_pred))
        # print("Shape y_pred: ", y_pred.shape)

        return np.dot((y_pred - y).T, x)
        # return np.dot(x.T, (y_pred - y))

    # TODO: Implement this method!
    def fit(self, X, y):
        # Set number of epochs
        runs = 10000

        # define a loss list for visualisation
        self.loss_list = []

        # Add bias values for the x to make it a 2x matrix
        X_preds = self.__basis1(X)
        # print("X with bias value (27x3):\n", X_preds)

        # Map the star types to one-hot encoded vectors:
        y_trans = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
        Y_hot = np.array([y_trans[x] for x in y])

        # print("Y hot encoded (27x3):\n", Y_hot)

        # Set random weights:
        self.W = np.random.rand(X_preds.shape[1], Y_hot.shape[1])

        # Training for a binary regressor:
        for i in range(runs):
            Y_hats = self.__predict(X_preds)
            # print("Y hats (27x3):\n", Y_hats)
            # print("Weights: \n", self.W)
            self.loss_list.append(self.__loss(Y_hot, Y_hats))
            gradient = self.__gradient(X_preds, Y_hot, Y_hats)
            # print("Gradient, this should be a 3x3:\n", gradient)
            # reg = self.lam*self.W
            self.W = self.W - (gradient*self.eta) - self.lam*self.W

    def __predict(self, X_preds):
        # This should return a 26x3 matrix where each row is a vector of probabilities of being in 
        #  each class
        return np.array([self.__softmax(row) for row in np.dot(X_preds, self.W.T)])



    def predict(self, X):
        # This should return a 26x3 matrix where each row is a vector of probabilities of being in 
        #  each class
        X_preds = self.__basis1(X)
        one_hot_results = np.array([self.__softmax(row) for row in np.dot(X_preds, self.W.T)])
        return np.array([np.argmax(row) for row in one_hot_results])
        
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        # Starter code:

        # preds = []
        # for x in X_pred:
        #     z = np.cos(x ** 2).sum()
        #     preds.append(1 + np.sign(z) * (np.abs(z) > 0.3))
        # return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):

        # Use the loss list to draw a graph of the loss value over iterations
        plt.title(output_file)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        iteration_list = np.arange(0, len(self.loss_list), 1)

        plt.plot(iteration_list, np.array(self.loss_list), 'b', linewidth=2)
        plt.savefig('figs/'+output_file+'.png', bbox_inches='tight')
        if not show_charts: plt.close()
