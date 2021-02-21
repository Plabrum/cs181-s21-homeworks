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

    def __gradient(self, x, y, y_pred):
        return np.dot((y_pred - y).T, x)

    # TODO: Implement this method!
    def fit(self, X, y):
        # Set number of epochs
        runs = 10000
        # define a loss list for visualisation
        self.loss_list = []
        # Add bias values for the x to make it a 2x matrix
        X_preds = self.__basis1(X)
        # Map the star types to one-hot encoded vectors:
        y_trans = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
        Y_hot = np.array([y_trans[x] for x in y])
        # Set random weights:
        self.W = np.random.rand(X_preds.shape[1], Y_hot.shape[1])

        # Training for a binary regressor:
        for i in range(runs):
            Y_hats = self.__predict(X_preds)
            self.loss_list.append(self.__loss(Y_hot, Y_hats))
            gradient = self.__gradient(X_preds, Y_hot, Y_hats)
            self.W = self.W - (gradient*self.eta) - self.lam*self.W

    def __predict(self, X_preds):
        return np.array([self.__softmax(row) for row in np.dot(X_preds, self.W.T)])

    def predict(self, X):
        X_preds = self.__basis1(X)
        one_hot_results = np.array([self.__softmax(row) for row in np.dot(X_preds, self.W.T)])
        return np.array([np.argmax(row) for row in one_hot_results])

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
