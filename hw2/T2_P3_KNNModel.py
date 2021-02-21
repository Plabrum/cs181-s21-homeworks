import numpy as np
import statistics

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def __distance(self, star1, star2):
        mag = ((star1[0] - star2[0])/3)**2
        temp = (star1[1] - star2[1])**2
        return mag + temp

    def predict(self, X_pred):
        # x is 27x2 and y is a 27x1

        # Map the star types to one-hot encoded vectors:
        y_trans = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
        Y_hot = np.array([y_trans[x] for x in self.y])
        y_hats =[]
        for test_x in X_pred:
            # Maybe use one hot encoding?
            nearest_list = []
            for star, cl in zip(self.X, Y_hot):
                distance = self.__distance(test_x, star)
                nearest_list.append((distance, cl))
            
            # sort list
            nearest_list.sort(key=(lambda fs: fs[0]), reverse=False)
            # print("nearest\n",nearest_list)
            k_nearest = nearest_list[:self.K]

            y_hat = np.argmax(sum(np.array([y for _,y in k_nearest])))
            # print(y_hat)
            # y_hat = statistics.mode(y_array)
            y_hats.append(y_hat)

        return np.array(y_hats)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y