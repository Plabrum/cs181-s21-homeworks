import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance


    # TODO: Implement this method!
    def fit(self, X, y):
        self.mu_list = []
        self.cov_list = []
        self.pi_list = []
        self.K = len(np.unique(y, return_counts=True)[0])
        # y_trans = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
        # Y_hot = np.array([y_trans[x] for x in y])

        for k in range(self.K):
            y_count = np.unique(y, return_counts=True)[1][k]
            # print("Y counts: ", y_count)
            self.pi_list.append(y_count/len(X))
            # print("Pis: ", self.pi_list[k])
            self.mu_list.append(sum([x for (x,_y) in zip(X,y) if _y==k])/y_count)
            # print("Mu: ", self.mu_list[k])
            
            x_lis = []
            for xi, yi in zip(X, y):
                if yi == k:
                    dif = np.array([xi - self.mu_list[yi]])
                    x_lis.append(np.dot((dif).T, (dif)))

            # print("X list", x_lis)
            self.cov_list.append(sum(x_lis)/y_count)
            # Counts of each list 

        self.shared_cov = sum([self.cov_list[k]*self.pi_list[k] for k in range(self.K)])

    # TODO: Implement this method!
    def predict(self, X_pred):
        def pred(x):
            class_probs = []
            # return a 1x3 of the probabilities of each
            for k in range(self.K):
                if self.is_shared_covariance:
                    cov = self.shared_cov
                else:
                    cov = self.cov_list[k]
                d = x - self.mu_list[k]
                # print(cov)
                step1 = np.dot(np.linalg.inv(cov), d.T)
                step2 = np.exp(-0.5*np.dot(d, step1))
                class_probs.append(self.pi_list[k] * step2)
            return np.argmax(class_probs)

        return np.array([pred(x) for x in X_pred])
        
    
    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        log_likelihood = 0
        for i in range(len(X)):
            for k in range(self.K):
                if self.is_shared_covariance:
                    cov = self.shared_cov
                else:
                    cov = self.cov_list[k]

                if y[i] == k:
                    mv = mvn.pdf(x=X[i], mean=self.mu_list[k], cov=cov)
                    log_likelihood += np.log(self.pi_list[k] * mv)
        
        return log_likelihood*-1




