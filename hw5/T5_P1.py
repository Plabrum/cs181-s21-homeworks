import time
import numpy as np
import pandas as pd

class EM_Geometric(object):
    def __init__(self, num_clusters, min_step=1e-4, probs=None, priors=None):
        '''
        Initialize the EM algorithm based on given parameters

        Parameters
        ----------
        num_clusters : int
            The number of labels the EM algorithm should form clusters for
        min_step : int
            The minimum steps required for termination of the algorithm
        probs : np.array
            The initial parameter p for each label (geometric distribution)
        priors : np.array
            The initial mixing proportion of each label (aka theta)
        '''
        self.num_clusters = num_clusters
        self.min_step = min_step

        if probs is None:
            self.probs = np.random.random(self.num_clusters)
        else:
            self.probs = probs

        if priors is None:
            self.pis = np.ones(self.num_clusters) / self.num_clusters
        else:
            self.pis = priors

    def cluster(self, data, max_iters=50):
        '''
        Train on the given data to form clusters for the labels

        Parameters
        ----------
        data : pd.DataFrame
            The feature values of the data set
        '''
        self.data = data
        start_time = time.time()

        # Initialize the soft cluster assignments q for each data point to each label
        self.q = np.asarray(np.empty((len(self.data), self.num_clusters), dtype=float))

        num_iters = 0

        # Current and previous log-likelihoods
        curr_ll = 0
        prev_ll = 0

        # While termination conditions have not been met, run the algorithm
        while num_iters <= 1 or (curr_ll - prev_ll > self.min_step and num_iters < max_iters):
            # Perform the expectation step
            self.e_step()
            # Perform the maximization step
            self.m_step()

            num_iters += 1

            # Calculate and compare log-likelihoods
            prev_ll = curr_ll
            curr_ll = self.loglikelihood()
            print(f'Iteration {num_iters}, log-likelihood: {curr_ll}')

        total_time = time.time() - start_time
        print(f'Complete, time elapsed: {total_time} seconds')

        # Sort the clusters and re-arrange the soft labels
        i = np.argsort(self.probs)
        self.probs = np.sort(self.probs)
        self.q = self.q[:, i]

    def pmf(self, x, p):
        # each x for a given n, each p for a given k
        return np.power((1-p), x - 1)*p

    def loglikelihood(self):
        '''
        Calculate the current log-likelihood given data and soft assignments

        Returns
        -------
        ll : float
            The log-likelihood of current set of labels (according to a geometric distribution)
        '''

        N, K = self.q.shape
        
        loss_list = []
        for n in range(N):
            for k in range(K):
                loss_list.append( self.q[n][k] * np.log(self.pmf(self.data[n],self.probs[k]) * self.pis[k]) )

        return sum(loss_list)

    def e_step(self):
        '''
        Run the expectation step, calculating the soft assignments based on current parameter estimates
        '''
        N, K = self.q.shape
        for n in range(N):
            row = np.array([self.pmf(self.data[n], self.probs[k])*self.pis[k] for k in range(K)])
            self.q[n] = row/np.sum(row)
            assert(np.round(np.sum(self.q[n]), 2) == 1.0)
        
        # print("sum of qs: ", np.sum(self.q, axis=1))

    def m_step(self):
        '''
        Run the maximization step, calculating parameter estimates based on current soft assignments
        '''
        self.pis = np.sum(self.q, axis=0) / len(self.q)
        self.probs = (np.sum(self.q, axis=0) / np.dot(self.q.T, self.data)).flatten()
        print("sum of probs:", np.sum(self.probs))
        # assert(np.round(np.sum(self.probs), 2) == 1.0)

    def get_labels(self):
        '''
        Return the label with highest probability among the soft assignments for each data point
        '''
        return np.array([np.argmax(q) for q in np.array(self.q)])

def generate_geom_data(num_data, cluster_probs):
    '''
    Generate an equal number of data points for each cluster, based on geometric distribution
    '''
    data = np.array([])
    labels = np.array([], dtype=int)
    for i, prob in enumerate(cluster_probs):
        data = np.concatenate((data, np.random.geometric(p=prob, size=num_data)))
        labels = np.concatenate((labels, np.full(num_data, i, dtype=int)))
    return data, labels

def main():
    # TODO: Toggle these between 10 / 1000 and [0.1, 0.5, 0.9] / [0.1, 0.2, 0.9]
    num_data = 1000
    cluster_probs = [0.1, 0.5, 0.9]

    # Do not edit the below code, it is to help you run the algorithm
    # -------------------------------------------------------------------------------
    data, labels = generate_geom_data(num_data=num_data, cluster_probs=cluster_probs)

    em_algo = EM_Geometric(num_clusters=3)
    em_algo.cluster(data)
    print(f'Final probabilities: {em_algo.probs}')

    accuracy = np.equal(em_algo.get_labels(), labels).sum() / len(labels)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()