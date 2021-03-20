# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
# sns.set_style(style="whitegrid")

# This line loads the images for you. Don't change it!
# pics = np.load("data/images.npy", allow_pickle=False)
small_dataset = np.load("data/small_dataset.npy")
# small_labels = np.load("../data/small_dataset_labels.npy").astype(int)
large_dataset = np.load("data/large_dataset.npy")

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
    
    def __make_one_hot(self, array):
        one_hot = np.zeros(array.shape)
        min_pos = np.array([np.argmin(row) for row in array])
        one_hot[np.arange(array.shape[0]), min_pos] = 1
        return one_hot 

    def __distance(self, p1, p2):
        # calulate the l2 distance between points, i.e. euclidean
        # each is a 1xD matrix
        return sum(pow(p1-p2, 2))
       
    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        # 1 Find the cluster centers (cluster mean)
        def cluster_mean(resp, data):
            # for each datapoint in cluster c,
            # sum it's vals and divide by number in class
            class_sums = np.zeros((resp.shape[1], data.shape[1]))
            for point, res in zip(data, resp):
                class_sums[np.argmax(res)] += point
            # count up the number of datapoints in each class    
            class_counts = np.sum(resp, axis=0)
            
            # class_mean = np.divide(class_sums, class_counts)
            class_mean = []
            for class_sum, class_count in zip(class_sums, class_counts):
                class_mean.append(np.divide(class_sum, class_count))
                
            class_mean = np.array(class_mean)
            return class_mean 

        # 2 adjust reponsibility vectors
        def adjust_resp(mu, data):
            # resp dims: NxC
            # mu dims:
            positions = []
            for point in data:
                distances = []
                for class_mean in mu:
                    distances.append(self.__distance(class_mean, point))
                one_ht = np.zeros(mu.shape[0])
                one_ht[np.argmin(distances)] = 1
                positions.append(one_ht)
                
            return np.array(positions)

        # 3 calculate the loss
        def loss_func(data, mu, resp):
            loss_list = []
            # Loop through each datapoint in the dataset
            for datapoint, res in zip(data, resp):
                cluster = np.argmax(res)
                loss_list.append(self.__distance(datapoint, mu[cluster]))
            return sum(loss_list)

        # initialisation of cluster asignment i.e. responsibility vectors
        self.resp = self.__make_one_hot(np.random.randn(X.shape[0], self.K))
        # Set inital mu
        self.mu = np.random.randn(self.K, X.shape[1])
        self.loss_list = []

        # record the losses at each epoch
        converge = False
        epoch = 0
        while not converge:
            # 1 calc the mean of the clusters
            self.mu = cluster_mean(self.resp, X)
            
            # 2 Calc the loss
            current_loss = loss_func(X, self.mu, self.resp)

            # 3 adjust the responsibility vectors
            new_resp = adjust_resp(self.mu, X)
            if np.array_equal(new_resp, self.resp):
                converge = True
            self.resp = new_resp
            epoch += 1
            self.loss_list.append(current_loss)
            print("Epoch:", epoch)


    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu

# gets images in right order
def reshuffle(arr):
    nw = []
    for a,b,c in zip(arr[:10], arr[10:20], arr[20:30]):
        nw.append(a)
        nw.append(b)
        nw.append(c)
    return nw

def p2_1():
    KMeansClassifier = KMeans(K=10)
    KMeansClassifier.fit(large_dataset)

    fig, ax = plt.subplots()
    ax.plot(np.array(KMeansClassifier.loss_list));
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_xlabel('Iterations', fontsize=15)
    ax.set_title('K-means objective function', fontsize=20)
    plt.show()

    # This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
    # plt.figure()
    # plt.imshow(KMeansClassifier.mu[0].reshape(28,28), cmap='Greys_r')
    # plt.show()

save_kmcs = []

def p2_2():
    for k in [5,10,20]:
        for restart in range(5):
            KMC = KMeans(K=k)
            KMC.fit(large_dataset)
            save_kmcs.append(KMC)
            print("Converged1!")

    fig, ax = plt.subplots(figsize=(8, 6))

    los = np.array(lossls).reshape(3,5)
    klis = [5,10,20]

    for num,ls in enumerate(los):
        x = klis[num]
        stdev = np.std(ls)
        for y in ls:
    #         ax.scatter(x, y, color='b', marker="x")
            plt.errorbar(x, y, yerr=stdev, color='b', fmt='x')
        
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_xlabel('K Values', fontsize=15)
    ax.set_title('K-means objective value for K=5,10,20', fontsize=20)
    plt.show()

def p2_3():
    imgs= []
    for kmc in save_kmcs:
        for img in kmc.mu:
            imgs.append(img)
    w = 28
    h = 28
    fig = plt.figure(figsize=(9, 26))
    columns = 3
    rows = 10
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(columns*rows):
        img = imgs[i].reshape(28,28)
        ax.append( fig.add_subplot(rows, columns, i+1) )
        plt.imshow(img, cmap='Greys_r')

    plt.show()  # finally, render the plot

def p2_4():
    avrg = np.sum(large_dataset, axis=0)/large_dataset.shape[1]
    norm = np.array([av if av != 0 else 1 for av in avrg])
    normalized = large_dataset/norm

    save_kmcs = []
    k = 10
    for restart in range(3):
        KMC = KMeans(K=k)
        KMC.fit(normalized)
        save_kmcs.append(KMC)
        print("Converged!")
    imgs= []
    for kmc in save_kmcs:
        for img in kmc.mu:
            imgs.append(img)

    w = 28
    h = 28
    fig = plt.figure(figsize=(9, 26))
    columns = 3
    rows = 10
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(columns*rows):
        img = imgs[i].reshape(28,28)
        ax.append( fig.add_subplot(rows, columns, i+1) )
        plt.imshow(img, cmap='Greys_r')

    plt.show()  # finally, render the plot


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        
    def fit(self, dataset):
        cll = [point.reshape(1,-1) for point in dataset]

        while len(cll) > 10:
            l = len(cll)
            min_i, min_i = 0, 0
            minimum_distance = np.inf
            
            # Find the i and j of the closest two clusters
            for i in range(l):
                for j in range(l):
                    if i == j:
                        continue
                    dist = self.linkage(cll[i],cll[j])
                    if dist < minimum_distance:
                        # found a closer cluster
                        minimum_distance = dist
                        min_i, min_j = i, j

            # print("merging:", min_i,min_j)
            cll[min_i] = np.concatenate((cll[min_i], cll.pop(min_j)), axis=0)
            print("list length", len(cll))

        return [np.sum(cl, axis=0)/len(cl) for cl in cll]

def min_link(cl1, cl2):
    return np.max(cdist(cl1, cl2))

def max_link(cl1, cl2):
    return np.min(cdist(cl1, cl2))

def ce_link(cl1, cl2):
    a = np.sum(cl1, axis=0).reshape(1,-1) / len(cl1)
    b = np.sum(cl2, axis=0).reshape(1,-1) / len(cl1)
    return cdist(a, b)[0][0]


def p2_5_disp(imgs):
    w = 28
    h = 28
    fig = plt.figure(figsize=(9, 26))
    columns = 3
    rows = 10
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(columns*rows):
        img = imgs[i].reshape(28,28)
        ax.append( fig.add_subplot(rows, columns, i+1) )
        plt.imshow(img, cmap='Greys_r')
    plt.show()  # finally, render the plot

def p2_5():

    ml_hac = HAC(min_link)
    ml_mus = ml_hac.fit(small_dataset)

    ma_hac = HAC(max_link)
    ma_mus = ma_hac.fit(small_dataset)

    av_hac = HAC(av_link)
    av_mus = av_hac.fit(small_dataset)

    p2_5_disp(reshuffle(ml_mus + ma_mus + av_mus))

p2_5()

