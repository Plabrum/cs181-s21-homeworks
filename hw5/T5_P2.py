# Starter code for use with autograder.
import numpy as np
import matplotlib.pyplot as plt


# Lists for storing values for images
pc_list = []
v_list = []

def get_cumul_var(mnist_pics, num_leading_components=500):

    """
    Perform PCA on mnist_pics and return cumulative fraction of variance
    explained by the leading k components.

    Returns:
        A (num_leading_components, ) numpy array where the ith component
        contains the cumulative fraction (between 0 and 1) of variance explained
        by the leading i components.

    Args:

        mnist_pics, (N x D) numpy array:
            Array containing MNIST images.  To pass the test case written in
            T5_P2_Autograder.py, mnist_pics must be a 2D (N x D) numpy array,
            where N is the number of examples, and D is the dimensionality of
            each example.

        num_leading_components, int:
            The variable representing k, the number of PCA components to use.
    """
    # Standardize
    # mnist_pics = (mnist_pics - np.mean(mnist_pics))/ np.std(mnist_pics)
    mnist_pics = (mnist_pics - np.mean(mnist_pics, axis=0))
 
    S = np.cov(mnist_pics.T)
    values, vectors = np.linalg.eig(S)

    value_list = list(values.real)
    vector_list = list(vectors.T.real)

    for _ in range(num_leading_components):
        # Find the position of the  max eigen value
        max_pos = np.argmax(value_list)

        # Save the eigenvalue and eigenvector
        pc_list.append(vector_list[max_pos])
        v_list.append(value_list[max_pos])

        del value_list[max_pos]
        del vector_list[max_pos]

    return np.cumsum(v_list) / np.sum(v_list)


def p2_1():
    def plot_vals(ls, xl, yl, t):
        font = {'color':  'darkblue'}
        plt.title(t, fontdict=font)
        plt.xlabel(xl, fontdict=font)
        plt.ylabel(yl, fontdict=font)
        plt.plot(ls)

    # Plot the values for the 500 most signifigant aigenvectors in order from most signifgant to least
    plot_vals(v_list, t="500 most signifigant Values", yl="Eigenvalues", xl="Signifigance")
    plt.tight_layout()
    plt.savefig("plots/2_1a.png")
    plt.show()

    plot_vals(cum_var, t="Cumulative Variance", yl="Variance", xl="500 most signifigant components")
    plt.tight_layout()
    plt.savefig("plots/2_1b.png")
    plt.show()
    print("Variance explained by the first 500 images:", cum_var[-1])

def p2_2():
    columns, rows = 5, 2
    
    # Plot mean value
    plt.figure()
    plt.title("Mean image of the dataset")
    avrg_val = np.sum(mnist_pics, axis=0)/len(mnist_pics)
    plt.imshow(avrg_val.reshape(28,28), cmap='Greys_r')
    plt.savefig("plots/2_2a.png")
    plt.show()

    # plot PCAs
    fig = plt.figure(figsize=(11, 5))
    plt.tight_layout()
    ax = []
    fig.suptitle("First 10 Principle Componenets")
    # Plot subplot of prinicpal componenets
    for i in range(columns*rows):
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("Principal Component:"+str(i), fontsize="10")  # set title
        plt.imshow(pc_list[i].reshape(28,28), cmap='Greys_r')
    
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig("plots/2_2b.png")
    plt.show()

def p2_3():
    pass

if __name__ == "__main__":
    # Load MNIST.
    mnist_pics = np.load("data/images.npy")

    # Reshape mnist_pics to be a 2D numpy array.
    num_images, height, width = mnist_pics.shape
    mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

    num_leading_components = 500

    cum_var = get_cumul_var(
        mnist_pics=mnist_pics,
        num_leading_components=num_leading_components)

    p2_1()
    p2_2()