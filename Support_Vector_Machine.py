import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm

# filename = "ex6data1.mat"
# filename = "ex6data2.mat"
filename = "ex6data3.mat"
ex6data1 = scipy.io.loadmat(filename)
X = ex6data1["X"]
y = ex6data1["y"].flatten()
X_val = ex6data1["Xval"]
y_val = ex6data1["yval"].flatten()



def plot_data(X, y):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    plt.scatter(X[pos_idx, 0], X[pos_idx, 1], marker="+")
    plt.scatter(X[neg_idx, 0], X[neg_idx, 1], marker="o")


def plot_decision_boundary_linear(X, y, model):
    plot_data(X, y)
    intercept = model.intercept_
    coefficient = model.coef_
    x_ = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
    # y_ here is x2   decision boundary is || intercept + x1 * theta1 + x2 * theta2 = 0
    y_ = - (coefficient[0][0] * x_ + intercept) / coefficient[0][1]
    plt.plot(x_, y_)


# model = svm.SVC(kernel="linear", C=100)
# model.fit(X, y)
# plot_decision_boundary_linear(X, y, model)
# plt.show()


def gaussian_kernel(x1, x2, sigma=0.1):
    # x1 = x1.flatten()
    # x2 = x2.flatten()
    sim = np.exp(-np.sum(np.power(x1 - x2, 2)) / 2 / np.power(sigma, 2))
    return sim


def gaussian(sigma):
    return lambda x1, x2: np.exp(-np.sum(np.power(x1 - x2, 2)) / 2 / np.power(sigma, 2))


# pre-compute the kernel matrix from data matrices
# Your kernel must take as arguments two matrices of shape
# (n_samples_1, n_features),
# (n_samples_2, n_features)
# and return a kernel matrix of shape (n_samples_1, n_samples_2).
def gaussian_kernel_closure(sigma):
    def compute_kernel_matrix(X1,X2):
        # n_sample, n_features = X1.shape
        k_matrix = np.zeros([X1.shape[0], X2.shape[0]])
        gaussian_function = gaussian(sigma)
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                k_matrix[i][j] = gaussian_function(x1,x2)
        return k_matrix
    return compute_kernel_matrix

def plot_contour_decision_boundary(X, y, model):
    plot_data(X, y)
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1, x2)
    # vals (100,100,1) X1(100,100) X2(100,100)
    # this_X (2,100,100) turn it into (10000,2)
    this_X = np.asarray([X1, X2])
    this_X = this_X.reshape(2, 10000).T
    # vals (10000,)
    vals = model.predict(this_X)
    vals = vals.reshape(100, 100)
    plt.contour(X1, X2, vals, [0.5])


kernel = gaussian_kernel_closure(0.1)
model = svm.SVC(kernel=kernel,C=1)
model.fit(X, y)
plot_contour_decision_boundary(X, y, model)
plt.show()

# x1 = np.asanyarray([1 ,2 ,1])
# x2 = np.asanyarray([0, 4 ,-1])
# sigma = 2
# print(gaussian_kernel(x1,x2,sigma))


def find_best_combination_C_sigma(X,y,X_val,y_val):
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    Sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    target_C = 0
    target_sigma = 0
    highest_correctness = 0

    for c in C:
        for sigma in Sigma:
            kernel = gaussian_kernel_closure(sigma)
            model = svm.SVC(C=c,kernel=kernel)
            model.fit(X,y)
            predict = model.predict(X_val)
            correctness = sum(predict == y_val) / len(y_val) * 100
            print(correctness)
            if correctness > highest_correctness:
                highest_correctness = correctness
                target_sigma = sigma
                target_C = c

    return target_sigma,target_C,highest_correctness


# print(find_best_combination_C_sigma(X,y,X_val,y_val))
