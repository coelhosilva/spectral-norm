from sys import argv
import time
import numpy as np


def get_a_ij(i, j):
    return 2/(i*(i+1) + (j-1)*(j+2*i - 2))


def multiply_At_A_b(A, b):
    return np.matmul(np.transpose(A), np.matmul(A, b))


def calculate_norm(u, v):
    return np.sqrt(np.sum(np.multiply(u, v))/np.sum(np.square(v)))


def calculate(N):
    X, Y = np.ix_(np.arange(1, N+1), np.arange(1, N+1))
    matrix = get_a_ij(X, Y)
    u = np.ones(N)
    for _ in range(10):
        v = multiply_At_A_b(matrix, u)
        u = multiply_At_A_b(matrix, v)
    return calculate_norm(u, v)


if __name__ == '__main__':
    if len(argv) > 1:
        N = int(argv[1])
    else:
        N = 100
    start_time = time.time()
    result = calculate(N)
    print("{0:.40f}".format(result))
    print("Time elapsed is {} seconds".format(time.time() - start_time))
