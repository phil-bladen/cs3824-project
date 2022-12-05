import scipy.sparse as sc
import numpy as np

def create_prob_matrix(input_matrix_file: str, alpha: list, k: int, output_matrix_file: str):
    sp_matrix = sc.load_npz(input_matrix_file).todense()
    A = createA(sp_matrix, alpha)
    nA = createK(A, k)
    print("\n", nA)
    np.save(output_matrix_file, nA)

def createA(Y, alpha):
    n = Y.shape[0]
    m = Y.shape[1]
    A = np.zeros((n,m))
    sums_i = np.sum(Y, axis=1)
    sums_j = np.sum(Y, axis=0)
    sum_all = np.sum(Y)

    for i in range(n):
        for j in range(m):
            v = [Y[i,j], 1/n*sums_j[0,j], 1/m*sums_i[i,0], 1/(n*m)*sum_all]
            A[i][j] = np.dot(v, alpha)
            #print(A)
        if (i % 100) == 0: print("i: %d" % i)
    return A

def createK(A, k):
    # U, S, V = svd(A)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    counter = 0
    print("len(S): %d" % len(S))
    for eig in S:
        if eig > k:
            S[counter] = 0
        counter += 1   
    E = np.diag(S)
    nA = U @ E @ V
    return nA

# #Load here the matrix with whatever name we provide 
# sp_matrix = sc.load_npz("tryout3.npz").todense()
# alpha = [0, 1, 0, 0]
# #M = [[1, 0, 0, 0, 1, 0],[0, 1, 1, 0, 0, 0],[0, 0, 1, 1, 1, 1]]
# A = createA(sp_matrix,alpha)
# nA = createK(A, 1)
# print("\n", nA)
# np.save("lf-svd-output-from-tryout3.npy", nA)
# loaded = np.load("lf-svd-output-from-tryout3.npy")
# print(loaded)