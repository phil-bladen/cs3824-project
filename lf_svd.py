import scipy.sparse as sc
import numpy as np

def create_prob_matrix(input_matrix_file: str, alpha: list, output_matrix_file: str):
    sp_matrix = sc.load_npz(input_matrix_file).todense()
    #Load here the matrix with whatever name we provide
    A = createA(sp_matrix, alpha)
    nA = createK(A, 1)
    print("\n", nA)
    np.save(output_matrix_file, nA)

    # testing for successful save
    loaded = np.load("lf-svd-output-from-tryout3.npy")
    print(loaded)
    return 0


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
    return A


def createK(A, k):
    # U, S, V = svd(A)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    counter = 0
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