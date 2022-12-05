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
        if i % 100 == 0: print("LFSVD first loop i %d of %d" % (i, n))
        for j in range(m):
            v = [Y[i,j], 1/n*sums_j[0,j], 1/m*sums_i[i,0], 1/(n*m)*sum_all]
            A[i][j] = np.dot(v, alpha)
    return A

def createK(A, k):
    # U, S, V = svd(A)
    print("starting step 2 of LFSVD")
    U, S, V = np.linalg.svd(A, full_matrices=False)
    counter = 0
    for eig in S:
        if eig > k:
            S[counter] = 0
        counter += 1   
    E = np.diag(S)
    nA = U @ E @ V
    print("finished step 2 of LFSVD")
    return nA