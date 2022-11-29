using EcologicalNetworks
using LinearAlgebra
using NPZ
using PyCall

# ------------- LF Step --------------

# Alpha values:
# a1 -> relative importance of the original value of interaction
# a2 -> in degree
# a3 -> out degree
# a4 -> connectance degree
# Sum of alpha values equals 1


# #Imagine we get adjacency matrix Y
# print("hello")
np = pyimport("numpy")
# pyfile = npzread("tryout.npy.npz")
data = npzread("tryout3.npz")
# #data = np.load(pyfile, allow_pickle = true)
# print(data)


function createA(Y::Matrix, alpha::Vector)
    n = size(Y,1)
    m = size(Y,2)
    A = zeros(n,m)
    sums_j = sum(Y, dims=1)
    print(sums_j)
    sums_i = sum(Y, dims=2)
    print(sums_i)
    sum_all = sum(Y)
    for i=1:n
        for j=1:m
            v = [Y[i,j], 1/n*sums_j[j], 1/m*sums_i[i], 1/(n*m)*sum_all]
            A[i,j] = dot(v, alpha)
            #print(A)
        end
    end
    return A
end

#Start with k=n (rank of the matrix) -> it should be the original matrix
function createK(A::Matrix, k::Int)
    U, S, V = svd(A)
    print(S)
    counter = 1
    for eig in S
        if eig > k
            S[counter] = 0
        end
        counter += 1
    end
    print(S)
    E = Diagonal(S)
    nA = U*E*transpose(V)
    return nA
end
#The value of Aij is the score for the interaction

M = [1 0 0 0 1 0
    0 1 1 0 0 0
    0 0 1 1 1 1]
T = ["A", "B", "C", "D", "E", "F"]
B = ["1", "2", "3"]

# net = AbstractBipartiteNetwork(M, T, B)
# print(net)
alpha = zeros(4)
alpha[2] = 1
A = createA(M, alpha)
nA = createK(A, 1)
