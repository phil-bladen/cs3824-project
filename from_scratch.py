import networkx as nx
import numpy as np
import scipy.sparse as sc
from networkx.algorithms import bipartite
import random
import lf_svd
from sklearn import metrics as skl_metrics
import matplotlib.pyplot as plt
import os
import time
import pickle

class interaction:
    def __init__(self):

        self.host_tax_id = ""
        self.virus_tax_id = ""
        # We could add the rest of the columns from the .csv to this class
        # if we found it useful. This would be done as below
        self.host = ""
        self.virus = ""
        # self.host_ncbi_resolved = ""
        # self.virus_ncbi_resolved = ""
        self.hostclass = ""
        self.virusclass = ""

def parse_csv(input_file_name: str, host_to_set_of_viruses: dict, virus_to_set_of_hosts: dict, viruses_to_hosts_DAG: nx.DiGraph):
    file = open(input_file_name, 'r')
    on_first_line = True # used to skip the first line
    lines_encountered = 0 # helpful for debugging

    while True:
        line = file.readline()
        if (line != ""): # if line isn't empty
            lines_encountered += 1
            if (on_first_line):
                on_first_line = False
                continue # skip the first line
            else:                          
                line_array = line.split("\t")
                # should we process this line?
                if (len(line_array) < 4 or line_array[2] == "" or line_array[3] == ""): # if line doesn't have IDs, or host_tax_id is missing, or virus_tax_id is missing
                    continue # ignore the line
                else:
            
                    curr_interaction = interaction()
                    curr_interaction.host_tax_id = line_array[2]
                    curr_interaction.virus_tax_id = line_array[3]

                    curr_interaction.host = line_array[0]
                    curr_interaction.virus = line_array[1]
                    curr_interaction.hostclass = line_array[10]
                    curr_interaction.virusclass = line_array[15]
                    # add entry to host_to_set_of_viruses map
                    if (curr_interaction.host_tax_id not in host_to_set_of_viruses):
                        host_to_set_of_viruses[curr_interaction.host_tax_id] = set()
                    host_to_set_of_viruses[curr_interaction.host_tax_id].add(curr_interaction.virus_tax_id)

                    # add entry to host_to_set_of_viruses map
                    if (curr_interaction.virus_tax_id not in virus_to_set_of_hosts):
                        virus_to_set_of_hosts[curr_interaction.virus_tax_id] = set()
                    virus_to_set_of_hosts[curr_interaction.virus_tax_id].add(curr_interaction.host_tax_id)

                    # add virus and host nodes and an edge from virus to host in viruses_to_hosts_DAG
                    # note: NetworkX won't add a duplicate node or duplicate edge if it is already in the graph
                    viruses_to_hosts_DAG.add_node(curr_interaction.virus_tax_id, type='virus', name=curr_interaction.virus, bio_class=curr_interaction.hostclass)   
                    viruses_to_hosts_DAG.add_node(curr_interaction.host_tax_id, type='host', name=curr_interaction.host, bio_class=curr_interaction.virusclass)
                    viruses_to_hosts_DAG.add_edge(curr_interaction.virus_tax_id, curr_interaction.host_tax_id)

        else: # line is empty, we have finished reading the file
            break
    # print("debug here to inspect data structures")
    return 0 # success!

def create_sets(G: nx.Graph, fraction: float):
    path_name= "edge_sets_" + str(time.time())
    os.mkdir(path=path_name, mode=0o777)
    num_edges_to_remove = int(fraction * G.number_of_edges())
    training_edges = list(G.edges) # at the start, the entire graph is the "training" set
    # write training set to disk
    t_edges_file = open(path_name + "/training_edges", "ab")
    pickle.dump(training_edges, t_edges_file)
    t_edges_file.close()

    testing_edges = list()
    
    for i in range(num_edges_to_remove):
        random_edge_chosen = random.choice(training_edges)
        G.remove_edge(random_edge_chosen[0], random_edge_chosen[1]) # edges are tuples of vertices u and v
        training_edges.remove(random_edge_chosen)
        testing_edges.append(random_edge_chosen)

    # ensure no overlap between sets
    for testing_edge in testing_edges:
        assert(testing_edge not in training_edges)

    positive_edges = testing_edges
    # write positive edges to disk
    p_edges_file = open(path_name + "/positive_edges", "ab")
    pickle.dump(positive_edges, p_edges_file)
    p_edges_file.close()

    negative_edges = list()
    # create one negative edge for every positive edge in the training set
    # negative edges are created by:
    #  1. selecting 2 random nodes
    #  2. ensuring there is no existing edge between them (select new nodes if there is an edge)
    #  3. if not, add the edge to the testing set
    for i in range(len(positive_edges)):
        while(True):
            # G.nodes
            random_node_1 = random.choice(list(G))
            random_node_2 = random.choice(list(G))
            if (random_node_1 == random_node_2): continue # same node picked twice
            if (G.has_edge(random_node_1, random_node_2) or G.has_edge(random_node_2, random_node_1)): continue # edge already exists
            else: negative_edges.append((random_node_1, random_node_2))
            break
    # write negative edges to disk
    n_edges_file = open(path_name + "/negative_edges", "ab")
    pickle.dump(negative_edges, n_edges_file)
    n_edges_file.close()
    
    
    return (training_edges, positive_edges, negative_edges)

def RandomWalk(viruses_to_hosts_DAG: nx.DiGraph):
    virusHostMat = nx.to_numpy_array(viruses_to_hosts_DAG)
    print(virusHostMat.shape)
    rowSize = len(virusHostMat)
    columnSize = rowSize 
    #columnSize = len(virusHostMat[0])
    probMatrix = np.zeros(virusHostMat.shape)
    probArray = np.sum(virusHostMat, axis=1) #calculates degree of node i
    for row in range(rowSize):
        for col in range(columnSize):
            if virusHostMat[row][col] != 0:
                probMatrix[row][col] = 1 / probArray[row]
    c = 0.9
    cPrime = 0.1
    cProbTransMatrix = np.array(c * probMatrix.transpose())
    idenMatrix = np.identity(rowSize)
    #print (len(cProbTransMatrix))
    #print(len(cProbTransMatrix[0]))
    diffMatrix = np.array(idenMatrix - cProbTransMatrix)
    invMatrix = np.linalg.inv(diffMatrix)
    qMatrix = cPrime * invMatrix
    #rwrIndex = qMatrix + qMatrix.transpose()
    rwrArray = np.zeros(virusHostMat.shape)
    for row in range(rowSize):
        for col in range(columnSize):
            if row == col:
                rwrArray [row] [col] = -1
            else:
                rwrArray [row] [col] = qMatrix [row] [col] + qMatrix [col] [row] 
    rwrDict = dict(enumerate(rwrArray.flatten(), 1))
    sortedRWR = sorted(rwrDict.items(), key = lambda val: val[1])
    sortedDict = dict(sortedRWR)

def main():
    host_to_set_of_viruses = {}
    virus_to_set_of_hosts = {}
    viruses_to_hosts_DAG = nx.DiGraph()
    parse_csv("Virion.csv", host_to_set_of_viruses, virus_to_set_of_hosts, viruses_to_hosts_DAG)
    # print("starting RandomWalk")
    # RandomWalk(viruses_to_hosts_DAG) # new
    # print("finished RandomWalk")
    print("testing new PRC and ROC calculations")
    #new_calculate("output-when-using-tryout3.npy", "edge_sets_1670117625.3284886/positive_edges", "edge_sets_1670117625.3284886/negative_edges")
    


    sets = create_sets(viruses_to_hosts_DAG, 0.1)

    # write the graph here, then try the lookup thing
    nx.write_adjlist(viruses_to_hosts_DAG, "nxGraphFile")
    # print(viruses_to_hosts_DAG[5])
    test_read = nx.read_adjlist("nxGraphFile")
    # print(test_read[5])
    
    list_hosts = [x for x,y in viruses_to_hosts_DAG.nodes(data=True) if y['type']=='host'] # might be 9600 long
    list_viruses = [x for x,y in viruses_to_hosts_DAG.nodes(data=True) if y['type']=='virus'] # might be 4100 long
    
    h_list_ID_to_index = dict()
    v_list_ID_to_index = dict()

    h_counter = 0
    for host in list_hosts:
        h_list_ID_to_index[host] = h_counter
        h_counter += 1
    
    v_counter = 0
    for virus in list_viruses:
        v_list_ID_to_index[virus] = v_counter
        v_counter += 1

    
    # A = bipartite.biadjacency_matrix(viruses_to_hosts_DAG, list_viruses)
    A = bipartite.biadjacency_matrix(viruses_to_hosts_DAG, list_viruses, list_hosts)
    A_arr = A.toarray()

    # for virus in list_viruses:
    #     for host in list_hosts:
    #         virus_biadj_index = v_list_ID_to_index[virus]
    #         host_biadj_index = h_list_ID_to_index[host]

    #         # virus_list_index = v_list_ID_to_index[virus]
    #         # host_list_index = h_list_ID_to_index[host]

    #         # virus_biadj_index = int(list_hosts[virus_list_index])
    #         # host_biadj_index = int(list_hosts[host_list_index])


    #         assert(A_arr[virus_biadj_index][host_biadj_index] == 1)
    edges_list = viruses_to_hosts_DAG.edges
    for edge in edges_list:
        virus_biadj_index = v_list_ID_to_index[edge[0]]
        host_biadj_index = h_list_ID_to_index[edge[1]]
        assert(A_arr[virus_biadj_index][host_biadj_index] == 1)
    print("tests passed! you now have a map of nodeIDs to biadj (and probably also lfsvd) elements")
        



    # A_adj = nx.adjacency_matrix(viruses_to_hosts_DAG)
    # print("size of A_adj")
    # print(A_adj.get_shape())
    m1 = sc.csr_matrix.toarray(A)
    # sc.save_npz("tryout3", A, compressed=False)
    # sc.save_npz("13724_attempt.npz", A_adj, compressed=False)
    # N1 = sc.load_npz("tryout3.npz")
    # print(N1)
    #SciPy and Numpy error handling

    #Load here the matrix with whatever name we provide
    # lf_svd.create_prob_matrix("tryout3.npz", [0, 1, 0, 0], "output-when-using-tryout3.npy")
    lf_svd.create_prob_matrix("13724_attempt.npz", [0, 1, 0, 0], "results_13724_attempt.npy")
    edge_prob_matrix = np.load("results_13724_attempt.npy", allow_pickle=True)
    print(edge_prob_matrix.shape)
    print("debug")
    # sp_matrix = sc.load_npz("tryout3.npz").todense()
    # print("printing sp_matrix")
    # print(sp_matrix)
    # alpha = [0, 1, 0, 0]
    # #M = [[1, 0, 0, 0, 1, 0],[0, 1, 1, 0, 0, 0],[0, 0, 1, 1, 1, 1]]
    # print("starting createA()")
    # A = createA(sp_matrix,alpha)
    # print("finished createA()")
    # print("starting createK()")
    # nA = createK(A, 1)
    # print("\n", nA)
    # print("finished createK(), saving now")
    # # sc.save_npz("lf-svd-output-from-tryout3.npz", nA, compressed=False)
    # np.save("lf-svd-output-from-tryout3.npy", nA)
    # loaded = np.load("lf-svd-output-from-tryout3.npy")
    # print(loaded)

# def createA(Y, alpha):
#     n = Y.shape[0]
#     m = Y.shape[1]
#     A = np.zeros((n,m))
#     sums_i = np.sum(Y, axis=1)
#     sums_j = np.sum(Y, axis=0)
#     sum_all = np.sum(Y)

#     for i in range(n):
#         for j in range(m):
#             v = [Y[i,j], 1/n*sums_j[0,j], 1/m*sums_i[i,0], 1/(n*m)*sum_all]
#             A[i][j] = np.dot(v, alpha)
#             #print(A)
#     return A


# def createK(A, k):
#     # U, S, V = svd(A)
#     U, S, V = np.linalg.svd(A, full_matrices=False)
#     counter = 0
#     for eig in S:
#         if eig > k:
#             S[counter] = 0
#         counter += 1   
#     E = np.diag(S)
#     nA = U @ E @ V
#     return nA

def validate():
    ten_auroc_values = list()
    ten_auprc_values = list()
    
    for i in range(10):
        G = nx.karate_club_graph()
        edge_sets = create_sets(G, 0.1)

        # generate a sample output from LF-SVD run on the testing set of edges from the graph
        # note, the below method is implemented in a placeholder fashion
        lfsvd_sample_output = get_lfsvd_output(edge_sets[0], edge_sets[1], edge_sets[2])
        
        # calculate auroc and auprc value given lfsvd_output
            # instead of doing the real calculation, as a placeholder I will:
                # find the average value of the matrix and use that as a placeholder for auroc
                # find the max value of the matrix and use that as a placeholder for auroc 
        auroc_and_auprc = calculate_auroc_and_auprc(lfsvd_sample_output, edge_sets[1], edge_sets[2])
        ten_auroc_values.append(auroc_and_auprc[0])
        ten_auprc_values.append(auroc_and_auprc[1])

    # plot the 10 AUROC and AUPRC values
    plot_values(ten_auroc_values, ten_auprc_values)

def new_calculate(lfsvd_output_file: str, positive_testing_edges_file: str, negative_testing_edges_file: str):
    # load lfsvd output
    edge_prob_matrix = np.load(lfsvd_output_file, allow_pickle=True)
    print(edge_prob_matrix[1][4])
    
    # load positive edges output
    p_filepointer = open(positive_testing_edges_file, "rb")
    positive_edges = pickle.load(p_filepointer)
    p_filepointer.close()

    # load negative edges output
    n_filepointer = open(negative_testing_edges_file, "rb")
    negative_edges = pickle.load(n_filepointer)
    n_filepointer.close()

    # real auprc_calculation
    testing_set = positive_edges + negative_edges
    testing_edges_plus_scores = list()

    for edge in testing_set:
        u = int(edge[0])
        v = int(edge[1])
        edge_lfsvd_score = edge_prob_matrix[u][v]
        #edge_lfsvd_score = sc.csr_matrix.__getitem__ edge_prob_matrix[u][v]
        # create a tuple with the edge and its score. add this tuple to the list
        testing_edges_plus_scores.append((edge, edge_lfsvd_score))
        #print(edge_lfsvd_score)

    # sort edges by their lfsvd score
    testing_edges_plus_scores.sort(key=lambda edge_score_tuple: edge_score_tuple[1], reverse=True)

    sorted_testing_edges = list()

    # create a list of the edges (sorted by lfsvd score)
    for tup in testing_edges_plus_scores:
        sorted_testing_edges.append(tup[0])
    
    prc_counter = 0
    true_positives = 0
    false_positives = 0
    # false_negatives = len(sorted_testing_edges) # false negative defined as an edge not yet reached, but is in the positive testing set
    # true_negatives = 0
    precision_at_each_edge = list()
    recall_at_each_edge = list()
    predictions_vector = list()
    testing_score_list = list()
    for edge in sorted_testing_edges:
        if (edge in positive_edges):
            print("true positive")
            true_positives += 1
            predictions_vector.append(1)
            
        elif (edge in negative_edges):
            print("false positive")
            false_positives += 1
            predictions_vector.append(0)
        # false_negatives -= 1
        # precision_at_current_edge = true_positives / (true_positives + false_negatives * 1.0) # 1.0 used for float conversion
        # precision_at_each_edge.append(precision_at_current_edge)
        # recall_at_current_edge = true_positives / (true_positives + false_positives * 1.0) # 1.0 used for float conversion
        # recall_at_each_edge.append(recall_at_current_edge)
        testing_score_list.append(edge_prob_matrix[edge[0]][edge[1]])
        prc_counter += 1

    #prc_display = skl_metrics.PrecisionRecallDisplay.from_predictions(predictions_vector, testing_score_list, name="PRC")
    avg_ps = skl_metrics.average_precision_score(predictions_vector, testing_score_list)
    print("avg_ps: %f" % avg_ps)
    prc_display = skl_metrics.PrecisionRecallDisplay.from_predictions(predictions_vector, testing_score_list)
    # _ = prc_display.ax_.set_title("2-class Precision-Recall curve")
    prc_display.plot()

    auroc = skl_metrics.roc_auc_score(predictions_vector, testing_score_list)
    print("auroc: %f" % auroc)
    auroc_display = skl_metrics.RocCurveDisplay.from_predictions(predictions_vector, testing_score_list)
    auroc_display.plot()

    
    # roc_display = RocCurveDisplay.from_predictions()
    

    #plt.plot(recall_at_each_edge, precision_at_each_edge)
    # plt.plot(precision_at_each_edge, recall_at_each_edge)
    #plt.show()

    # testing_set[0] = (2, 27)
    # lfsvd_output_matrix[2][27] == 1.09 (score for that testing set edge).         

    return (avg_ps, auroc)

    print("e")


def calculate_auroc_and_auprc(lfsvd_output_matrix, positive_testing_edges: list, negative_testing_edges: list):

    # real auprc_calculation
    testing_set = positive_testing_edges + negative_testing_edges
    testing_edges_plus_scores = list()
    #testing_set.sort(reverse=True)

    for edge in testing_set:
        u = edge[0]
        v = edge[1]
        edge_lfsvd_score = lfsvd_output_matrix[u][v]
        # create a tuple with the edge and its score. add this tuple to the list
        testing_edges_plus_scores.append((edge, edge_lfsvd_score))
        #print(edge_lfsvd_score)

    # sort edges by their lfsvd score
    testing_edges_plus_scores.sort(key=lambda edge_score_tuple: edge_score_tuple[1], reverse=True)

    sorted_testing_edges = list()

    # create a list of the edges (sorted by lfsvd score)
    for tup in testing_edges_plus_scores:
        sorted_testing_edges.append(tup[0])
    
    prc_counter = 0
    true_positives = 0
    false_positives = 0
    # false_negatives = len(sorted_testing_edges) # false negative defined as an edge not yet reached, but is in the positive testing set
    # true_negatives = 0
    precision_at_each_edge = list()
    recall_at_each_edge = list()
    predictions_vector = list()
    testing_score_list = list()
    for edge in sorted_testing_edges:
        if (edge in positive_testing_edges):
            print("true positive")
            true_positives += 1
            predictions_vector.append(1)
            
        elif (edge in negative_testing_edges):
            print("false positive")
            false_positives += 1
            predictions_vector.append(0)
        # false_negatives -= 1
        # precision_at_current_edge = true_positives / (true_positives + false_negatives * 1.0) # 1.0 used for float conversion
        # precision_at_each_edge.append(precision_at_current_edge)
        # recall_at_current_edge = true_positives / (true_positives + false_positives * 1.0) # 1.0 used for float conversion
        # recall_at_each_edge.append(recall_at_current_edge)
        testing_score_list.append(lfsvd_output_matrix[edge[0]][edge[1]])
        prc_counter += 1

    #prc_display = skl_metrics.PrecisionRecallDisplay.from_predictions(predictions_vector, testing_score_list, name="PRC")
    avg_ps = skl_metrics.average_precision_score(predictions_vector, testing_score_list)
    print("avg_ps: %f" % avg_ps)
    prc_display = skl_metrics.PrecisionRecallDisplay.from_predictions(predictions_vector, testing_score_list)
    # _ = prc_display.ax_.set_title("2-class Precision-Recall curve")
    prc_display.plot()

    auroc = skl_metrics.roc_auc_score(predictions_vector, testing_score_list)
    print("auroc: %f" % auroc)
    auroc_display = skl_metrics.RocCurveDisplay.from_predictions(predictions_vector, testing_score_list)
    auroc_display.plot()

    
    # roc_display = RocCurveDisplay.from_predictions()
    

    #plt.plot(recall_at_each_edge, precision_at_each_edge)
    # plt.plot(precision_at_each_edge, recall_at_each_edge)
    #plt.show()

    # testing_set[0] = (2, 27)
    # lfsvd_output_matrix[2][27] == 1.09 (score for that testing set edge).         

    return (avg_ps, auroc)

# this is a placeholder method for obtaining the output from running lfsvd. I've just generated a random score from 0 to 2 for every edge between nodes i and j in the training set
def get_lfsvd_output(training_set: list, positive_edges_list: list, negative_edges_list: list):
        training_set_size = len(training_set)
        lfsvd_sample_output = np.zeros((training_set_size, training_set_size)) # empty nxn matrix
        for i in range(training_set_size):
            for j in range(training_set_size):
                lfsvd_sample_output[i][j] = round(random.uniform(0.0, 2.0), 2)
        return lfsvd_sample_output

def plot_values(auroc_values: list, auprc_values: list):
    print("graphing results")
    fig, axs = plt.subplots(1, 2)
    axs[0].boxplot(auroc_values)
    axs[0].set_title("AUROC Values")
    axs[1].boxplot(auprc_values)
    axs[1].set_title("AUPRC Values")
    plt.show()

if __name__ == "__main__":
    main()