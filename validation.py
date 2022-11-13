import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

# validation steps:
    # 0. for a given graph, split the graph into set of training data and testing data
    # 1. run link prediction algorithm on the testing data
    # 2. rank the results by the score produced
    # 3. generate precision-recall curve and ROC curve
    # 4. find the area under both curves
    # 5. create a box and whisker plot of the results you get from running the above steps 10 times

def main():
    ten_auroc_values = list()
    ten_auprc_values = list()
    
    for i in range(10):
        G = nx.karate_club_graph()
        edge_sets = create_sets(G, 0.1)

        # generate a sample output from LF-SVD run on the testing set of edges from the graph
        # note, the below method is implemented in a placeholder fashion
        lfsvd_sample_output = get_lfsvd_output(edge_sets[0], edge_sets[1])
        
        # calculate auroc and auprc value given lfsvd_output
            # instead of doing the real calculation, as a placeholder I will:
                # find the average value of the matrix and use that as a placeholder for auroc
                # find the max value of the matrix and use that as a placeholder for auroc 
        auroc_and_auprc = calculate_auroc_and_auprc(lfsvd_sample_output)
        ten_auroc_values.append(auroc_and_auprc[0])
        ten_auprc_values.append(auroc_and_auprc[1])

    # plot the 10 AUROC and AUPRC values
    plot_values(ten_auroc_values, ten_auprc_values)

def calculate_auroc_and_auprc(lfsvd_output_matrix):
    placeholder_auroc = 0
    placeholder_auprc = 0    
    output_values = lfsvd_output_matrix.tolist()

    # create a flattened list of all matrix values
    # also, create a dictionary of the coordinates of every element in the original output matrix. these coordinates will be checked when calculating AUROC and AUPRC
    # also, create a descending order version of the flattened list. this is used for AUROC and AUPRC calculation
    coordinates_dict = dict()
    flattened_list = list()
    descending_flattened_list = list()
    for row_list in output_values:
        for elem in row_list:
            flattened_list.append(elem)
            coordinates_dict[elem] = (row_list, row_list.index(elem))
    descending_flattened_list = flattened_list.sort(reverse=True)

    # placeholder_auroc_calculation (sum)
    for row_list in output_values:
        for elem in row_list:
            placeholder_auroc += elem
    # print("placeholder_auroc: %d" % placeholder_auroc)

    # placeholder_auprc_calculation (average)
    for row_list in output_values:
        for elem in row_list:
            placeholder_auprc += elem
    placeholder_auprc = placeholder_auprc / len(output_values)
    # print("placeholder_auprc: %d" % placeholder_auprc)

    return (placeholder_auroc, placeholder_auprc)

# takes a graph as input. and outputs two lists of subsets of the edges in the graph
#   1. a set of training edges
#   2. a set of testing edges
# this function removes the testing edges from the input NetworkX graph
# the "fraction" parameter designates what fraction of edges from the original graph
# will be randomly selected to be placed into the training set
def create_sets(G: nx.Graph, fraction: float):
    num_edges_to_remove = int(fraction * G.number_of_edges())
    training_edges = list(G.edges) # at the start, the entire graph is the "training" set
    testing_edges = list()
    
    for i in range(num_edges_to_remove):
        random_edge_chosen = random.choice(training_edges)
        G.remove_edge(random_edge_chosen[0], random_edge_chosen[1]) # edges are tuples of vertices u and v
        training_edges.remove(random_edge_chosen)
        testing_edges.append(random_edge_chosen)

    # ensure no overlap between sets
    for testing_edge in testing_edges:
        assert(testing_edge not in training_edges)

    # create one negative edges for every positive edge in the training set
    # negative edges are created by:
    #  1. selecting 2 random nodes
    #  2. ensuring there is no existing edge between them (select new nodes if there is an edge)
    #  3. if not, add the edge to the testing set
    for i in range(len(testing_edges)):
        while(True):
            # G.nodes
            random_node_1 = random.choice(list(G))
            random_node_2 = random.choice(list(G))
            if (random_node_1 == random_node_2): continue # same node picked twice
            if (G.has_edge(random_node_1, random_node_2) or G.has_edge(random_node_2, random_node_1)): continue # edge already exists
            else: testing_edges.append((random_node_1, random_node_2))
            break
    
    return (training_edges, testing_edges)

# this is a placeholder method for obtaining the output from running lfsvd. I've just generated a random score from 0 to 2 for every edge between nodes i and j in the training set
def get_lfsvd_output(training_set: list, testing_set: list):
        training_set_size = len(testing_set)
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
