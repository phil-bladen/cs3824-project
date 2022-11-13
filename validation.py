import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    G = nx.karate_club_graph()

    ten_auroc_values = list()
    ten_auprc_values = list()
    
    for i in range(10):
        # start of main for loop that should run 10 times
        edge_sets = create_sets(G, 0.1)

        # generate a sample output from LF-SVD run on the testing set of edges from the graph
        training_set_size = len(edge_sets[1])
        lfsvd_sample_output = np.zeros((training_set_size, training_set_size)) # empty nxn matrix
        for i in range(training_set_size):
            for j in range(training_set_size):
                lfsvd_sample_output[i][j] = round(random.uniform(0.0, 2.0), 2)
        
        # calculate auroc and auprc value
            # instead of doing the real calculation, I will:
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

    # here, I will:
    # 1. run some link prediction algorithm on the testing data
    # 2. rank the results by the score produced
    # 3. generate precision-recall curve and ROC curve
    # 4. find the area under both curves
    # 5. create a box and whisker plot of the results you get from running the above steps 10 times
    
    # while I wait for our group to finish the LF-SVD prediction, i need to complete steps 2-5
    # in order to do that, I need to think of some 'placeholder' procedure I can use to simulate step 1 so that
    # I can get steps 2-5 working.

    # TODO: ask group exactly the LF-SVD will output, that way I can model some hard-coded thing that
    # approximates its output temporarily so that I can build steps 2-5 while other group
    # members finish step 1. This way, the group can just paste the LF-SVD results over my hard-coded
    # placeholder, and we will be done with the assignment

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
