import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    G = nx.karate_club_graph()
    create_sets(G, 0.1)

    # plot data
    random_auroc = list()
    for i in range(20):
        random_auroc.append(random.randint(0, 100))

    random_auprc = list()
    for i in range(20):
        random_auprc.append(random.randint(0, 100))

    plot_values(random_auroc, random_auprc)
    return 0

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
