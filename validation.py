import networkx as nx
import random

def main():
    G = nx.karate_club_graph()
    create_sets(G, 0.1)

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

    
    return (training_edges, testing_edges)

# what are the next steps? now consider running some operation on the testing edges
    # what would it require to do that properly? 
    # in the project description it says that we can just run the algorithm on the testing edges


if __name__ == "__main__":
    main()