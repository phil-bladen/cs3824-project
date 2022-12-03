import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skl_metrics

# validation steps:
    # 0. for a given graph, split the graph into set of training data, positive testing edges, and negative testing edges
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

    positive_edges = testing_edges
    negative_edges = list()

    # create one negative edges for every positive edge in the training set
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
    
    return (training_edges, positive_edges, negative_edges)

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
