# ------------------------ Part 1 --------------------------------
from array import array
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# 1.1 -> General code 

def create_graph(nodesfile, edgesfile):
    G = nx.DiGraph()
    n = open(nodesfile, "r")
    e = open(edgesfile, "r")
    #First, inputing the nodes
    next(n)
    for line in n:
        node_info = line.split()
        if node_info[1] == 'None':
            G.add_node(node_info[0])
        else:
            G.add_node(node_info[0], type=node_info[1])
    n.close()
    next(e)
    for line in e:
        edge_info = line.split()
        if edge_info[5] == 'physical':
            if not G.has_edge(edge_info[0], edge_info[1]):
                G.add_edge(edge_info[0], edge_info[1]) 
                G.add_edge(edge_info[1], edge_info[0])    
        else:
            if G.has_edge(edge_info[0], edge_info[1]) and G.has_edge(edge_info[1], edge_info[0]):
                G.remove_edge(edge_info[1], edge_info[0])
            else:
                G.add_edge(edge_info[0], edge_info[1]) 
            
    e.close()
    return G

EGFR_Graph = create_graph("EGFR1-nodes.txt", "EGFR1-edges.txt")
TGF_Graph = create_graph("TGF_beta_Receptor-nodes.txt", "TGF_beta_Receptor-edges.txt")
TNF_Graph = create_graph("TNFalpha-nodes.txt", "TNFalpha-edges.txt")
Wnt_Graph = create_graph("Wnt-nodes.txt", "Wnt-edges.txt")

## ------ How to draw the graph -------
#nx.draw(EGFR_Graph, with_labels=True, font_weight='bold')
#plt.show()
## ------------------------------------

# Graph for the human protein interaction network
HPI_Graph = nx.DiGraph()
e = open("pathlinker-human-network.txt", "r")
next(e)
for line in e:
    edge_info = line.split()
    HPI_Graph.add_edge(edge_info[0], edge_info[1], weight=edge_info[2]) 
e.close()

#print(list(HPI_Graph.nodes)[12059])
#HPI_Graph.remove_node('Q6UXB4')
#HPI_Graph.remove_node('P17213')



# ------------------------ Part 2 --------------------------------

def create_rank_dict(ranked_edges):
    rank_dict = {}
    r = open(ranked_edges, "r")
    next(r)
    for line in r:
        edge_info = line.split()
        if edge_info[2] in rank_dict.keys():
            rank_dict[edge_info[2]].append((edge_info[0], edge_info[1]))
        else:
            rank_dict[edge_info[2]] = [(edge_info[0], edge_info[1])]
    r.close()
    return rank_dict

# EGFR_ranked_edges = create_rank_dict("PathLinker-results\EGFR1-k_20000-ranked-edges.txt")
# TGF_ranked_edges = create_rank_dict("PathLinker-results\TGF_beta_Receptor-k_20000-ranked-edges.txt")
# TNF_ranked_edges = create_rank_dict("PathLinker-results\TNFalpha-k_20000-ranked-edges.txt")
# Wnt_ranked_edges = create_rank_dict("PathLinker-results\Wnt-k_20000-ranked-edges.txt")




# ------------------------ Part 3 --------------------------------

def precision_recall(ranked_list, graph):
    prec = []
    rec = []
    true_positives = 0
    computed_edges = 0
    rk = 0
    P = len(list(graph.edges))
    for rank in ranked_list:
        rk += 1
        for tuple in ranked_list[rank]:
            computed_edges += 1
            if tuple in list(graph.edges):
                true_positives += 1
        
        rec.append(true_positives / P)
        prec.append(true_positives / computed_edges)
    
    return rec, prec

# recEGFR, precEGFR = precision_recall(EGFR_ranked_edges, EGFR_Graph)
# recTGF, precTGF = precision_recall(TGF_ranked_edges, TGF_Graph)
# recTNFpl, precTNFpl = precision_recall(TNF_ranked_edges, TNF_Graph)
# recWnt, precWnt = precision_recall(Wnt_ranked_edges, Wnt_Graph)

# plt.plot(recEGFR,precEGFR, label="EGFR")
# plt.plot(recTGF,precTGF, label="TGF")
# plt.plot(recTNFpl,precTNFpl, label="TNF")
# plt.plot(recWnt,precWnt, label="Wnt")
# plt.legend()
# plt.show()



# ------------------------ Part 4 --------------------------------

#nx.dijkstra_path(G, 'a', 'd')
#path_lenght = nx.dijkstra_path_length(path)
#file = open(textfile, "w")

def shortest_paths(Graph):
    tfs = [x for x,y in Graph.nodes(data=True) if y['type']=='tf']
    receptors = [x for x,y in Graph.nodes(data=True) if y['type']=='receptor']

    sp_dict = {}
    for t in tfs:
        spaths = nx.single_source_dijkstra_path(Graph, t)
        for r in receptors:
            if r in spaths:
                path = spaths[r]
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    if edge not in sp_dict:
                        sp_dict.update({edge : 1})
                    else:
                        sp_dict.update({edge : sp_dict[edge] + 1})
    return sp_dict


def translate_file(spdict, filename):
    ordered_edges = dict(reversed(sorted(spdict.items(), key=lambda item: item[1])))
    file = open(filename, "w")
    for edge in ordered_edges:
        L = [edge[0], " " ,edge[1], " ",str(ordered_edges[edge]), "\n"]
        file.writelines(L)
    file.close()


# spEGFR = shortest_paths(EGFR_Graph)
# spTGF = shortest_paths(TGF_Graph)
# spTNF = shortest_paths(TNF_Graph)
# spWnt = shortest_paths(Wnt_Graph)
# translate_file(spEGFR, "EGFRshortest_paths.txt")
# translate_file(spTGF, "TGFshortest_paths.txt")
# translate_file(spTNF, "TNFshortest_paths.txt")
# translate_file(spWnt, "Wntshortest_paths.txt")
    
# EGFRsp = create_rank_dict("EGFRshortest_paths.txt")
# TGFsp = create_rank_dict("TGFshortest_paths.txt")
# TNFsp = create_rank_dict("TNFshortest_paths.txt")
# Wntsp = create_rank_dict("Wntshortest_paths.txt")

# recEGFR, precEGFR = precision_recall(EGFRsp, EGFR_Graph)
# recTGF, precTGF = precision_recall(TGFsp, TGF_Graph)
# recTNFsp, precTNFsp = precision_recall(TNFsp, TNF_Graph)
# recWnt, precWnt = precision_recall(Wntsp, Wnt_Graph)

# plt.plot(recEGFR,precEGFR, label="EGFR")
# plt.plot(recTGF,precTGF, label="TGF")
# plt.plot(recTNFsp,precTNFsp, label="TNF")
# plt.plot(recWnt,precWnt, label="Wnt")
# plt.legend()
# plt.show()




# ------------------------ Part 5 --------------------------------

def RWR(Graph):
    #TODO: fix this
    largest_scc = max(nx.strongly_connected_components(HPI_Graph), key=len)
    print(largest_scc)
    sg = HPI_Graph.subgraph(largest_scc)
    print(sg)
    A = nx.adjacency_matrix(sg)
    D = np.zeros([np.shape(A)[0], np.shape(A)[1]])
    sum = A.sum(axis=1)
    for i in range(np.shape(A)[0]):
        D[i,i] = sum[i] - A[i,i]
        if D[i,i] == 0:
            print(i)

    DA = np.transpose(np.invert(D) * A)

    receptors = [x for x,y in Graph.nodes(data=True) if y['type']=='receptor']
    S = len(receptors)

    s = np.zeros(np.shape(A)[0])
    all_nodes = list(sg.nodes)
    for r in receptors:
        ind = all_nodes.index(r)
        s[ind] = 1/S
    
    q = 0.5
    p = np.invert(np.identity(np.shape(A)[0]) - (1-0.5)*DA) * (q*s)

    edges = list(Graph.edges)
    edges_flux = {}
    for e in edges:
        w = e[0][1]["weight"]
        indu = all_nodes.index(e[0])
        u = e[0]
        f = p[indu]*w/Graph.out_degree(u)
        edges_flux.update({e : f})

    return edges_flux
    

RWR(EGFR_Graph)

# rwrEGFR = RWR(EGFR_Graph)
# rwrTGF = RWR(TGF_Graph)
# rwrTNF = RWR(TNF_Graph)
# rwrWnt = RWR(Wnt_Graph)
# translate_file(rwrEGFR, "EGFR_RWR.txt")
# translate_file(rwrTGF, "TGF_RWR.txt")
# translate_file(rwrTNF, "TNF_RWR.txt")
# translate_file(rwrWnt, "Wnt_RWR.txt")

# EGFRrwr = create_rank_dict("EGFR_RWR.txt")
# TGFrwr = create_rank_dict("TGF_RWR.txt")
# TNFrwr = create_rank_dict("TNF_RWR.txt")
# Wntrwr = create_rank_dict("Wnt_RWR.txt")

# recEGFRrwr, precEGFRrwr = precision_recall(EGFRrwr, EGFR_Graph)
# recTGFrwr, precTGFrwr = precision_recall(TGFrwr, TGF_Graph)
# recTNFrwr, precTNFrwr = precision_recall(TNFrwr, TNF_Graph)
# recWntrwr, precWntrwr = precision_recall(Wntrwr, Wnt_Graph)