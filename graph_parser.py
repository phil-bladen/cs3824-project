from unicodedata import name
import networkx as nx
import numpy as np
import scipy.sparse as sc
from networkx.algorithms import bipartite

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

def main():
    host_to_set_of_viruses = {}
    virus_to_set_of_hosts = {}
    viruses_to_hosts_DAG = nx.DiGraph()
    parse_csv("Virion.csv", host_to_set_of_viruses, virus_to_set_of_hosts, viruses_to_hosts_DAG)
    
    print(len(list(viruses_to_hosts_DAG.nodes())))
    print(len(list(viruses_to_hosts_DAG.neighbors('37124'))))
    #print(nx.get_node_attributes(viruses_to_hosts_DAG, 'name'))

    #A = nx.to_numpy_matrix(viruses_to_hosts_DAG)
    #print(A)

    print(bipartite.is_bipartite(viruses_to_hosts_DAG))

    list_hosts = [x for x,y in viruses_to_hosts_DAG.nodes(data=True) if y['type']=='host']
    list_viruses = [x for x,y in viruses_to_hosts_DAG.nodes(data=True) if y['type']=='virus']
    
    A = bipartite.biadjacency_matrix(viruses_to_hosts_DAG, list_viruses)
    print(A)
    m1 = sc.csr_matrix.toarray(A)
    sc.save_npz("tryout3", A, compressed=False)
    N1 = sc.load_npz("tryout3.npz")
    print(N1)
    #SciPy and Numpy error handling 


if __name__ == "__main__":
    main()