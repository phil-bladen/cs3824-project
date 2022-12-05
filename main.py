import networkx as nx
import matplotlib as plt
import numpy as np

class interaction:
    def __init__(self):

        self.host_tax_id = ""
        self.virus_tax_id = ""
        # We could add the rest of the columns from the .csv to this class
        # if we found it useful. This would be done as below
        # self.host = ""
        # self.virus = ""
        # self.host_ncbi_resolved = ""
        # self.virus_ncbi_resolved = ""
        # etc.

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
                    viruses_to_hosts_DAG.add_node(curr_interaction.virus_tax_id)   
                    viruses_to_hosts_DAG.add_node(curr_interaction.host_tax_id)
                    viruses_to_hosts_DAG.add_edge(curr_interaction.virus_tax_id, curr_interaction.host_tax_id)

        else: # line is empty, we have finished reading the file
            break
    # print("debug here to inspect data structures")
    return 0 # success! 
    

def RandomWalk(viruses_to_hosts_DAG: nx.DiGraph):
    virusHostMat = nx.to_numpy_array(viruses_to_hosts_DAG)
    #print(virusHostMat.shape)
    rowSize = len(virusHostMat)
    columnSize = rowSize 
    #columnSize = len(virusHostMat[0])
    probMatrix = np.zeros(virusHostMat.shape)
    #probArray = np.sum(virusHostMat, axis = 0) #calculates degree of node i
    probArray = np.zeros(rowSize)
    for rowSum in range(rowSize):
        sum = 0
        for colSum in range(columnSize):
            sum = sum + virusHostMat[rowSum][colSum]
        probArray [rowSum] = sum
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
                rwrArray [row] [col] = 0
            else:
                rwrArray [row] [col] = qMatrix [row] [col] + qMatrix [col] [row] 
    #rwrDict = dict(enumerate(rwrArray.flatten(), 1))
    
    #sortedRWR = sorted(rwrDict.items(), key = lambda val: val[1])
    #sortedDict = dict(sortedRWR)
    print("done")
    #print(sortedDict)
    #rowSizeRWR = len(rwrIndex)
    #colSizeRWR = len(rwrIndex[0])
    #for rowRWR in range(rowSizeRWR):
    #    for colRWR in range(colSizeRWR):
    #        max = 0
    #        if abs(rwrIndex [row] [col]) > max:
    #            max = rwrIndex [row] [col]
    #            colMin = col
         #display predicted false positive
         #edge is [row] [colMin]

    
    #edges = row * col - row
    #while edges > 0:
     #   max(rwrIndex)

        #find indices
        #print
        #change to 0





    #dMat = np.zeros(virusHostMat.shape)
    #for i in range(rowSize):
        #rowSum = 0
        #for j in range(columnSize):
            #if i != j: 
                #rowSum += virusHostMat[i][j]
        #dMat[i][i] = rowSum
    #dInv = np.linalg.inv(dMat)
    #matrix inverse




def main():
    host_to_set_of_viruses = {}
    virus_to_set_of_hosts = {}
    viruses_to_hosts_DAG = nx.DiGraph()
    parse_csv("Virion.csv", host_to_set_of_viruses, virus_to_set_of_hosts, viruses_to_hosts_DAG)
    RandomWalk(viruses_to_hosts_DAG)


if __name__ == "__main__":
    main()