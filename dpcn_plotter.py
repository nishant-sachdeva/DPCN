import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


# Read the math overflow data
# it should contain 3 columns, nodeA, nodeB, timestamp

df = pd.read_csv("sx-mathoverflow-a2q.txt", sep=" ", names=["A", "B", "timestamp"])
df = df.sort_values(by=['timestamp'])

## DIVIDE THE DATA INTO 20 GROUPS BASED ON TIMESTAMPS

# calculate the partition size
num_rows = len(df)
number_of_paritions = 5
partition_size = num_rows // number_of_paritions

# divide the data into partitions
partitions = []
for i in range(1, number_of_paritions+1):
    end_index = i*partition_size
    partition = df.iloc[:end_index]
    partitions.append(partition)

## WE HAVE THE 20 PARTITIONS
# Now, we make 20 static graphs, simulate attack on them.

import networkx as nx

graphs = []
for p in partitions:
    G = nx.from_pandas_edgelist(p, source='A', target='B', edge_attr='timestamp')
    graphs.append(G)

# here we have our graphs

import matplotlib.pyplot as plt
import random


def simulate_attack(graph):
    attack_fraction = 0.001
    attack_so_far = 0

    number_of_attacks_so_far = 0


    captured_graphs = []

    while attack_so_far < 0.4:
        # plot the appropriate data points for the current graph
        # capture_graph_data(graph)

        # remove attack_fraction
        def attack():
            nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.in_degree(x), reverse=True)
            num_nodes_to_remove = int(attack_fraction * graph.number_of_nodes())
            nodes_to_remove = nodes_by_degree[:num_nodes_to_remove]
            graph.remove_nodes_from(nodes_to_remove)


        def failure():
            # do nothing
            num_nodes_to_remove = int(attack_fraction * graph.number_of_nodes())
            nodes_remove = np.random.choice(graph.nodes(), size = num_nodes_to_remove, replace = False)
            graph.remove_nodes_from(nodes_remove)
        
        attack()
        #failure()

        captured_graphs.append(graph.copy())
        attack_so_far += attack_fraction
        number_of_attacks_so_far += 1
        

    return captured_graphs

import multiprocessing

result_graphs = []

pool = multiprocessing.Pool()
pool = multiprocessing.Pool(processes=20)
inputs = graphs

result_graphs = pool.map(simulate_attack, inputs)

print("Done")

attack_fractions = []
attack = 0

while attack < 0.4:
    attack += 0.02
    attack_fractions.append(attack)


def plot_graphs(marked_graph_set):
    (i, graph_set) = marked_graph_set
    print("Plotting graphs for set {}".format(i))

    largest_clusters_attack = []
    other_clusters_attack = []
    diameters = []

    for (j, graph) in zip([i for i in range(len(graph_set))], graph_set):
        print("Plotting graph for set {} {}".format(i, j))
        connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        largest_connected_component = connected_components[0]

        ### DIAMETER OF LARGEST CLUSTER
        D = nx.average_shortest_path_length(
            graph.subgraph(largest_connected_component)
        )
        diameters.append(D)

        ### LARGEST CLUSTER SIZE RELATIVE TO THE GRAPH
        relative_size = len(largest_connected_component) // graph.number_of_nodes()
        largest_clusters_attack.append(relative_size)
        
        graph.remove_nodes_from(largest_connected_component)

        cluster_sizes = [len(c) for c in connected_components]

        ### CLUSTER SIZE DISTRIBUTION
        freq_dist = Counter(cluster_sizes)
        values = list(freq_dist.keys())
        counts = list(freq_dist.values())
        plt.scatter(values, counts)
        plt.xlabel('Cluster size')
        plt.ylabel('Frequency')
        plt.title('Cluster size distribution')
        plt.savefig("dpcn_attack_clusters/set{}/graph{}.png".format(i, j))

        other_cluster_sizes = cluster_sizes[1:]

        if len(other_cluster_sizes) > 0:
            other_cluster = np.mean(np.array(other_cluster_sizes))
        else:
            other_cluster = 0
        other_clusters_attack.append(other_cluster)

    
    plt.scatter(attack_fractions, largest_clusters_attack, label="Largest cluster (attack)")
    plt.scatter(attack_fractions, other_clusters_attack, label="Average other clusters (attack)")
    plt.xlabel("Fraction of nodes removed")
    plt.ylabel("Relative size of clusters")
    plt.savefig("dpcn_attack_plots/clusterSizes/plot_{}.png".format(i))
    # plt.show()

    plt.scatter(attack_fractions, diameters, label="Average other clusters (attack)")
    plt.xlabel("Fraction of nodes removed")
    plt.ylabel("Diameters of clusters")
    plt.savefig("dpcn_attack_plots/diameters/plot_{}.png".format(i))
    # plt.show()



newPool = multiprocessing.Pool()
newPool = multiprocessing.Pool(processes=20)
input = zip([i for i in range(len(result_graphs))], result_graphs)
plotted_graphs = newPool.map(plot_graphs, input)
print("Graph plotting done")
