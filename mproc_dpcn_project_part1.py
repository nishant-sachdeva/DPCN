# -*- coding: utf-8 -*-
"""MPROC_DPCN_PROJECT_PART1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fFWimNIb1JfG344XjnqlM8N-m2zfwFY-

Here are the steps we are looking to implement.

1. Get the Math overflow data, present it, and see what it looks like
    - We assume that it will contain A, B, timestamp. 
    - Verify the same

2. Break it into 20 or so sets, by time.

3. Plot static graphs for these 20 sets.

4. Simulate attacks on these static graphs.
    - We want 3 graphs, for three different stages of the attack
    - Beginning of the attack, some point in the middle, and then when the critical fraction is reached.

5. Collect these three graphs and send them on forward to get to the next stage.

# New Section
"""

import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing


class createGraphs:
    def __init__(self, mathoverFlowDataPath, numPartitions, attackFraction, attackFractionMax, attackFunction, store_directory):
        self.mathoverFlowDataPath = mathoverFlowDataPath
        self.numPartitions = numPartitions
        self.attackFraction = attackFraction
        self.attackFractionMax = attackFractionMax
        self.attackFunction = attackFunction
        self.directory_name = store_directory
        self.result_graphs = []


    # Read the math overflow data
    # it should contain 3 columns, nodeA, nodeB, timestamp
    def read_data_into_graphs(self):
        df = pd.read_csv(
            self.mathoverFlowDataPath, sep=" ", names=["A", "B", "timestamp"]
        ).sort_values(by=['timestamp'])

        # calculate the partition size
        partition_size = len(df) // self.numPartitions

        # divide the data into partitions
        partitions = [
            df.iloc[:(i*partition_size)] for i in range(1, self.numPartitions+1)
        ]
        
        graphs = [nx.from_pandas_edgelist(p, source='A', target='B', edge_attr='timestamp') for p in partitions]

        return graphs


    def conduct_attack(self, graph):
        def attack(graph, attack_fraction):
            nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.degree[x], reverse=True)
            num_nodes_to_remove = int(attack_fraction * graph.number_of_nodes())
            nodes_to_remove = nodes_by_degree[:num_nodes_to_remove]
            graph.remove_nodes_from(nodes_to_remove)
            return graph


        def failure(graph, attack_fraction):
            # do nothing
            num_nodes_to_remove = int(attack_fraction * graph.number_of_nodes())
            nodes_remove = np.random.choice(graph.nodes(), size = num_nodes_to_remove, replace = False)
            graph.remove_nodes_from(nodes_remove)
            return graph



        def simulate_attack(graph, attack_fraction, attack_fraction_max, attack_function):
            attack_so_far = 0
            captured_graphs = []

            while attack_so_far < attack_fraction_max:
                # plot the appropriate data points for the current graph
                # capture_graph_data(graph)
                if attack_function == "attack":
                    graph = attack(graph, attack_fraction)
                elif attack_function == "failure":
                    graph = failure(graph, attack_fraction)
                
                captured_graphs.append(graph.copy())
                attack_so_far += attack_fraction

            return captured_graphs
    
        return simulate_attack(
            graph, 
            self.attackFraction, 
            self.attackFractionMax,
            self.attackFunction
        )

    def generate_graphs(self):
        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=20)
        inputs = self.read_data_into_graphs()
        self.result_graphs = pool.map(self.conduct_attack, inputs)
        pool.close()
        pool.join()
        print("Created Graphs")

    def store_graph_into_file(self, marked_graph):
        (graph_set_id, graph_id, graph) = marked_graph
        folder_name = "set" + str(graph_set_id)
        file_name = "graph" + str(graph_id) + ".txt"

        file_path = self.directory_name + folder_name + '/' + file_name

        with open(file_path, 'w') as f:
            for edge in graph.edges(data=True):
                nodeA, nodeB, timestamp = edge[0], edge[1], edge[2]['timestamp']
                print(nodeA, nodeB, timestamp)
                f.write(f"{nodeA} {nodeB} {timestamp}\n")
    
    def store_graph_set(self, marked_graph_set):
        (i, graph_set) = marked_graph_set

        folder_name = "set" + str(i)

        dir_path = self.directory_name + folder_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        input = [(i, j, graph) for (j, graph) in zip([i for i in range(len(graph_set))], graph_set)]

        for marked_graph in input:
            self.store_graph_into_file(marked_graph)
        
        print("storage done for folder", dir_path)
        return 1

    def store_graphs(self):
        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=20)
        input = zip([i for i in range(len(self.result_graphs))], self.result_graphs)

        _ = pool.map(self.store_graph_set, input)
        pool.close()
        pool.join()
        print("Done")


if __name__ == "__main__":
    attack_types = ["failure"]

    for attack_type in attack_types:
        graphSet = createGraphs(
            mathoverFlowDataPath = "sx-mathoverflow-a2q.txt",
            numPartitions = 5,
            attackFraction = 0.02,
            attackFractionMax = 0.4,
            attackFunction = attack_type,
            store_directory = 'dpcn_result_graphs_' + attack_type + '/'
        )
        graphSet.generate_graphs()
        graphSet.store_graphs()
