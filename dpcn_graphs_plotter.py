def store_graph_into_file(marked_graph):
    (graph_set_id, graph_id, graph) = marked_graph
    folder_name = "set" + str(graph_set_id)
    file_name = "graph" + str(graph_id) + ".txt"
    file_path = 'dpcn_result_graphs_attack/' + folder_name + '/' + file_name
    # print(file_path)

    with open(file_path, 'w') as f:
        for (k, edge) in enumerate(graph.edges(data=True)):
            nodeA, nodeB, timestamp = edge[0], edge[1], edge[2]['timestamp'].iloc[k]
            # print(nodeA, nodeB, timestamp)
            f.write(f"{nodeA} {nodeB} {timestamp}\n")


def graphset_storage_function(marked_graph_set):
    (i, graph_set) = marked_graph_set

    folder_name = "set" + str(i)

    dir_path = 'dpcn_result_graphs_attack/' + folder_name
    
    print("storage starting for folder", dir_path)
    input = [(i, j, graph) for (j, graph) in zip([i for i in range(len(graph_set))], graph_set)]

    for marked_graph in input:
        store_graph_into_file(marked_graph)
    
    print("storage done for folder", dir_path)
    return 1
