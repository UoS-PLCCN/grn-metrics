"""random_nets.py
Code for generation of random reference networks.
"""
import random

import networkx as nx
import numpy as np


def make_equiv_ER(n_nodes: int, n_edges: int) -> nx.DiGraph:
    """
    Get an ER network with a certain number of nodes and number of edges.

    Args:
        n_nodes (int): the number of nodes to put into the ER network.
        n_edges (int): the number of edges to put into the ER network.

    Returns:
        networkx.DiGraph: the desired ER network as a NetworkX Directed Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(n_nodes)))
    while len(G.edges()) < n_edges:
        i = np.random.randint(n_nodes)
        j = np.random.randint(n_nodes)
        G.add_edge(i, j)
    while not nx.is_connected(G): #lots of garbage to make sure it's actually connected
        edges = G.edges()
        edge_to_bin = random.choice(list(edges))
        G.remove_edge(edge_to_bin[0], edge_to_bin[1])
        CC = list(nx.connected_components(G))
        i_to_connect = np.random.randint(len(CC))
        j_to_connect = np.random.randint(len(CC))
        while j_to_connect == i_to_connect:
            j_to_connect = np.random.randint(len(CC))
        i_CC = CC[i_to_connect]
        j_CC = CC[j_to_connect]
        i = random.choice(list(i_CC))
        j = random.choice(list(j_CC))
        G.add_edge(i, j)
    return G
