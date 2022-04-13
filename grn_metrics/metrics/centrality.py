from multiprocessing.sharedctypes import Value
import networkx as nx
import pandas as pd
from igraph import Graph

from ..utils import force_undirected
from .networkx_mods import rich_club_coefficient, sigma


def get_centrality_metrics(network: nx.Graph) -> pd.DataFrame:
    iNet = Graph.Adjacency((nx.to_numpy_matrix(network) > 0).tolist())
    cent_mat = pd.DataFrame()
    cent_mat["Names"] = network.nodes()
    cent_mat["Node Degree"] = Graph.degree(iNet)
    cent_mat["2nd Centrality"] = nx.second_order_centrality(
        force_undirected(network)
    ).values()
    cent_mat["Alpha Central"] = Graph.alpha(iNet)
    cent_mat["KATZ"] = nx.katz_centrality_numpy(network).values()
    cent_mat["Sub-Graph Centrality"] = nx.subgraph_centrality(
        force_undirected(network)
    ).values()
    cent_mat["Betweenness"] = Graph.betweenness(iNet)
    cent_mat["Page Rank"] = Graph.pagerank(iNet)
    cent_mat["Closeness"] = Graph.closeness(iNet)
    cent_mat["Strength"] = Graph.strength(iNet)
    cent_mat["Authority"] = Graph.authority_score(iNet)
    cent_mat["Hub Score"] = Graph.hub_score(iNet)
    cent_mat["Constraint"] = Graph.constraint(iNet)
    cent_mat["Eigen Centrality"] = Graph.eigenvector_centrality(iNet)
    cent_mat["Clustering"] = nx.algorithms.clustering(network).values()
    return cent_mat


def small_worldness(network: nx.DiGraph, nrand: int = 1000) -> float:
    network = force_undirected(network)
    return sigma(network, nrand=nrand)


def get_average_degree(network: nx.Graph):
    total_n_edges = network.number_of_edges()
    total_n_nodes = network.number_of_nodes()
    return total_n_edges / total_n_nodes


def get_characteristic_path_length(network: nx.Graph):
    return nx.average_shortest_path_length(network)


def get_rich_club_coefficient(network, nrand=1000) -> dict:
    undirected_network = force_undirected(network)
    return rich_club_coefficient(undirected_network, n_rand=nrand)
