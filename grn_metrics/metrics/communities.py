import networkx as nx
from grn_metrics.utils import force_undirected
from networkx.algorithms import cluster
from networkx.algorithms.community import greedy_modularity_communities, modularity


def get_communities(xNet: nx.Graph) -> list:
    """Get the communities within the GRN using Greedy Modularity Communities.

    Args:
        xNet (nx.Graph): the GRN as a NetworkX graph.

    Returns:
        list: the list of communities. Each community has genes associated with it.
    """
    xNet = force_undirected(xNet)
    communities = greedy_modularity_communities(xNet)
    return communities


def get_modularity(xNet: nx.Graph) -> float:
    _communities = get_communities(xNet)
    return modularity(xNet, _communities)


def get_global_clustering_coefficient(xNet: nx.Graph) -> float:
    xNet = force_undirected(xNet)
    triangles = cluster.triangles(xNet)
    n_vertices = len(triangles)
    n_existing_triangles = sum(triangles.values()) / 3
    n_possible_triangles = (n_vertices * (n_vertices - 1) * (n_vertices - 2)) / 6
    return n_existing_triangles / n_possible_triangles


def get_average_clustering(xNet: nx.Graph) -> float:
    return nx.algorithms.cluster.average_clustering(xNet)
