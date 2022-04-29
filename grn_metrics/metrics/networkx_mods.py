"""networkx_mods.py
Functions ripped out of NetworkX and modified for a certain purpose.
"""
from itertools import accumulate

import networkx as nx
import numpy as np

from ..utils import progressbar
from .reference_nets import gen_degree_preserving_network, gen_er_network


def sigma(G, niter=100, nrand=10, seed=None):
    """
    Original: https://networkx.org/documentation/stable/_modules/networkx/algorithms/smallworld.html#sigma
    Mod: use an ER reference network instead of a random one.
    """
    randMetrics = {"C": [], "L": []}
    # with progressbar(range(nrand), label="Computing small-worldness...") as random_nets:
    random_nets = range(nrand)
    for _ in random_nets:
        Gr = gen_er_network(G)  # HACK
        randMetrics["C"].append(nx.transitivity(Gr))
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))

    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)

    return sigma


def rich_club_coefficient(graph, Q=100, n_rand=100, seed=None):
    """
    Original: https://networkx.org/documentation/stable/_modules/networkx/algorithms/richclub.html#rich_club_coefficient
    Mod: Use the average of `n_rand` degree-preserving random graphs as the reference instead of just 1.
    """
    rc_raw = _compute_RC(graph)
    rcran_sum = {}
    for degree, _ in rc_raw.items():
        rcran_sum[degree] = 0
    with progressbar(
        range(n_rand), label="Calculating Rich Club Coefficient..."
    ) as reference_nets:
        for _ in reference_nets:
            R = gen_degree_preserving_network(graph, Q, seed)
            rcran = _compute_RC(R)
            for ran_deg, ran_rcc in rcran.items():
                rcran_sum[ran_deg] += ran_rcc
    for degree, rcc in rcran_sum.items():
        rcran_sum[degree] = rcc / n_rand
    rcc_norm = {}
    for degree, rcc in rc_raw.items():
        rcc_norm[degree] = rcc / rcran_sum[degree]
    return rcc_norm


def _compute_RC(G):
    """
    Original: https://networkx.org/documentation/stable/_modules/networkx/algorithms/richclub.html#rich_club_coefficient
    Mod: none
    """
    deghist = nx.degree_histogram(G)
    total = sum(deghist)
    # Compute the number of nodes with degree greater than `k`, for each
    # degree `k` (omitting the last entry, which is zero).
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    # Create a sorted list of pairs of edge endpoint degrees.
    #
    # The list is sorted in reverse order so that we can pop from the
    # right side of the list later, instead of popping from the left
    # side of the list, which would have a linear time cost.
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()), reverse=True)
    ek = G.number_of_edges()
    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc
