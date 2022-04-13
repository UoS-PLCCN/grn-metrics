"""utils.py

Various utility functions.
"""
from typing import Iterable

import click
import networkx as nx
import numpy as np


def progressbar(iterable: Iterable, label: str):
    fill_char = click.style("#", fg="green")
    empty_char = click.style("-", fg="white", dim=True)
    return click.progressbar(
        iterable=iterable, label=label, fill_char=fill_char, empty_char=empty_char
    )


def force_undirected(graph: nx.Graph):
    if type(graph) == nx.DiGraph:
        graph = graph.to_undirected()
    return graph


def force_directed(graph: nx.Graph):
    if type(graph) != nx.DiGraph:
        graph = graph.to_directed()
    return graph


Z_STAR_MAP = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}


def get_confidence_interval(
    mean: float, std: float, n: int, confidence_level: int = 0.95
) -> float:
    if confidence_level not in Z_STAR_MAP.keys():
        raise ValueError(
            f"Please provide a confidence level in: [{', '.join(Z_STAR_MAP.keys())}]"
        )

    z_star = Z_STAR_MAP[confidence_level]
    interval = z_star * (std / np.sqrt(n))
    return interval
