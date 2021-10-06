"""utils.py

Various utility functions.
"""
from typing import Iterable

import click
import networkx as nx


def progressbar(iterable: Iterable, label: str):
    fill_char = click.style("#", fg="green")
    empty_char = click.style("-", fg="white", dim=True)
    return click.progressbar(iterable=iterable, label=label, fill_char=fill_char, empty_char=empty_char)

def force_undirected(graph: nx.Graph):
    if type(graph) == nx.DiGraph:
        graph = graph.to_undirected()
    return graph

def force_directed(graph: nx.Graph):
    if type(graph) != nx.DiGraph:
        graph = graph.to_directed()
    return graph
