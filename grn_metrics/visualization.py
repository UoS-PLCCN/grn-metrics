import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import networkx.drawing.nx_pydot as pydot
from matplotlib.figure import Figure

from grn_metrics.metrics.communities import get_communities


def _get_fig() -> tuple[Figure, object]:
    # No colours, WARNING: Super resolution, downscale after export
    fig = plt.figure(figsize=(25, 18), dpi=1000)
    ax = fig.add_subplot(1, 1, 1)
    return (fig, ax)


def _get_labels(graph: nx.Graph) -> dict:
    return {key: i + 1 for i, key in enumerate(graph.nodes())}


def draw_network(graph: nx.Graph, community_colours: bool = False, output: Path = None):
    _, ax = _get_fig()
    labels = _get_labels(graph)

    kwargs = {
        "pos": pydot.graphviz_layout(graph, "neato"),
        "with_labels": True,
        "labels": labels,
        "node_size": 1200,
        "node_color": "lightblue",
        "linewidths": 0.25,
        "font_size": 12,
        "ax": ax,
    }

    if community_colours:
        # Community colours, WARNING: Super resolution, downscale after export
        communities = [list(community) for community in get_communities(graph)]
        community_series = [
            i + 1
            for node in graph
            for i, community in enumerate(communities)
            if node in community
        ]
        colors = plt.get_cmap("Set3").colors
        node_colors = [colors[value - 1] for value in community_series]
        kwargs["node_color"] = node_colors

    nx.draw(graph, **kwargs)
    ax.annotate(
        "\n".join([f"{value}: {key}" for key, value in labels.items()]),
        xy=(0, 0.5),
        xycoords="axes fraction",
        va="center",
        fontsize=12,
    )

    if not output:
        output = Path(f"grn_vis{'_community_colours' if community_colours else ''}.png")

    plt.savefig(output, dpi=1000, bbox_inches="tight")
