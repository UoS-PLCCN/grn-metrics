import functools
import glob
import pickle
from pathlib import Path
from typing import Callable, Union

import click
import networkx as nx
import numpy as np
import pandas as pd
from yaspin import yaspin

from grn_metrics.metrics import centrality, communities
from grn_metrics.metrics.reference_nets import (gen_degree_preserving_network,
                                                gen_er_network)
from grn_metrics.utils import get_confidence_interval, progressbar
from grn_metrics.visualization import draw_network


def _get_networks() -> list[Path]:
    return [Path(g) for g in glob.glob("networks/*.nxgraph.pkl")]


def _get_network_path(network: str) -> Union[Path, None]:
    return next((p for p in _get_networks() if p.name == f"{network}.nxgraph.pkl"), None)


def validate_network(ctx: click.core.Context, param: click.core.Argument, value: str) -> nx.Graph:
    if type(value) != str:
        raise click.BadParameter("The network value needs to be a string.")

    network_path = _get_network_path(value)
    if not network_path:
        raise click.BadParameter(f"Invalid network name: {value}. Make sure {value}.nxgraph.pkl exists in the networks directory.")
    
    with open(network_path, "rb") as f:
        GRN = pickle.load(f)
    
    return GRN


REFERENCE_NET_GENERATORS = {
    "ER": gen_er_network,
    "RandomDegreePreserving": gen_degree_preserving_network
}
REFERENCE_NET_CHOICES = list(REFERENCE_NET_GENERATORS.keys()) + ["all"]


def _process_compare(ctx: click.core.Context, param: click.core.Argument, value: str) -> list[str]:
    if not value:
        return []
    
    return [value] if value != "all" else REFERENCE_NET_GENERATORS.keys()


def _compute_reference_nets(network: nx.Graph, compare: str, comparison_network_n: int) -> dict:
    ret = {}
    for reference_network_type in compare:
        reference_networks = []
        with progressbar(range(comparison_network_n), label=f"Generating {reference_network_type} networks...") as reference_networks_n:
            for _ in reference_networks_n:
                reference_networks.append(REFERENCE_NET_GENERATORS[reference_network_type](network))
        ret[reference_network_type] = reference_networks
    click.echo("\n", nl=False)
    return ret


def _get_metric_stats(
    network: nx.Graph, function: Callable, metric_name: str,
    reference_networks: dict, confidence_level: float = 0.95
):
    with yaspin(text=f"Computing {metric_name}", color="green") as spinner:
        metric = function(network)
        spinner.ok()
    
    ret = {
        "value": metric
    }

    for reference_network_type, reference_net_list in reference_networks.items():  # This could be empty, btw.
        n = len(reference_net_list)
        recorded_values = np.zeros(n)
        with progressbar(range(n), f" - Computing {metric_name} for {reference_network_type} networks...") as _reference_networks:
            for i in _reference_networks:
                recorded_values[i] = function(reference_net_list[i])
        mean = np.mean(recorded_values)
        std = np.std(recorded_values)
        confidence_interval = get_confidence_interval(mean, std, len(recorded_values), confidence_level=confidence_level)
        ret[reference_network_type] = {
            "mean": mean, "std": std, "confidence_interval": confidence_interval
        }
    click.echo("\n", nl=False)
    
    return ret

def _print_metric_stats(stats: dict, metric_name: str, compare: list[str]):
    click.echo(f"{metric_name}: " + click.style(stats["value"], fg="green"))
    for reference_network_type in compare:
        click.echo(f"- {reference_network_type}:")
        click.echo(f"- - Mean value: {click.style(stats[reference_network_type]['mean'], fg='green')}")
        click.echo(f"- - Standard Deviation: {click.style(stats[reference_network_type]['std'], fg='green')}")
        click.echo(f"- - Confidence Interval: ±{click.style(stats[reference_network_type]['confidence_interval'], fg='green')}")
    click.echo("\n", nl=False)


@click.group()
def metrics():
    pass


def comparison_params(func: Callable):
    @click.option(
        "-c", "--compare",
        type=click.Choice(REFERENCE_NET_CHOICES, case_sensitive=False),
        callback=_process_compare,
        help="The type of network(s) to compare against. All chooses all of them."
    )
    @click.option(
        "-cn", "--comparison-network-n", type=click.IntRange(2, 1000), default=1000, help="The number of reference networks to generate."
    )
    @click.option("-cf", "--confidence-level", type=float, default=0.95, help="The confidence interval %.")
    # @click.option("-v", "--visualize", type=click.Path(), help="Path to save comparison visualization to.")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper



@metrics.command("small-worldness")
@click.argument("network", callback=validate_network)
@comparison_params
def small_worldness(network: nx.Graph, compare: list[str], comparison_network_n: int, confidence_level: float, visualize: Path = None):
    """
    Get the small worldness of a network.
    """
    _reference_networks = _compute_reference_nets(network, compare, comparison_network_n)
    small_worldness_stats = _get_metric_stats(network, centrality.small_worldness, "Small-worldness", _reference_networks, confidence_level)
    _print_metric_stats(small_worldness_stats, "Small-worldness", compare)


@metrics.command("communities")
@click.argument("network", callback=validate_network)
def community_calculation(network: nx.Graph):
    """
    Get the different communities in the network.
    """
    with yaspin(text="Computing communities", color="green") as spinner:
        network_communities = communities.get_communities(network)
        spinner.ok()

    click.echo("\n", nl=False)
    for i, community in enumerate(network_communities):
        click.echo(click.style(f"Community {i + 1}: ", bold=True) + ", ".join(list(community)))


@metrics.command("clustering")
@click.argument("network", callback=validate_network)
@comparison_params
def clustering_metrics(network: nx.Graph, compare: list[str], comparison_network_n: int, confidence_level: float):
    _reference_networks = _compute_reference_nets(network, compare, comparison_network_n)
    modularity_stats = _get_metric_stats(network, communities.get_modularity, "Modularity", _reference_networks, confidence_level)
    global_clustering_coefficient_stats = _get_metric_stats(
        network, communities.get_global_clustering_coefficient, "Global Clustering Coefficient", _reference_networks, confidence_level
    )
    average_clustering_stats = _get_metric_stats(
        network, communities.get_average_clustering, "Average Clustering", _reference_networks, confidence_level
    )

    click.echo("\n", nl=False)
    _print_metric_stats(modularity_stats, "Modularity", compare)
    _print_metric_stats(global_clustering_coefficient_stats, "Global Clustering Coefficient", compare)
    _print_metric_stats(average_clustering_stats, "Average Clustering", compare)


@metrics.command("degree-metrics")
@click.argument("network", callback=validate_network)
@comparison_params
def degree_metrics(network: nx.Graph, compare: list[str], comparison_network_n: int, confidence_level: float):
    _reference_networks = _compute_reference_nets(network, compare, comparison_network_n)
    average_degree_stats = _get_metric_stats(network, centrality.get_average_degree, "Average Degree", _reference_networks, confidence_level)
    characteristic_path_length_stats = _get_metric_stats(network, centrality.get_characteristic_path_length, "Characteristic Path Length", _reference_networks, confidence_level)

    click.echo("\n", nl=False)
    _print_metric_stats(average_degree_stats, "Average Degree", compare)
    _print_metric_stats(characteristic_path_length_stats, "Characteristic Path Length", compare)


@metrics.command("rich-club-coefficient")
@click.argument("network", callback=validate_network)
@click.option("-o", "--output", type=click.Path(), help="Path to the CSV file to save to")
@click.option("-n", "--n-rand", type=int, help="The number of random reference networks", default=1000)
def rich_club_coefficient(network: nx.Graph, output: click.Path, n_rand: int):
    rcc = centrality.get_rich_club_coefficient(network, n_rand)
    
    click.echo("\n", nl=False)
    if not output:
        click.echo("Degree: Normalized Value")
        click.echo("------------------------")
        for degree, coefficient in rcc.items():
            click.echo(f"{degree}: {coefficient}")
    else:
        pd.DataFrame({
            "Degree": [degree for degree in rcc],
            "Normalized Rich Club Coefficient Value": [coefficient for _, coefficient in rcc.items()]
        }).to_csv(output, index=False)
        click.echo(f"Exported the Rich Club Coefficients to: " + click.style(output, fg="green"))


@metrics.command("centrality-metrics")
@click.argument("network", callback=validate_network)
@click.option("-o", "--output", required=True, help="Path to the CSV file to save to")
def centrality_metrics(network: nx.Graph, output: click.Path):
    with yaspin(text="Computing centrality metrics", color="green") as spinner:
        # TODO spinner for every centrality metric
        _metrics = centrality.get_centrality_metrics(network)
        spinner.ok()

    _metrics.to_csv(output, index=False)
    click.echo(f"Exported the centrality metrics to: " + click.style(output, fg="green"))


@metrics.command("visualize")
@click.argument("network", callback=validate_network)
@click.option("-c", "--colours", type=click.Choice(["Communities"]), default=None, help="The colouring of the nodes in the graph.")
@click.option("-o", "--output", type=click.Path(), help="The path to the output image.")
def visualize(network: nx.Graph, colours: str = None, output: Path = None):
    with yaspin(text="Visualizing the network...", color="green") as spinner:
        draw_network(network, colours == "Communities", output)
        spinner.ok()

if __name__ == "__main__":
    metrics()
