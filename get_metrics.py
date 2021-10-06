import glob
import pickle
from pathlib import Path
from typing import Union

import click
import networkx as nx
import pandas as pd
from yaspin import yaspin

from grn_metrics.metrics import centrality, communities


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


@click.group()
def metrics():
    pass


@metrics.command("small-worldness")
@click.argument("network", callback=validate_network)
def small_worldness(network: nx.Graph):
    """
    Get the small worldness of a network.
    """
    _small_worldness = centrality.small_worldness(network)
    click.echo(f"Small-worldness: " + click.style(_small_worldness, fg="green"))


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
def clustering_metrics(network: nx.Graph):
    with yaspin(text="Computing Modularity", color="green") as spinner:
        modularity = communities.get_modularity(network)
        spinner.ok()

    with yaspin(text="Computing Global Clustering Coefficient", color="green") as spinner:
        global_clustering_coefficient = communities.get_global_clustering_coefficient(network)
        spinner.ok()
    
    with yaspin(text="Get Average Clustering", color="green") as spinner:
        average_clustering = communities.get_average_clustering(network)
        spinner.ok()

    click.echo("\n", nl=False)
    click.echo("Modularity: " + click.style(modularity, fg="green"))
    click.echo("Global Clustering Coefficient: " + click.style(global_clustering_coefficient, fg="green"))
    click.echo("Average Clustering: " + click.style(average_clustering, fg="green"))


@metrics.command("degree-metrics")
@click.argument("network", callback=validate_network)
def degree_metrics(network: nx.Graph):
    with yaspin(text="Computing Average Degree", color="green") as spinner:
        average_degree = centrality.get_average_degree(network)
        spinner.ok()

    with yaspin(text="Computing Characteristic Path Length", color="green") as spinner:
        characteristic_path_length = centrality.get_characteristic_path_length(network)
        spinner.ok()

    click.echo("\n", nl=False)
    click.echo("Average Degree" + click.style(average_degree, fg="green"))
    click.echo("Characteristic Path Length: " + click.style(characteristic_path_length, fg="green"))


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


if __name__ == "__main__":
    metrics()
