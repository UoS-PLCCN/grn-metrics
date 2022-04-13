# GRN Metrics

- **What**: Scripts to get certain analytics for a given GRN.
- **Why**: These analytics can give us more insight on how the cell the GRN is modelling functions.
- **How**: Using the Python 3 [NetworkX](https://networkx.org/) library.

## Setup

### Installation

Requirements: [Python 3.7+](https://www.python.org/downloads/).

0. Clone the repository.
1. Create a virtual environment using Python's `venv` module.
   ```sh
   python -m venv .env
   ```
2. Activate the environment given your platform.
   - Windows:
     ```sh
     .\.env\scripts\activate
     ```
   - Linux / MacOS:
     ```sh
     source .env/scripts/activate
     ```
3. Install the requirements:
   ```sh
   python -m pip install -r requirements.txt
   ```

### Inserting a GRN of your choice

You will need to have the GRN exported as a [pickle](https://docs.python.org/3/library/pickle.html) file that contains a Python object representing a NetworkX `DiGraph`. Then put it into the `networks` directory. Make sure it follows the naming convention of `XXXXXXX.nxgraph.pkl`.

`XXXXXXX` will now be the network's name.

### Running

After setting up an environment and having it activated, simply run:

```sh
python get_metrics.py --help
```

to view the available commands.

For instance, to calculate the Rich Club Coefficients per node degree for the network in the paper, you can run:

```sh
python get_metrics.py rich-club-coefficient mapk_49
```
