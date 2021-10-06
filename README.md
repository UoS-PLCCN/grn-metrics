# GRN Metrics
- **What**: Scripts to get certain analytics for a given GRN.
- **Why**: These analytics can give us more insight on how the cell the GRN is modelling functions.
- **How**: Using the Python 3 [NetworkX](https://networkx.org/) library.

## Setup
### Installation
Requirements: [Python 3.7+](https://www.python.org/downloads/).

1. Create a virtual environment using Python's `venv` module.
   ```sh
   python -m venv .env
   ```
2. Install the requirements:
   ```sh
   python -m pip install -r requirements.txt
   ```

### Inserting a GRN of your choice
You will need to have the GRN exported as a [pickle](https://docs.python.org/3/library/pickle.html) file that contains a Python object representing a NetworkX graph. Then put it into the `networks` directory. Make sure it follows the naming convention of `XXXXXXX.nxgraph.pkl`.

`XXXXXXX` will now be the network's name.

### Running
`TODO`