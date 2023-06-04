# plotMcpFiducials

Port of the Python version of `plotMcpFiducials` that was initially part of [mcpop](https://github.com/sdss/mcpop/blob/main/bin/plotMcpFiducials). The script has been ported to Python 3.9+ but is otherwise mostly unchanged.

## Installation

To install for production, use

```console
pip install .
```

For development

```console
poetry install
```

The script requires [mpc_fiducials](https://github.com/sdss/mc_fiducials) to be available at the `$MCP_FIDUCIALS_DIR` envvar.

The installation simply sets up a Python console script, `plotMcpFiducials` pointing to `src/plotmcpfiducials/plotmcpfiducials.py`.

## Testing

The script can be tested by running `test/alt.sh`, `test/az.sh`, `test/rot.sh` which use the `test/mcpFiducials-57996.dat` file to produce fiducial tables. The new tables can be compared with the `_reference.dat` tables under `test/`.
