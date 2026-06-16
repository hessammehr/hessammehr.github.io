# Using `uv` projects on a network share

The wondeful [`uv`](https://astral.sh/uv) has a strong preference for virtual environments that are placed within a project tree rather than kept centrally *a la* conda. One unfortunate consequence of this is for projects stored on network shares, for us usually data analysis scripts and Jupyter/marimo notebooks for visualisation and as dashboards. Simply `uv add`ing `numpy` alone will dump heaps of tiny files into the folder and completely overwhelm our network share (normally OneDrive since we get 1 TB free as part of 365).

A little trick to avoid this situation, while still being able to use `pyproject.toml` to specify dependencies via `uv`. Here it goes.

1. `uv init --bare` to initialise your `pyproject.toml` if it doesn't already exist.
2. `uv add --no-sync <dependencies>`, the key is `--no-sync` so `uv` doesn't make a `.venv` folder and install packages there.
3. `uvx --with-requirements pyproject.toml <program that you want to run>`. Here are a few examples:
   * `uvx --with-requirements pyproject.toml marimo edit`
   * `uvx --with-requirements pyproject.toml jupyter lab`
   * `uvx --with-requirements pyproject.toml ipython`
   * `uvx --with-requirements pyproject.toml python script.py`

You might be tempted to run `uv run script.py`. Don't! That's going to create a virtual environment in the project folder. Either use the last example above or the nifty `--isolated` flag: `uv run --isolated script.py`.

And just like that, the most frustration-free Python workflow that I am aware of.
