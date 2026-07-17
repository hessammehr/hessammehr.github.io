# /// script
# dependencies = [
#     "jax==0.11.0",
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpyro==0.21.0",
#     "pandas==3.0.3",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from numpyro import distributions as dist, sample
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    import seaborn as sns
    import pandas as pd
    from matplotlib import pyplot as plt
    from jax.random import PRNGKey

    return MCMC, NUTS, PRNGKey, Predictive, dist, mo, pd, sample, sns


@app.cell(hide_code=True)
def _(PRNGKey, mo, sns):
    KEY = PRNGKey(0)

    sns.set_theme(
        context="notebook",
        style="ticks",
        font="Inter",
        rc={"svg.fonttype": "none", "savefig.format": "svg"},
    )

    def fig(f):
        return mo.as_html(f).style({"width": "max-content", "display": "block"})

    return (KEY,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ancestral sampling with variational relaxation
    I recently discovered a little trick that worked surprisingly well for sampling from a complex prior distribution. Unsure how rigorous it is mathematically but thought it would be worth sharing/documenting here. Here is how it goes.

    ## Easy and fast ancestral sampling
    I used to think that some form of MCMC was necessary to sample from any probabilistic program. This turns out to be incorrect; a lot of the time you can mentally step through the program and at every sample site simply do exactly that: sample from the respective distribution. Take the example below: nothing is stopping us from sampling `x` from a beta distribution, then sampling `y` from another beta. It is clean, simple and much faster than MCMC, and we don't have to worry about step size, warm-up, effective sample size, etc. I have learned that this straightline "execution" mode is called ancestral sampling, presumably because there can even be dependencies between the variables.
    """)
    return


@app.cell
def _(KEY, Predictive, dist, pd, sample, sns):
    def random_points(n, scenes=1):
        x = sample('x', dist.Beta(2.0, 2.0).expand((n, scenes)))
        y = sample('y', dist.Beta(0.01 + x , 0.01 + x))

    # Ancestral sampling
    p = Predictive(random_points, num_samples=100)
    ancestral_samples = p(KEY, 50)
    ancestral_df = pd.DataFrame({k: v.ravel() for k, v in ancestral_samples.items()})
    a_fig = sns.jointplot(data=ancestral_df, x='x', y='y', s=5, alpha=0.25)
    return a_fig, random_points


@app.cell
def _(KEY, MCMC, NUTS, a_fig, mo, pd, random_points, sns):
    mcmc = MCMC(NUTS(random_points), num_warmup=500, num_samples=100, progress_bar=False)
    mcmc.run(KEY, 50)
    mcmc_samples = mcmc.get_samples()
    mcmc_df = pd.DataFrame({k: v.ravel() for k, v in mcmc_samples.items()})
    mo.hstack([a_fig, sns.jointplot(data=mcmc_df, x='x', y='y', color='C1', s=5, alpha=0.25)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here ancestral
    """)
    return


if __name__ == "__main__":
    app.run()
