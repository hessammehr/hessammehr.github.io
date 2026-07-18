# /// script
# dependencies = [
#     "jax[cuda13]==0.11.0",
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "numpyro==0.21.0",
#     "pandas==3.0.3",
#     "python-dotenv==1.2.2",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    app_title="Ancestral sampling with variational relaxation",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from dotenv import load_dotenv
    load_dotenv('.env')
    from numpyro import distributions as dist, sample, factor
    import jax
    import jax.numpy as jnp
    import time
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoDelta, AutoNormal
    from numpyro.infer.initialization import init_to_value
    import numpyro.optim as optim
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from jax.random import PRNGKey

    return (
        AutoDelta,
        AutoNormal,
        MCMC,
        NUTS,
        PRNGKey,
        Predictive,
        SVI,
        Trace_ELBO,
        dist,
        factor,
        init_to_value,
        jax,
        jnp,
        mo,
        np,
        optim,
        pd,
        plt,
        sample,
        sns,
        time,
    )


@app.cell(hide_code=True)
def _(PRNGKey, mo, sns):
    KEY = PRNGKey(0)
    N_PARTICLES = 1000
    CONCENTRATION = 1.0
    R = 0.010
    STRENGTH = 100.0

    sns.set_theme(
        context="notebook",
        style="ticks",
        font="Inter",
        rc={"svg.fonttype": "none", "savefig.format": "svg"},
    )

    def fig(f):
        return mo.as_html(f).style({"width": "max-content", "display": "block"})

    return CONCENTRATION, KEY, N_PARTICLES, R, STRENGTH


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ancestral sampling with variational relaxation
    I recently discovered a little trick that worked surprisingly well for sampling from a complex prior distribution. Unsure how rigorous it is mathematically but thought it would be worth sharing/documenting here. Here is how it goes.

    ## Easy and fast ancestral sampling
    I used to think that some form of MCMC was always necessary to sample from any probabilistic program. This turns out to be incorrect; a lot of the time you can essentially step through the program and at every sample site simply do exactly that: sample from the respective distribution, potentially parameterised by previous variables. Take the example below: nothing is stopping us from sampling `x` from a beta distribution, then sampling `y` from another beta. It is clean, simple and much faster than MCMC, and we don't have to worry about step size, warm-up, effective sample size, etc. I have learned that this straightline "execution" mode is called ancestral sampling, presumably because there can even be dependencies between variables and their ancestors. As long as each variable is still something you know how to sample from by the time you reach it in the execution trace, ancestral sampling is possible.
    """)
    return


@app.cell
def _(CONCENTRATION, dist, sample):
    def random_points(n, scenes=1):
        x = sample('x', dist.Beta(1.5, 1.5).expand((n, scenes)))
        y = sample('y', dist.Beta(CONCENTRATION + 100.0 * x, CONCENTRATION + 100.0 * x))

    return (random_points,)


@app.cell
def _(
    KEY,
    MCMC,
    NUTS,
    N_PARTICLES,
    Predictive,
    jax,
    mo,
    pd,
    random_points,
    time,
):
    # Ancestral sampling
    _t0 = time.time()
    p = Predictive(random_points, num_samples=1)
    ancestral_samples = p(KEY, N_PARTICLES)
    jax.block_until_ready(ancestral_samples)
    ancestral_seconds = time.time() - _t0
    ancestral_df = pd.DataFrame({k: v.ravel() for k, v in ancestral_samples.items()})

    # MCMC sampling
    _t0 = time.time()
    mcmc = MCMC(NUTS(random_points), num_warmup=500, num_samples=1, progress_bar=False)
    mcmc.run(KEY, N_PARTICLES)
    mcmc_samples = mcmc.get_samples()
    jax.block_until_ready(mcmc_samples)
    mcmc_seconds = time.time() - _t0
    mcmc_df = pd.DataFrame({k: v.ravel() for k, v in mcmc_samples.items()})

    mo.md(f"Ancestral sampling: **{ancestral_seconds:.2f} s** · MCMC: **{mcmc_seconds:.2f} s**")
    return ancestral_df, mcmc_df


@app.cell
def _(ancestral_df, mcmc_df, mo, plot_param_chooser, sns):
    a_fig = sns.jointplot(data=ancestral_df, x='x', y='y', **plot_param_chooser.value)
    m_fig = sns.jointplot(data=mcmc_df, x='x', y='y', color='C1', **plot_param_chooser.value)

    mo.vstack([
        plot_param_chooser,
        mo.hstack([mo.vstack([mo.md("#### Ancestral sampling"), a_fig], align='center'),
                   mo.vstack([mo.md("#### MCMC"), m_fig], align='center')])    
    ], align='center')
    return


@app.cell(hide_code=True)
def _(N_PARTICLES, mo):
    mo.md(rf"""
    This is the sort of model that you might imagine using to represent a cohort of particles (here `N_PARTICLES` = {N_PARTICLES}). Ancestral sampling and NUTS are mostly on par in terms of speed and more or less portray the expected joint distributions.

    Now, let's say on top of this the particles don't like to be too close to each other, like molecules in a gas or people in a room. We might model this with a repulsive potential or `factor` in `numpyro`.
    """)
    return


@app.cell
def _(CONCENTRATION, R, STRENGTH, dist, factor, jnp, sample):
    def repulsive_points(n, scenes=1):
        x = sample('x', dist.Beta(1.5, 1.5).expand((n, scenes)))
        y = sample('y', dist.Beta(CONCENTRATION + 100.0 * x, CONCENTRATION + 100.0 * x))
        pos = jnp.stack([x, y], axis=-1)
        d = jnp.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(-1) + 1e-12)
        overlap = jnp.clip(2 * R - d, min=0.0) * (1 - jnp.eye(n))[:, :, None]
        factor('repulsion', -STRENGTH * 0.5 * overlap.sum())

    return (repulsive_points,)


@app.cell
def _(
    KEY,
    MCMC,
    NUTS,
    N_PARTICLES,
    Predictive,
    jax,
    mo,
    pd,
    repulsive_points,
    time,
):
    # One configuration, sampled ancestrally (ignores the repulsion factor)
    _t0 = time.time()
    rp = Predictive(repulsive_points, num_samples=100)
    r_ancestral_samples = rp(KEY, N_PARTICLES)
    jax.block_until_ready(r_ancestral_samples)
    r_ancestral_seconds = time.time() - _t0
    r_ancestral_df = pd.DataFrame({'x': r_ancestral_samples['x'][0, :, 0],
                                   'y': r_ancestral_samples['y'][0, :, 0]})

    # One configuration from MCMC (honours the repulsion factor)
    _t0 = time.time()
    r_mcmc = MCMC(NUTS(repulsive_points),
                  num_warmup=500, num_samples=100, progress_bar=False)
    r_mcmc.run(KEY, N_PARTICLES)
    r_mcmc_samples = r_mcmc.get_samples()
    jax.block_until_ready(r_mcmc_samples)
    r_mcmc_seconds = time.time() - _t0
    r_mcmc_df = pd.DataFrame({'x': r_mcmc_samples['x'][0, :, 0],
                              'y': r_mcmc_samples['y'][0, :, 0]})

    mo.md(f"Ancestral sampling: **{r_ancestral_seconds:.2f} s** · MCMC: **{r_mcmc_seconds:.1f} s**")
    return r_ancestral_samples, r_mcmc_samples


@app.cell
def _(
    mo,
    pd,
    plot_param_chooser,
    r_ancestral_samples,
    r_mcmc_samples,
    sample_slider,
    sns,
):
    _r_sample = sample_slider.value
    _r_ancestral_df = pd.DataFrame({
        "x": r_ancestral_samples["x"][_r_sample, :, 0],
        "y": r_ancestral_samples["y"][_r_sample, :, 0],
    })
    _r_mcmc_df = pd.DataFrame({
        "x": r_mcmc_samples["x"][_r_sample, :, 0],
        "y": r_mcmc_samples["y"][_r_sample, :, 0],
    })
    ra_fig = sns.jointplot(data=_r_ancestral_df, x="x", y="y", **plot_param_chooser.value)
    rm_fig = sns.jointplot(data=_r_mcmc_df, x="x", y="y", color="C1", **plot_param_chooser.value)
    ra_fig.ax_joint.set(xlim=(0,1), ylim=(0,1))
    rm_fig.ax_joint.set(xlim=(0,1), ylim=(0,1))

    mo.vstack([
        mo.hstack([plot_param_chooser, sample_slider]),
        mo.hstack([mo.vstack([mo.md("#### Ancestral sampling"), ra_fig], align="center"),
                   mo.vstack([mo.md("#### MCMC"), rm_fig], align="center")])
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The added repulsion breaks our assumption about being able to sample from each site sequentially. Now that we are sampling from an arbitrary joint distribution, ancestral sampling using `Predictive` simply ignores the `factor`, producing an identical result to before. I had to fiddle with MCMC settting to make it work within a reasonable amount of time, but the end result shows that it is clearly working, even if painfully slow. There are some concerning signs: note how the marginal of $x$ is spread to the left. This is not unexpected but as the case often is with MCMC, it has the smell of something that needs a bit more poking before I trust it. What if we could use ancestral sampling to generate lots of proposal initial distributions that could then be "relaxed" to reflect the repulsive potential. Let's see ...
    """)
    return


@app.cell
def _(
    KEY,
    N_PARTICLES,
    Predictive,
    SVI,
    Trace_ELBO,
    guide_chooser,
    init_to_value,
    jax,
    mo,
    optim,
    repulsive_points,
    time,
):
    guide_type, iterations = guide_chooser.value
    _t0 = time.time()
    relax_init = Predictive(repulsive_points, num_samples=1)(KEY, N_PARTICLES, scenes=100)
    relax_init_x, relax_init_y = relax_init['x'][0], relax_init['y'][0]
    relax_guide = guide_type(
        repulsive_points,
        init_loc_fn=init_to_value(values={'x': relax_init_x, 'y': relax_init_y}),
    )
    relax_svi = SVI(repulsive_points, relax_guide, optim.Adam(1e-4), Trace_ELBO())
    relax_result = relax_svi.run(KEY, iterations, N_PARTICLES, scenes=100, progress_bar=False)
    relaxed_posterior = relax_guide.sample_posterior(KEY, relax_result.params, N_PARTICLES, 100, sample_shape=(1,))
    relaxed_x, relaxed_y = relaxed_posterior['x'].squeeze(), relaxed_posterior['y'].squeeze()

    jax.block_until_ready((relaxed_x, relaxed_y))
    relax_seconds = time.time() - _t0

    mo.md(f"Ancestral + variational relaxation of **100** configurations: "
          f"**{relax_seconds:.1f} s** (~{relax_seconds / 100:.2f} s per configuration)")
    return relax_guide, relax_init_x, relax_init_y, relaxed_x, relaxed_y


@app.cell
def _(
    guide_chooser,
    mo,
    pd,
    plot_param_chooser,
    relax_init_x,
    relax_init_y,
    relaxed_x,
    relaxed_y,
    scene_slider,
    sns,
):
    _s = scene_slider.value
    proposal_df = pd.DataFrame({'x': relax_init_x[:, _s], 'y': relax_init_y[:, _s]})
    relaxed_df = pd.DataFrame({'x': relaxed_x[:, _s], 'y': relaxed_y[:, _s]})
    pj_fig = sns.jointplot(data=proposal_df, x='x', y='y', **plot_param_chooser.value)
    rv_fig = sns.jointplot(data=relaxed_df, x='x', y='y', color='C2', **plot_param_chooser.value)
    pj_fig.ax_joint.set(xlim=(0,1), ylim=(0,1))
    rv_fig.ax_joint.set(xlim=(0,1), ylim=(0,1))

    mo.vstack([
        mo.hstack([guide_chooser, plot_param_chooser, scene_slider]),
        mo.hstack([mo.vstack([mo.md("#### Ancestral proposal"), pj_fig], align='center'),
                   mo.vstack([mo.md("#### Relaxed"), rv_fig], align='center')])
    ], align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Much better! Still squeezed in the middle at high $x$, where the prior dominates but nicely spaced out in the flat middle, as expected. Quantitative comparison of the target log density shows much higher likelihoods than the ancestral starting point and potentially even MCMC. Note this is not necessarily a good thing; the relaxation may be (and indeed is) pushing the distribution towards the maximum a posteriori (MAP) point. In this case, the fact that we deliberately ran a very small number of steps and learning rate, _i.e._ aborted SVI early was precisely to prevent this.
    """)
    return


@app.cell
def _(mcmc_logp, mo, plt, proposal_logp, relaxed_logp, sns):
    _f, _ax = plt.subplots(figsize=(6.5, 3.6))
    sns.histplot(proposal_logp, color='C0', label='Ancestral proposals', kde=True, stat='count', ax=_ax)
    sns.histplot(mcmc_logp, color='C1', label='MCMC', kde=True, stat='count', ax=_ax)
    sns.histplot(relaxed_logp, color='C2', label='Relaxed ancestral', kde=True, stat='count', ax=_ax)
    _ax.set(xlabel='Target log-density', ylabel='# samples')
    _ax.legend(frameon=False)
    sns.despine(_f)

    mo.vstack([
        mo.md("### Unnormalised target log density"),
        _f,
    ], align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other potential applications
    Some problems don't _look_ like `factor` but are effectively the same thing. Any observed variables, for instance, including between latent sites. I've recently realised that a generative process that I wrote about a couple of years ago [here](https://hessammehr.github.io/blog/posts/2024-12-26-locally-constrained.html) has a name: a [Markov random field](https://en.wikipedia.org/wiki/Markov_random_field). This type of process is essentially equivalent to making adding observations between latent variables, for example positing that $x_n - x_{n-1} \sim \mathcal{N}(\mu, \sigma)$ for certain or all $n$. Again, we're adding a potential/factor to the joint probability distribution.

    Anyway this was a lot of fun to discover, and I have yet to find out whether it is commonly used or known about. Hopefully interesting/useful!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Update: More on caveats
    There are some caveats to this method that I'm aware of, and some that I am not! The main one as noted above is that SVI + the `AutoDelta` guide alone will simply squeeze the distribution towards MAP. I think we're doing quite well here, but that's because I chose a number of SVI steps that was just enough to relax the added factor without significant drift towards MAP. Too many steps may be detectable as the joint log-density distribution of samples starts to collapse to a spike, like in the plot below. I actually had to update the notebook to work with $\text{Beta}(\cdot, \cdot)$ parameters $>1$ because between 0 and 1 the beta distribution has singularities as 0 and/or 1 which complicate the story.

    I have briefly experimented with using `AutoNormal` instead of `AutoDelta`, where the former's entropy term will presumably prevent this type of collapse when maximising the evidence lower bound (ELBO). In fact, you can switch the guide to `AutoNormal` above and see how it affects the results. There is definitely no collapse even after a large number of SVI steps, though from a quick visual check I'm not sure overall whether the goal of particles avoiding each other is fulfilled in this case.
    """)
    return


@app.cell
def _(compute_svi_logp_path, mo, plt, sns):
    _logp_fig, _logp_ax = plt.subplots(figsize=(7.2, 4.4))

    svi_logp_path = compute_svi_logp_path((0, 100, 300, 1000, 10000))

    sns.histplot(
        data=svi_logp_path,
        x="Target log density",
        hue="SVI steps",
        bins=50,
        stat="density",
        common_norm=False,
        common_bins=True,
        linewidth=1.5,
        ax=_logp_ax,
    )
    _logp_ax.set_ylabel("Density")
    sns.despine(_logp_fig)

    mo.vstack([
        mo.md("### Joint log density across 100 configurations"),
        _logp_fig,
    ], align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    plot_param_options = {
        'Scatter': dict(s=10, alpha=0.25, marginal_kws={'stat': 'density'}),
        'KDE': dict(kind='kde', fill=True, levels=20, bw_method=0.25, marginal_kws={'clip': (0, 1), 'cut': 0})
    }

    plot_param_chooser = mo.ui.dropdown(plot_param_options, value='Scatter', label='Plot type')
    scene_slider = mo.ui.slider(0, 99, value=0, label='Configuration')
    sample_slider = mo.ui.slider(0, 99, value=0, label='Sample')
    return plot_param_chooser, sample_slider, scene_slider


@app.cell(hide_code=True)
def _(
    jax,
    np,
    r_mcmc_samples,
    relax_init_x,
    relax_init_y,
    relaxed_x,
    relaxed_y,
    repulsive_points,
):
    from numpyro.infer.util import log_density

    def config_logp(xc, yc):
        n = xc.shape[0]
        ld, _ = log_density(repulsive_points, (n,), {'scenes': 1},
                            {'x': xc[:, None], 'y': yc[:, None]})
        return ld

    _logp_batch = jax.jit(jax.vmap(config_logp, in_axes=(1, 1)))
    relaxed_logp = np.asarray(_logp_batch(relaxed_x, relaxed_y))
    proposal_logp = np.asarray(_logp_batch(relax_init_x, relax_init_y))
    mcmc_logp = np.asarray(_logp_batch(
        r_mcmc_samples['x'][..., 0].T,
        r_mcmc_samples['y'][..., 0].T,
    ))
    return config_logp, mcmc_logp, proposal_logp, relaxed_logp


@app.cell(hide_code=True)
def _(
    KEY,
    N_PARTICLES,
    SVI,
    Trace_ELBO,
    config_logp,
    jax,
    np,
    optim,
    pd,
    relax_guide,
    repulsive_points,
):
    def compute_svi_logp_path(checkpoints):
        relax_svi = SVI(repulsive_points, relax_guide, optim.Adam(3e-5), Trace_ELBO())
        _state = relax_svi.init(KEY, N_PARTICLES, scenes=100)
        _batched_logp = jax.jit(jax.vmap(config_logp, in_axes=(1, 1)))

        @jax.jit
        def _advance(_current_state, _count):
            return jax.lax.fori_loop(
                0,
                _count,
                lambda _, _inner_state: relax_svi.update(
                    _inner_state, N_PARTICLES, scenes=100
                )[0],
                _current_state,
            )

        _frames = []
        _previous = 0
        for _checkpoint in checkpoints:
            _state = _advance(_state, _checkpoint - _previous)
            _params = relax_svi.get_params(_state)
            relaxed_posterior = relax_guide.sample_posterior(KEY, _params, N_PARTICLES, 100, sample_shape=(1,))
            _values = np.asarray(
                _batched_logp(relaxed_posterior['x'].squeeze(), relaxed_posterior['y'].squeeze())
            )
            _frames.append(
                pd.DataFrame(
                    {
                        "SVI steps": str(_checkpoint),
                        "Target log density": _values,
                    }
                )
            )
            _previous = _checkpoint
        return pd.concat(_frames, ignore_index=True)

    return (compute_svi_logp_path,)


@app.cell(hide_code=True)
def _(AutoDelta, AutoNormal, mo):
    guide_options = {
        'Delta': (AutoDelta, 300),
        'Normal': (AutoNormal, 10000)
    }

    guide_chooser = mo.ui.dropdown(guide_options, value='Delta', label='Guide type')
    return (guide_chooser,)


if __name__ == "__main__":
    app.run()
