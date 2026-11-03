# Goldilocks tidal-deformability prior for bilby

Goldilocks prior is informed by a broad semi-parametric ensemble of equations of state, accounts for the joint correlations among the chirp mass, mass ratio, and both tidal deformability parameters, and is tailored to detector sensitivity.

This repository is a standalone, public-release version of the "Goldilocks"
joint tidal-deformability prior used in [paper reference / link here]. It
lets you use the Goldilocks prior with [bilby](https://git.ligo.org/lscsoft/bilby)
without installing any private/internal package -- only `numpy`, `scipy`,
and `bilby` (tested with the plain `pip install bilby` release 2.7.1) are
required. The joint-prior classes here build only on bilby's public
`BaseJointPriorDist` and `JointPrior`.

This release is deliberately **consumption-only**: it contains everything
needed to *use* an existing Goldilocks prior grid in a bilby analysis, but
nothing related to *building* one from an equation-of-state population.
See the paper for how the full-resolution grids were constructed.

## What's here

```
goldilocks_helper_functions.py   -- the joint-prior machinery; the only file
                                     you need to import if luminosity_distance
                                     is fixed (not sampled)
fast_conversion.py                -- optional: only needed if luminosity_distance
                                     is a free/sampled parameter -- see
                                     "Why fast_conversion.py?" below
example_grids/
  example_grid_1.4_1.4_4D.npz    -- full-resolution example grid (50x50x100x49),
                                     for the equal-mass (1.4, 1.4) Msun case
quickstart_prior_only.py         -- sanity check (no sampler)
test_bilby_run.py                -- full example: injection + short bilby PE run
requirements.txt                 -- pip dependencies
```

## Installation

```bash
pip install numpy scipy bilby
```

is all you need for `goldilocks_helper_functions.py` itself. If you also
want to use `fast_conversion.py` (only needed when `luminosity_distance` is
a free/sampled parameter -- see below), additionally:

```bash
pip install wcosmo pandas astropy
```

Then just put the files you need (and, if you want to run the examples
as-is, the `example_grids/` folder) somewhere importable -- there's nothing
to build or compile.

## Quickstart

```bash
python quickstart_prior_only.py
```

This builds the joint prior from the shipped example grid, draws samples,
and evaluates their log-probability -- no detector, no waveform, no
sampler involved. Building the 4D inverse-CDF tables for the example grid
takes about 15 s and peaks at 2.5 GB of memory (the tables are
50x50x100x2048 float32); sampling and `ln_prob` afterwards are
milliseconds. The script fixes `luminosity_distance`, see "Why
fast_conversion.py?" below for what happens if it is sampled instead.

```bash
python test_bilby_run.py
```

This is a complete, runnable example: it injects a zero-noise binary
neutron star signal, installs the Goldilocks prior, and runs a short bilby
parameter estimation with `dynesty`. `luminosity_distance` is left as a
free/sampled parameter, so it uses `fast_conversion.py`'s
`generate_all_bns_parameters_preserve_source_mc` as bilby's
`conversion_function` -- see "Why fast_conversion.py?" below. It uses
demo-scale settings (small `nlive`, short duration) so it finishes in a few
minutes rather than hours -- see the comments at the top of the script for
what to change for a science-quality run (in particular: swap in your own
full-resolution grid, and raise `nlive` substantially).

## Using it in your own script

```python
import bilby
from goldilocks_helper_functions import joint4D

priors = bilby.gw.prior.BNSPriorDict()

# Remove bilby's own defaults for whatever the joint prior will replace.
for key in ["chirp_mass_source", "chirp_mass", "mass_ratio",
            "lambda_tilde", "delta_lambda_tilde", "lambda_1", "lambda_2"]:
    try:
        priors.pop(key)
    except KeyError:
        pass

active = ["chirp_mass_source", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]
fixed = {}  # nothing fixed -- use the full 4D joint prior
for jp in joint4D(active, fixed, "path/to/your_grid_4D.npz"):
    priors[jp.name] = jp

# Physicality constraints -- individual tidal deformabilities and component
# masses must be physical; adjust bounds to match how your grid was built.
priors["lambda_1"] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name="lambda_1")
priors["lambda_2"] = bilby.core.prior.Constraint(minimum=0, maximum=10000, name="lambda_2")
priors["mass_1"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_1")
priors["mass_2"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_2")
```

`priors` is now a normal bilby `PriorDict` -- pass it to `bilby.run_sampler`
exactly as you would with any other prior.

### 2D / 3D variants

`joint4D` is the standard way to use this prior (all four parameters free).
`joint2D`/`joint3D` are also provided, for grids where one or more axes are
fixed (e.g. `mass_ratio` fixed to 1 for an equal-mass-only 2D grid over
`(chirp_mass_source, lambda_tilde)`):

```python
from goldilocks_helper_functions import joint2D
for jp in joint2D(["chirp_mass_source", "lambda_tilde"], {"mass_ratio": 1.0}, "your_2D_grid.npz"):
    priors[jp.name] = jp
```

Note: `active_keys`/`fixed_params` only tell `joint2D/3D/4D` which axes to
expect and which to fix -- the actual axis *order* is always taken from the
grid file itself, and which axes exist at all is determined by what the
grid file contains, not by what you pass in.

## Using your own grid

Any `.npz` file with:
- a `dims` array naming its axes (any of `chirp_mass`/`chirp_mass_source`,
  `mass_ratio`, `lambda_tilde`, `delta_lambda_tilde` -- see `NAME_ALIASES`
  in `goldilocks_helper_functions.py` for all recognized spellings),
- one 1D array per axis (strictly increasing), matching a name in `dims`,
- a single density array `P` with shape `(len(axis_0), len(axis_1), ...)`
  in the same order as `dims`,

will work directly with `joint2D`/`joint3D`/`joint4D`. The shipped example
grid is the project's actual full-resolution (1.4, 1.4) M&#8857; grid
(50x50x100x49) -- for real analyses, swap in the grid for your own event/mass
point.

## Why the data-informed sampling tables?

A gridded joint prior that builds its inverse-CDF sampling tables on a
grid spaced uniformly in *cumulative probability* has a problem. When a prior's probability mass sits far from the region
that actually matters physically (as this tidal prior's does), that
leaves very few table points where they're needed, producing a coarse
"staircase" in the real sampling density -- invisible to `ln_prob()` (which
uses a separate, correctly-resolved interpolator) but visible as spurious
clustering in posteriors. `goldilocks_helper_functions.py` avoids this by
tying the table's resolution to the axis's own (already non-uniform) node
spacing instead. See the module's own docstring for the full technical
writeup, and the paper.

## Why fast_conversion.py?

Only relevant if `luminosity_distance` is a free/sampled parameter (i.e. a
real-event analysis, not a fixed-distance injection study). bilby's default
`generate_all_bns_parameters` converts luminosity distance to redshift with
astropy's per-sample `z_at_value`, which is called on every likelihood
evaluation and dominates runtime. `fast_conversion.py` swaps in `wcosmo`'s
vectorized, table-based `z_at_value` (~10,000x faster) and additionally
preserves a directly-sampled `chirp_mass_source` from being overwritten by
bilby's own frame-conversion chain. Two calls are needed, and
`test_bilby_run.py` shows both:

- `patch_bilby_permanently()` before building priors or running the
  sampler. This is what speeds up the per-likelihood conversion (and
  prior sampling with constraints). Without it, every prior draw and every
  likelihood evaluation goes through astropy's root finder, which turns a
  200-sample prior draw into a 20-minute job.
- `generate_all_bns_parameters_preserve_source_mc` as `conversion_function`
  to `bilby.run_sampler`, which post-processes the result samples.

## What's deliberately not included

- Anything related to *building* a grid (equation-of-state population
  sampling, node-scheme generation, Gaussian-process/spectral EOS
  machinery, etc.) -- this release is consumption-only.
- A monkeypatch for a separate, unrelated bilby/nessai bug affecting
  `JointPrior.ln_prob` with length-1 array inputs (only matters if you use
  `nessai` as your sampler; dynesty is unaffected). If you hit this with
  nessai, the fix is a small `ln_prob` override -- get in touch or see the
  paper's supplementary code.
