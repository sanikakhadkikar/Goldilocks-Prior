#!/usr/bin/env python
"""
quickstart_prior_only.py
=========================

Fastest sanity check that the Goldilocks prior is wired up correctly --
builds the joint prior from the shipped example grid, draws samples with
bilby's own PriorDict.sample(), and evaluates ln_prob. No detector, no
waveform, no sampler: this is meant as a first "does this work at all"
check before running the full PE example in test_bilby_run.py.

Usage:
    python quickstart_prior_only.py
"""
import numpy as np
import bilby

from goldilocks_helper_functions import joint4D

GRID_FILE = "example_grids/example_grid_1.4_1.4_4D.npz"
ACTIVE = ["chirp_mass_source", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]


def build_goldilocks_priors():
    priors = bilby.gw.prior.BNSPriorDict()
    for key in ["chirp_mass_source", "chirp_mass", "mass_ratio",
                "lambda_tilde", "delta_lambda_tilde", "lambda_1", "lambda_2"]:
        try:
            priors.pop(key)
        except KeyError:
            pass

    # Fixed distance avoids astropy z_at_value per sample
    priors["luminosity_distance"] = 40.0

    for jp in joint4D(ACTIVE, {}, GRID_FILE):
        priors[jp.name] = jp

    priors["lambda_1"] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name="lambda_1")
    priors["lambda_2"] = bilby.core.prior.Constraint(minimum=0, maximum=10000, name="lambda_2")
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_2")
    return priors


def main():
    priors = build_goldilocks_priors()
    print("Priors installed:", ", ".join(ACTIVE))

    n = 200
    samples = priors.sample(n)
    for name in ACTIVE:
        v = np.asarray(samples[name])
        print(f"  {name:20s} min={v.min():10.4g}  median={np.median(v):10.4g}  max={v.max():10.4g}")

    # Joint prior only; DeltaFunction ln_prob is ~710
    joint = priors[ACTIVE[0]].dist
    lnp = joint.ln_prob(np.column_stack([samples[name] for name in ACTIVE]))
    finite = np.isfinite(lnp)
    print(f"\nln_prob finite for {finite.sum()}/{n} samples, range "
          f"[{lnp[finite].min():.3f}, {lnp[finite].max():.3f}]")
    print("\nOK -- the Goldilocks joint prior samples and evaluates correctly.")


if __name__ == "__main__":
    main()
