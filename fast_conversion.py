"""
fast_conversion.py

Drop-in replacement for bilby's cosmological conversion chain, using wcosmo's
vectorized interpolation table instead of astropy's per-sample z_at_value
(Brent root-finding), and preserving directly-sampled chirp_mass_source from
being overwritten by bilby's own generate_source_frame_parameters.

Requires: numpy, pandas, astropy, wcosmo, bilby.

When you need this (and when you don't)
-----------------------------------------
This only matters when luminosity_distance is a FREE, sampled parameter
(e.g. a real-event analysis with a UniformSourceFrame prior), since the
speedup targets the d_L -> z conversion inside bilby's per-sample parameter
conversion during likelihood evaluation -- a real hot loop there.

If luminosity_distance is fixed (as in an injection-recovery study), d_L -> z
is computed once, not per sample, and this module isn't needed.

Usage
-----
    from fast_conversion import generate_all_bns_parameters_preserve_source_mc

    result = bilby.run_sampler(
        ...,
        conversion_function=generate_all_bns_parameters_preserve_source_mc,
    )

or patch bilby globally for the rest of the process:

    from fast_conversion import patch_bilby_permanently
    patch_bilby_permanently()
"""

import numpy as np
from pandas import Series
import astropy.units as u

import wcosmo
from wcosmo import Planck15

import bilby.gw.conversion as _bilby_conversion

_cosmo = Planck15


def luminosity_distance_to_redshift(distance, cosmology=None):
    """d_L [Mpc] -> z, via wcosmo's vectorized interpolation table (built
    once per batch, not once per sample -- O(1) vs. astropy's O(N) Brent
    calls). `cosmology` is accepted for API compatibility with bilby's own
    signature and ignored."""
    if isinstance(distance, Series):
        distance = distance.values
    distance = np.asarray(distance, dtype=float)
    return np.asarray(wcosmo.z_at_value(_cosmo.luminosity_distance, distance * u.Mpc))


def comoving_distance_to_redshift(distance, cosmology=None):
    """Comoving distance [Mpc] -> z, same approach as luminosity_distance_to_redshift."""
    if isinstance(distance, Series):
        distance = distance.values
    distance = np.asarray(distance, dtype=float)
    return np.asarray(wcosmo.z_at_value(_cosmo.comoving_distance, distance * u.Mpc))


def redshift_to_luminosity_distance(redshift, cosmology=None):
    return np.asarray(_cosmo.luminosity_distance(np.asarray(redshift)))


def redshift_to_comoving_distance(redshift, cosmology=None):
    return np.asarray(_cosmo.comoving_distance(np.asarray(redshift)))


def comoving_distance_to_luminosity_distance(distance, cosmology=None):
    return redshift_to_luminosity_distance(comoving_distance_to_redshift(distance))


def luminosity_distance_to_comoving_distance(distance, cosmology=None):
    return redshift_to_comoving_distance(luminosity_distance_to_redshift(distance))


def generate_source_frame_parameters(sample):
    """Fast drop-in for bilby.gw.conversion.generate_source_frame_parameters,
    using wcosmo instead of astropy z_at_value."""
    output_sample = sample.copy()
    output_sample["redshift"] = luminosity_distance_to_redshift(output_sample["luminosity_distance"])
    output_sample["comoving_distance"] = redshift_to_comoving_distance(output_sample["redshift"])
    for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
        if key in output_sample:
            output_sample[f"{key}_source"] = output_sample[key] / (1 + output_sample["redshift"])
    return output_sample


# Keys sampled directly in the source frame that must not be overwritten by
# generate_source_frame_parameters (which recomputes them from d_L). Extend
# if you also sample mass_1_source/mass_2_source directly.
_SOURCE_FRAME_SAMPLED_KEYS = ["chirp_mass_source"]

_PATCH_MAP = {
    "luminosity_distance_to_redshift": luminosity_distance_to_redshift,
    "comoving_distance_to_redshift": comoving_distance_to_redshift,
    "redshift_to_luminosity_distance": redshift_to_luminosity_distance,
    "redshift_to_comoving_distance": redshift_to_comoving_distance,
    "comoving_distance_to_luminosity_distance": comoving_distance_to_luminosity_distance,
    "luminosity_distance_to_comoving_distance": luminosity_distance_to_comoving_distance,
    "generate_source_frame_parameters": generate_source_frame_parameters,
}
_originals = {}


def _patch_bilby_conversion():
    for name, fast_fn in _PATCH_MAP.items():
        _originals[name] = getattr(_bilby_conversion, name)
        setattr(_bilby_conversion, name, fast_fn)


def _unpatch_bilby_conversion():
    for name, orig_fn in _originals.items():
        setattr(_bilby_conversion, name, orig_fn)
    _originals.clear()


def patch_bilby_permanently():
    """Replace bilby.gw.conversion's cosmological functions with the wcosmo
    versions for the lifetime of the process. Call once, before running the
    sampler, if you'd rather not wrap conversion_function explicitly."""
    for name, fast_fn in _PATCH_MAP.items():
        setattr(_bilby_conversion, name, fast_fn)


def generate_all_bns_parameters_preserve_source_mc(sample, likelihood=None, priors=None, npool=1):
    """
    Fast, source-frame-preserving drop-in for
    bilby.gw.conversion.generate_all_bns_parameters: snapshots any directly-
    sampled source-frame keys, runs bilby's conversion chain with wcosmo
    patched in, then restores the snapshotted values (bilby's own chain
    would otherwise silently recompute and overwrite chirp_mass_source from
    d_L, corrupting an EoS-informed posterior that sampled it directly).
    """
    import pandas as pd

    snapshots = {}
    for key in _SOURCE_FRAME_SAMPLED_KEYS:
        if isinstance(sample, pd.DataFrame):
            if key in sample.columns:
                snapshots[key] = sample[key].copy()
        elif isinstance(sample, dict) and key in sample:
            try:
                snapshots[key] = sample[key].copy()
            except AttributeError:
                snapshots[key] = sample[key]

    _patch_bilby_conversion()
    try:
        output_sample = _bilby_conversion.generate_all_bns_parameters(
            sample, likelihood=likelihood, priors=priors, npool=npool
        )
    finally:
        _unpatch_bilby_conversion()

    for key, value in snapshots.items():
        output_sample[key] = value
    return output_sample
