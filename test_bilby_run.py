#!/usr/bin/env python
"""
test_bilby_run.py
==================

A complete, runnable example: inject a zero-noise binary neutron star signal,
install the Goldilocks joint tidal prior (via goldilocks_helper_functions.py
and the example grid shipped in example_grids/), and run a short bilby
parameter estimation with dynesty.

luminosity_distance is left as a FREE parameter here (not fixed at the
injected value) to demonstrate the real-event-style setup, where it's
resampled at every likelihood evaluation -- that's the hot loop
fast_conversion.py exists for (see its own module docstring): it replaces
bilby's astropy-based d_L -> z conversion with wcosmo's vectorized version,
and preserves the directly-sampled chirp_mass_source from being overwritten
by bilby's own frame conversion.

This is a DEMO configuration, not a science-quality run: nlive is small and
the frequency settings are chosen for speed. For a real analysis, raise
nlive substantially (the project this prior comes from used nlive=2048) and
swap in your own full-resolution Goldilocks grid.

Usage:
    python test_bilby_run.py
"""
import os

import bilby

from goldilocks_helper_functions import joint4D
from fast_conversion import generate_all_bns_parameters_preserve_source_mc, patch_bilby_permanently

GRID_FILE = "example_grids/example_grid_1.4_1.4_4D.npz"
OUTDIR = "outdir_test_run"
LABEL = "goldilocks_test"


def main():
    bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)

    # Patch d_L -> z before sampling; conversion_function only post-processes
    patch_bilby_permanently()

    injection_parameters = dict(
        mass_1=1.42, mass_2=1.38, chi_1=0.01, chi_2=0.01,
        luminosity_distance=40.0, theta_jn=1.7, psi=1.57, phase=0.0,
        geocent_time=1187008882.4, ra=3.16, dec=-0.08,
        lambda_1=450.0, lambda_2=600.0,
    )

    duration = 128
    sampling_frequency = 2048
    minimum_frequency = 40.0

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2_NRTidal",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
    start_time = injection_parameters["geocent_time"] + 2 - duration
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
    )
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    # --- Priors: Goldilocks joint tidal prior + standard extrinsic priors ---
    # luminosity_distance is left as BNSPriorDict's own default (free, not
    # fixed), so d_L is a live parameter -- see module docstring.
    priors = bilby.gw.prior.BNSPriorDict()
    for key in ["geocent_time", "dec", "ra", "theta_jn", "psi", "phase", "chi_1", "chi_2"]:
        priors[key] = injection_parameters[key]

    for key in ["chirp_mass_source", "chirp_mass", "mass_ratio",
                "lambda_tilde", "delta_lambda_tilde", "lambda_1", "lambda_2"]:
        try:
            priors.pop(key)
        except KeyError:
            pass

    active = ["chirp_mass_source", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]
    for jp in joint4D(active, {}, GRID_FILE):
        priors[jp.name] = jp

    priors["lambda_1"] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name="lambda_1")
    priors["lambda_2"] = bilby.core.prior.Constraint(minimum=0, maximum=10000, name="lambda_2")
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_2")

    # --- Likelihood ---
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
    )

    # --- Sampler (demo settings -- see module docstring) ---
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=250,
        naccept=20,
        sample="acceptance-walk",
        maxmcmc=2000,
        npool=1,
        outdir=OUTDIR,
        label=LABEL,
        injection_parameters=injection_parameters,
        conversion_function=generate_all_bns_parameters_preserve_source_mc,
        result_class=bilby.gw.result.CBCResult,
    )

    print(f"\nDone. Result saved under {os.path.abspath(OUTDIR)}/")
    try:
        result.plot_corner(parameters=["chirp_mass_source", "mass_ratio",
                                        "lambda_tilde", "delta_lambda_tilde"])
        print(f"Corner plot saved under {os.path.abspath(OUTDIR)}/")
    except Exception as e:  # corner plotting is optional -- don't fail the demo over it
        print(f"(corner plot skipped: {e})")


if __name__ == "__main__":
    main()
