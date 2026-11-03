"""
goldilocks_helper_functions.py
===============================

Standalone functions for using a precomputed "Goldilocks" tidal-deformability
prior grid with bilby. This module has NO dependency on any private/internal
package -- only `numpy`, `scipy`, and `bilby` (>= 2.7.1) are required.

What this is
------------
The Goldilocks prior is a joint density over
    (chirp_mass_source, mass_ratio, lambda_tilde, delta_lambda_tilde)
built from a population of neutron-star equations of state, and stored as a
gridded `.npz` file (one 1D array per axis, plus a single N-D array `P` of
the density evaluated on the grid). This module turns such a grid file into
bilby `JointPrior` objects that can be dropped straight into a
`bilby.gw.prior.BNSPriorDict` (or any bilby `PriorDict`) and used exactly
like any other bilby prior -- sampled from, and evaluated with `ln_prob`.

This file does NOT include anything related to *building* a grid from an
equation-of-state population -- only to *consuming* an existing grid file.
See the accompanying paper/data release for how the shipped example grid(s)
were produced.

Quickstart
----------
    import bilby
    from goldilocks_helper_functions import joint4D

    priors = bilby.gw.prior.BNSPriorDict()
    for key in ["chirp_mass_source", "chirp_mass", "mass_ratio",
                "lambda_tilde", "delta_lambda_tilde", "lambda_1", "lambda_2"]:
        try:
            priors.pop(key)
        except KeyError:
            pass

    active = ["chirp_mass_source", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]
    fixed = {}
    for jp in joint4D(active, fixed, "example_grids/example_grid_1.4_1.4_4D.npz"):
        priors[jp.name] = jp

    priors["lambda_1"] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name="lambda_1")
    priors["lambda_2"] = bilby.core.prior.Constraint(minimum=0, maximum=10000, name="lambda_2")
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=1.0, maximum=3.0, name="mass_2")

See `test_bilby_run.py` in this folder for a complete, runnable example
(injection + a short bilby PE run), and `quickstart_prior_only.py` for a
near-instant sanity check that doesn't run a sampler at all.

Why the patched interpolated-distribution classes exist
---------------------------------------------------------
A straightforward gridded joint prior builds the inverse-CDF
lookup tables that `_rescale()` uses (i.e. what the sampler actually draws
from) on a `ugrid` spaced UNIFORMLY IN CUMULATIVE PROBABILITY. When a prior's
probability mass is concentrated far from a physically important region
(e.g. this project's tidal prior pushes most mass to lambda_tilde ~ 900-1600,
far from where any real signal's lambda_tilde is expected to sit), very few
of those uniformly-spaced table points land inside the low-lambda_tilde
region that actually matters. Linear interpolation between such sparse
points produces a coarse *staircase* in the real sampling density -- this is
invisible to `ln_prob()` (which uses a separate, correctly-resolved
interpolator) and only shows up as spurious clustering/apparent bimodality
in posteriors.

The fix (`_data_informed_ugrid`, used by the three distribution classes
below) ties
the inverse-CDF table's resolution to the axis's OWN node spacing (which is
already finer wherever the grid's own construction cared about resolution)
instead of to probability mass -- removing the staircase without needing an
arbitrarily large table. These classes also fix a separate floating-point
bug where linear interpolation can overshoot an axis's exact bounds by a few
ULPs, which otherwise raises spurious `ValueError`s for samples landing very
close to a hard prior boundary.

The three distribution classes below are self-contained: they build on
bilby's public `BaseJointPriorDist` / `JointPrior` only, so a plain
`pip install bilby` is all that is required. No forked or patched bilby.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RegularGridInterpolator, interp1d

from bilby.core.prior import JointPrior
from bilby.core.prior.joint import BaseJointPriorDist
from bilby.core.utils import random

# ---------------------------------------------------------------------------
# Grid file IO: load a .npz (or text .dat) grid file, identify its axes by
# name, and hand back the density array + axis arrays in a fixed physical
# order. No bilby dependency in this section.
# ---------------------------------------------------------------------------

# Every consumer of a grid file sees the parameter axes in this order,
# restricted to whichever parameters are present in the file, regardless of
# the order the file itself stores them in.
PHYSICAL_ORDER = [
    "chirp_mass_source",
    "mass_ratio",
    "lambda_tilde",
    "delta_lambda_tilde",
]

# Recognized axis-name spellings -> canonical physical name.
NAME_ALIASES = {
    "chirp_mass": "chirp_mass_source", "chirp_mass_source": "chirp_mass_source",
    "mc": "chirp_mass_source", "chirpmass": "chirp_mass_source",
    "mass_ratio": "mass_ratio", "q": "mass_ratio", "massratio": "mass_ratio",
    "lambda_tilde": "lambda_tilde", "lambdatilde": "lambda_tilde",
    "lt": "lambda_tilde", "lambda_t": "lambda_tilde",
    "delta_lambda_tilde": "delta_lambda_tilde", "deltalambdatilde": "delta_lambda_tilde",
    "dlt": "delta_lambda_tilde", "delta_lambda_t": "delta_lambda_tilde",
}
PROB_TOKENS = {"prob", "probability", "p", "density", "weight", "weights"}

# Cache of loaded grid files, keyed by file path, so repeated joint2D/3D/4D
# calls against the same file (e.g. once per waveform/spin combo in a batch
# of run scripts) don't re-read it from disk each time.
GRID_CACHE = {}


def parse_header(data_file):
    """For a text `.dat` grid file: read its first line as a whitespace-
    separated header, map each token to its canonical physical name, and
    return the parameter columns in enforced physical order plus the prob
    column index. Returns (ordered_params, prob_col), where ordered_params
    is a list of (canonical_name, file_column_index)."""
    with open(data_file) as f:
        header = f.readline().lstrip("#").strip()
    tokens = header.split()
    if not tokens:
        raise ValueError(f"Empty header in {data_file}")

    param_cols = {}
    prob_col = None
    for i, tok in enumerate(tokens):
        t = tok.strip().lower()
        if t in PROB_TOKENS:
            if prob_col is not None:
                raise ValueError(f"Multiple prob columns in header: {header!r}")
            prob_col = i
        elif t in NAME_ALIASES:
            canon = NAME_ALIASES[t]
            if canon in param_cols:
                raise ValueError(f"Duplicate parameter '{canon}' in header: {header!r}")
            param_cols[canon] = i
        else:
            raise ValueError(
                f"Unrecognized column '{tok}' in header of {data_file}. "
                f"Known params: {sorted(set(NAME_ALIASES.values()))}; "
                f"prob tokens: {sorted(PROB_TOKENS)}."
            )

    if prob_col is None:
        raise ValueError(f"No prob column found in header {header!r}.")
    if not param_cols:
        raise ValueError(f"No parameter columns found in header: {header!r}")

    ordered_params = [(n, param_cols[n]) for n in PHYSICAL_ORDER if n in param_cols]
    return ordered_params, prob_col


def _canonicalize(raw_name, data_file):
    t = str(raw_name).strip().lower()
    if t not in NAME_ALIASES:
        raise ValueError(
            f"Unrecognized array/column name '{raw_name}' in {data_file}. "
            f"Known params: {sorted(set(NAME_ALIASES.values()))}."
        )
    return NAME_ALIASES[t]


def _load_npz_grid(data_file):
    """Load the structure-of-arrays .npz format: one 1D array per axis
    (keyed by its name in the stored `dims` array) plus a single N-D density
    array `P` in that same axis order. Returns (file_keys, grids, P_nd) with
    file_keys/P_nd re-ordered into PHYSICAL_ORDER."""
    with np.load(data_file, allow_pickle=False) as z:
        raw_names = [str(x) for x in z["dims"]]
        raw_grids = {n: np.asarray(z[n]) for n in raw_names}
        P_raw = np.asarray(z["P"])

    canon_names = [_canonicalize(n, data_file) for n in raw_names]
    if len(set(canon_names)) != len(canon_names):
        raise ValueError(f"Duplicate parameter after aliasing in {data_file}: {canon_names}")
    grids = {canon: raw_grids[raw] for raw, canon in zip(raw_names, canon_names)}

    phys_shape_check = tuple(grids[n].size for n in canon_names)
    if phys_shape_check != P_raw.shape:
        raise ValueError(
            f"File {data_file}: axis sizes {phys_shape_check} don't match P's shape {P_raw.shape}."
        )

    file_keys = [n for n in PHYSICAL_ORDER if n in canon_names]
    perm = [canon_names.index(n) for n in file_keys]
    P_nd = np.transpose(P_raw, perm)
    return file_keys, grids, P_nd


def _load_text_grid(data_file):
    """Load a flat text `.dat` grid: one header line naming the columns
    (see `parse_header`), followed by one row per grid cell."""
    ordered_params, prob_col = parse_header(data_file)
    data = np.genfromtxt(data_file, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    file_keys = [n for n, _ in ordered_params]
    grids = {n: np.unique(data[:, c]) for n, c in ordered_params}

    phys_shape = tuple(grids[n].size for n in file_keys)
    total = int(np.prod(phys_shape))
    if total != data.shape[0]:
        raise ValueError(
            f"File {data_file}: grid product {phys_shape} = {total} rows expected, "
            f"but file has {data.shape[0]} rows -- file is incomplete/truncated or "
            f"not a full rectangular grid over its unique axis values."
        )

    file_order = sorted(ordered_params, key=lambda t: t[1])
    file_order_names = [n for n, _ in file_order]
    file_order_shape = tuple(grids[n].size for n in file_order_names)
    P_file = data[:, prob_col].reshape(file_order_shape)
    perm = [file_order_names.index(n) for n in file_keys]
    P_nd = np.transpose(P_file, perm)
    return file_keys, grids, P_nd


def get_sorted_grid_and_prob(active_keys, fixed_params, data_file):
    """
    Load a grid file (.npz or text .dat, auto-detected by extension) and
    return its density array + axis arrays in enforced physical order:
        chirp_mass_source, mass_ratio, lambda_tilde, delta_lambda_tilde
    (restricted to whichever of these the file actually contains).

    `fixed_params` (dict of {name: value}) slices the grid at the nearest
    node along any of those axes present in the file (silently ignored for
    axes the file doesn't have). `active_keys` is accepted for API symmetry
    with joint2D/3D/4D but does not affect the result here -- which axes
    remain "active" is simply "every file axis not in fixed_params".

    Returns (sliced_P, grids, ordered_active_keys).
    """
    if data_file not in GRID_CACHE:
        if str(data_file).endswith(".npz"):
            file_keys, grids, P_nd = _load_npz_grid(data_file)
        else:
            file_keys, grids, P_nd = _load_text_grid(data_file)
        GRID_CACHE[data_file] = {"file_keys": file_keys, "grids": grids, "P_nd": P_nd}

    info = GRID_CACHE[data_file]
    file_keys = info["file_keys"]
    grids = info["grids"]
    P_nd = info["P_nd"].copy()

    fixed_canon = {}
    for key, val in fixed_params.items():
        canon = NAME_ALIASES.get(str(key).lower(), key)
        fixed_canon[canon] = val

    slices = [slice(None)] * len(file_keys)
    for canon, val in fixed_canon.items():
        if canon in file_keys:
            g = grids[canon]
            idx = int(np.argmin(np.abs(g - val)))
            slices[file_keys.index(canon)] = idx
    sliced_P = P_nd[tuple(slices)]

    ordered_active_keys = [k for k in file_keys if k not in fixed_canon]
    return sliced_P, grids, ordered_active_keys


def check_ndim(ordered_keys, expected, func_name):
    if len(ordered_keys) != expected:
        raise ValueError(
            f"{func_name} expects {expected} active parameter(s), but "
            f"{len(ordered_keys)} remain after fixing: {ordered_keys}. Check "
            f"fixed_params and the data file's own axes."
        )


def slice_grid_to_bounds(sliced_P, grids_in_order, ordered_keys, bounds):
    """
    Genuinely restrict one or more active axes to a user-specified [lo, hi]
    sub-range by dropping grid nodes outside it, BEFORE the JointPrior is
    built -- unlike bilby's own `bounds=` constructor argument on the
    Two/Three/FourDInterpolatedDist classes, which only sets the resulting
    Prior's cosmetic .minimum/.maximum display attributes and has no effect
    on actual sampling or ln_prob.

    `bounds`: dict of {axis_name: (lo, hi)}, applied only to names present in
    `ordered_keys` (ignored otherwise); None/empty is a no-op. Keeps one
    extra node beyond [lo, hi] on each side where available, and raises if
    fewer than 2 nodes would remain for an axis.
    """
    if not bounds:
        return sliced_P, grids_in_order

    grids_out = list(grids_in_order)
    P_out = sliced_P
    for axis_idx, key in enumerate(ordered_keys):
        if key not in bounds:
            continue
        lo, hi = bounds[key]
        g = grids_out[axis_idx]
        inside = np.flatnonzero((g >= lo) & (g <= hi))
        if len(inside):
            lo_idx = max(0, inside.min() - 1)
            hi_idx = min(len(g) - 1, inside.max() + 1)
        else:
            lo_idx = max(0, int(np.searchsorted(g, lo)) - 1)
            hi_idx = min(len(g) - 1, int(np.searchsorted(g, hi)))
        if hi_idx - lo_idx < 1:
            raise ValueError(
                f"bounds={(lo, hi)!r} for {key!r} leaves fewer than 2 grid nodes "
                f"(axis spans [{g.min()}, {g.max()}], {len(g)} nodes total)."
            )
        idx_slice = slice(lo_idx, hi_idx + 1)
        grids_out[axis_idx] = g[idx_slice]
        P_out = np.moveaxis(np.moveaxis(P_out, axis_idx, 0)[idx_slice], 0, axis_idx)
    return P_out, grids_out


def prenormalize(sliced_P, grids_in_order):
    """Pre-normalize the sliced density to O(1) before handing it to an
    interpolated distribution. Without this, a raw stored density that
    reaches very large or very small absolute values can push the
    distribution's own internal float32 renormalization to compute a huge
    or tiny integral, underflowing the eps floor to exact zero and
    poisoning the log-density grid with -inf/NaN."""
    Z = sliced_P
    for g in reversed(grids_in_order):
        Z = trapezoid(Z, g, axis=-1)
    return sliced_P / Z


# ---------------------------------------------------------------------------
# Distribution classes; data-informed ugrid, clipped rescale
# ---------------------------------------------------------------------------

UGRID_SIZE = 2048

AXIS_ATTRS = ("xx", "yy", "zz", "ww")


class InterpolatedDistMixin:
    """ln_prob / sample / equality hooks shared by the 2D, 3D and 4D classes."""

    def _ln_prob(self, samp, lnprob, outbounds):
        lnprob[:] = self._lnpdf(samp)
        lnprob[outbounds] = -np.inf
        return lnprob

    def _sample(self, size, **kwargs):
        u = random.rng.uniform(0.0, 1.0, (size, len(self.names)))
        return self._rescale(u)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if self.names != other.names or self.bounds != other.bounds:
            return False
        for attr in AXIS_ATTRS[:len(self.names)] + ("prob",):
            if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                return False
        return True


def _data_informed_ugrid(axis_nodes, prob_marginal, n_target=UGRID_SIZE):
    """
    Build a ugrid (cumulative-probability values in [0, 1], deliberately NOT
    evenly spaced) whose resolution in PHYSICAL axis space tracks the axis's
    own node spacing, rather than probability mass. A plain
    `np.linspace(0, 1, N)` ugrid is uniform in cumulative probability
    instead, which starves low-density-but-physically-important regions of
    table points (see module docstring).

    Subdivides each interval between adjacent axis nodes into equal physical
    steps, then evaluates the fully-marginalized 1D density's CDF at this
    finer physical grid -- so wherever the axis's own construction already
    concentrated nodes, the resulting ugrid automatically inherits that
    resolution.
    """
    axis_nodes = np.asarray(axis_nodes, dtype=np.float64)
    prob_marginal = np.asarray(prob_marginal, dtype=np.float64)
    n_intervals = axis_nodes.size - 1
    n_sub = max(1, round(n_target / n_intervals))

    pieces = [
        np.linspace(axis_nodes[i], axis_nodes[i + 1], n_sub, endpoint=False)
        for i in range(n_intervals)
    ]
    pieces.append(axis_nodes[-1:])
    fine_axis = np.concatenate(pieces)

    cdf_nodes = cumulative_trapezoid(prob_marginal, axis_nodes, initial=0.0)
    cdf_nodes /= cdf_nodes[-1]

    fine_cdf = np.interp(fine_axis, axis_nodes, cdf_nodes)
    fine_cdf = np.maximum.accumulate(fine_cdf)
    # Strict-monotonicity nudge (RegularGridInterpolator requires strictly
    # ascending axes); kept in float64 since 1e-12 is below float32's
    # precision near 1.0 and would otherwise silently collapse back together.
    fine_cdf += np.arange(fine_cdf.size, dtype=np.float64) * 1e-12
    fine_cdf /= fine_cdf[-1]
    fine_cdf[0] = 0.0
    fine_cdf[-1] = 1.0
    return fine_cdf


class PatchedTwoDInterpolatedDist(InterpolatedDistMixin, BaseJointPriorDist):
    def __init__(self, names, xx, yy, prob, bounds=None):
        xx = np.asarray(xx, float)
        yy = np.asarray(yy, float)
        if xx.ndim != 1 or yy.ndim != 1:
            raise ValueError("xx and yy must be 1D arrays")
        if not (np.all(np.diff(xx) > 0) and np.all(np.diff(yy) > 0)):
            raise ValueError("xx and yy must be strictly increasing")

        prob = np.asarray(prob, float)
        if prob.shape != (xx.size, yy.size):
            raise ValueError(
                f"prob must have shape (len(xx), len(yy)); got {prob.shape} vs {(xx.size, yy.size)}"
            )

        BaseJointPriorDist.__init__(self, names=names, bounds=bounds)

        eps = np.finfo(np.float32).tiny
        prob = np.clip(prob, eps, None).astype(np.float32)
        Z = trapezoid(trapezoid(prob, yy, axis=-1), xx)
        prob /= Z.astype(np.float32)

        self.xx = xx
        self.yy = yy
        self.prob = prob

        self._lnpdf = RegularGridInterpolator(
            (self.xx, self.yy), np.log(self.prob, dtype=np.float32),
            method="linear", bounds_error=False, fill_value=-np.inf,
        )

        px = trapezoid(self.prob, self.yy, axis=-1)
        Fx = cumulative_trapezoid(px, self.xx, initial=0.0)
        Fx /= Fx[-1]
        self._invFx = interp1d(Fx, self.xx, kind="linear", assume_sorted=True)

        cdf_y = cumulative_trapezoid(self.prob, self.yy, axis=-1, initial=0.0)
        norm = cdf_y[:, -1:]
        norm[norm == 0] = 1.0
        cdf_y = cdf_y / norm

        py_full = trapezoid(self.prob, self.xx, axis=0)
        ugrid = _data_informed_ugrid(self.yy, py_full, UGRID_SIZE)

        y_inv = np.empty((self.xx.size, ugrid.size), dtype=np.float32)
        for i in range(self.xx.size):
            y_inv[i, :] = np.interp(ugrid, cdf_y[i, :], self.yy)

        self._yinverse = RegularGridInterpolator(
            (self.xx, ugrid), y_inv, method="linear", bounds_error=True,
        )

    def _rescale(self, samp, **kwargs):
        """Clips each stage's output to its own axis's exact bounds before
        it feeds the next interpolator -- RegularGridInterpolator's linear
        interpolation can overshoot an axis's exact min/max by a few ULPs
        near a hard boundary, which otherwise raises a spurious ValueError
        the next time that value is used (every *_inverse table here uses
        bounds_error=True)."""
        u1, u2 = samp[:, 0], samp[:, 1]
        u1 = np.clip(u1, 0.0, 1.0)
        u2 = np.clip(u2, 0.0, 1.0)
        x = self._invFx(u1)
        x = np.clip(x, self.xx[0], self.xx[-1])
        y = self._yinverse(np.column_stack([x, u2]))
        y = np.clip(y, self.yy[0], self.yy[-1])
        return np.column_stack([x, y])


class PatchedThreeDInterpolatedDist(InterpolatedDistMixin, BaseJointPriorDist):
    def __init__(self, names, xx, yy, zz, prob, bounds=None):
        xx = np.asarray(xx, float)
        yy = np.asarray(yy, float)
        zz = np.asarray(zz, float)

        if not (xx.ndim == yy.ndim == zz.ndim == 1):
            raise ValueError("xx, yy, zz must be 1D")
        if not (np.all(np.diff(xx) > 0) and np.all(np.diff(yy) > 0) and np.all(np.diff(zz) > 0)):
            raise ValueError("grids must be strictly increasing")

        prob = np.asarray(prob, float)
        if prob.shape != (xx.size, yy.size, zz.size):
            raise ValueError(f"prob must have shape {(xx.size, yy.size, zz.size)}")

        BaseJointPriorDist.__init__(self, names=names, bounds=bounds)

        eps = np.finfo(np.float32).tiny
        prob = np.clip(prob, eps, None).astype(np.float32)

        Z = trapezoid(trapezoid(prob, zz, axis=-1), yy, axis=-1)
        Z = trapezoid(Z, xx, axis=-1)
        prob /= Z.astype(np.float32)

        self.xx, self.yy, self.zz = xx, yy, zz
        self.prob = prob

        self._lnpdf = RegularGridInterpolator(
            (xx, yy, zz), np.log(prob, dtype=np.float32),
            bounds_error=False, fill_value=-np.inf,
        )

        px = trapezoid(trapezoid(prob, zz, axis=-1), yy, axis=-1)
        Fx = cumulative_trapezoid(px, xx, initial=0.0)
        Fx /= Fx[-1]
        self._invFx = interp1d(Fx, xx, assume_sorted=True)

        py = trapezoid(prob, zz, axis=-1)
        cdf_y = cumulative_trapezoid(py, yy, axis=-1, initial=0.0)
        norm = cdf_y[:, -1:]
        norm[norm == 0] = 1.0
        cdf_y /= norm

        py_full = trapezoid(py, xx, axis=0)
        ugrid_y = _data_informed_ugrid(yy, py_full, UGRID_SIZE)

        y_inv = np.empty((xx.size, ugrid_y.size), dtype=np.float32)
        for i in range(xx.size):
            y_inv[i] = np.interp(ugrid_y, cdf_y[i], yy)

        self._yinverse = RegularGridInterpolator((xx, ugrid_y), y_inv, bounds_error=True)

        cdf_z = cumulative_trapezoid(prob, zz, axis=-1, initial=0.0)
        norm = cdf_z[:, :, -1:]
        norm[norm == 0] = 1.0
        cdf_z /= norm

        pz_full = trapezoid(trapezoid(prob, yy, axis=1), xx, axis=0)
        ugrid_z = _data_informed_ugrid(zz, pz_full, UGRID_SIZE)

        z_inv = np.empty((xx.size, yy.size, ugrid_z.size), dtype=np.float32)
        for i in range(xx.size):
            for j in range(yy.size):
                z_inv[i, j] = np.interp(ugrid_z, cdf_z[i, j], zz)

        self._zinverse = RegularGridInterpolator((xx, yy, ugrid_z), z_inv, bounds_error=True)

    def _rescale(self, samp, **kwargs):
        """See PatchedTwoDInterpolatedDist._rescale's docstring -- same fix,
        one more chained stage."""
        u1, u2, u3 = samp[:, 0], samp[:, 1], samp[:, 2]
        u1 = np.clip(u1, 0.0, 1.0)
        u2 = np.clip(u2, 0.0, 1.0)
        u3 = np.clip(u3, 0.0, 1.0)
        x = self._invFx(u1)
        x = np.clip(x, self.xx[0], self.xx[-1])
        y = self._yinverse(np.column_stack([x, u2]))
        y = np.clip(y, self.yy[0], self.yy[-1])
        z = self._zinverse(np.column_stack([x, y, u3]))
        z = np.clip(z, self.zz[0], self.zz[-1])
        return np.column_stack([x, y, z])


class PatchedFourDInterpolatedDist(InterpolatedDistMixin, BaseJointPriorDist):
    def __init__(self, names, xx, yy, zz, ww, prob, bounds=None):
        xx = np.asarray(xx, float)
        yy = np.asarray(yy, float)
        zz = np.asarray(zz, float)
        ww = np.asarray(ww, float)

        if not (xx.ndim == yy.ndim == zz.ndim == ww.ndim == 1):
            raise ValueError("xx, yy, zz, ww must be 1D")
        for g in (xx, yy, zz, ww):
            if not np.all(np.diff(g) > 0):
                raise ValueError("grids must be strictly increasing")

        prob = np.asarray(prob, float)
        if prob.shape != (xx.size, yy.size, zz.size, ww.size):
            raise ValueError(
                f"prob must have shape {(xx.size, yy.size, zz.size, ww.size)}; got {prob.shape}"
            )

        BaseJointPriorDist.__init__(self, names=names, bounds=bounds)

        eps = np.finfo(np.float32).tiny
        prob = np.clip(prob, eps, None).astype(np.float32)

        Z = trapezoid(
            trapezoid(trapezoid(trapezoid(prob, ww, axis=-1), zz, axis=-1), yy, axis=-1),
            xx, axis=-1,
        )
        prob /= Z.astype(np.float32)

        self.xx, self.yy, self.zz, self.ww = xx, yy, zz, ww
        self.prob = prob

        self._lnpdf = RegularGridInterpolator(
            (xx, yy, zz, ww), np.log(prob, dtype=np.float32),
            bounds_error=False, fill_value=-np.inf,
        )

        px = trapezoid(trapezoid(trapezoid(prob, ww, axis=-1), zz, axis=-1), yy, axis=-1)
        Fx = cumulative_trapezoid(px, xx, initial=0.0)
        Fx /= Fx[-1]
        self._invFx = interp1d(Fx, xx, assume_sorted=True)

        py = trapezoid(trapezoid(prob, ww, axis=-1), zz, axis=-1)
        cdf_y = cumulative_trapezoid(py, yy, axis=-1, initial=0.0)
        norm = cdf_y[:, -1:]
        norm[norm == 0] = 1.0
        cdf_y /= norm

        py_full = trapezoid(py, xx, axis=0)
        ugrid_y = _data_informed_ugrid(yy, py_full, UGRID_SIZE)

        y_inv = np.empty((xx.size, ugrid_y.size), dtype=np.float32)
        for i in range(xx.size):
            y_inv[i] = np.interp(ugrid_y, cdf_y[i], yy)

        self._yinverse = RegularGridInterpolator((xx, ugrid_y), y_inv, bounds_error=True)

        pz = trapezoid(prob, ww, axis=-1)
        cdf_z = cumulative_trapezoid(pz, zz, axis=-1, initial=0.0)
        norm = cdf_z[:, :, -1:]
        norm[norm == 0] = 1.0
        cdf_z /= norm

        pz_full = trapezoid(trapezoid(pz, yy, axis=1), xx, axis=0)
        ugrid_z = _data_informed_ugrid(zz, pz_full, UGRID_SIZE)

        z_inv = np.empty((xx.size, yy.size, ugrid_z.size), dtype=np.float32)
        for i in range(xx.size):
            for j in range(yy.size):
                z_inv[i, j] = np.interp(ugrid_z, cdf_z[i, j], zz)

        self._zinverse = RegularGridInterpolator((xx, yy, ugrid_z), z_inv, bounds_error=True)

        # delta_lambda_tilde (w) axis: left on a plain uniform-in-probability
        # ugrid. Its discontinuities are tied to a hard wedge-shaped support
        # (bounded by the lambda_1/lambda_2 >= 0 physicality constraint)
        # rather than a skewed tail, and need a different, symmetric
        # two-tailed treatment not implemented here.
        ugrid_w = np.linspace(0.0, 1.0, UGRID_SIZE, dtype=np.float32)
        cdf_w = cumulative_trapezoid(prob, ww, axis=-1, initial=0.0)
        norm = cdf_w[:, :, :, -1:]
        norm[norm == 0] = 1.0
        cdf_w /= norm

        w_inv = np.empty((xx.size, yy.size, zz.size, ugrid_w.size), dtype=np.float32)
        for i in range(xx.size):
            for j in range(yy.size):
                for k in range(zz.size):
                    w_inv[i, j, k] = np.interp(ugrid_w, cdf_w[i, j, k], ww)

        self._winverse = RegularGridInterpolator((xx, yy, zz, ugrid_w), w_inv, bounds_error=True)

    def _rescale(self, samp, **kwargs):
        """See PatchedTwoDInterpolatedDist._rescale's docstring -- same fix,
        one more chained stage. This is the class actually used for the 4D
        (chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde) Goldilocks
        prior."""
        u1, u2, u3, u4 = samp[:, 0], samp[:, 1], samp[:, 2], samp[:, 3]
        u1 = np.clip(u1, 0.0, 1.0)
        u2 = np.clip(u2, 0.0, 1.0)
        u3 = np.clip(u3, 0.0, 1.0)
        u4 = np.clip(u4, 0.0, 1.0)
        x = self._invFx(u1)
        x = np.clip(x, self.xx[0], self.xx[-1])
        y = self._yinverse(np.column_stack([x, u2]))
        y = np.clip(y, self.yy[0], self.yy[-1])
        z = self._zinverse(np.column_stack([x, y, u3]))
        z = np.clip(z, self.zz[0], self.zz[-1])
        w = self._winverse(np.column_stack([x, y, z, u4]))
        w = np.clip(w, self.ww[0], self.ww[-1])
        return np.column_stack([x, y, z, w])


# ---------------------------------------------------------------------------
# Public entry points: joint2D / joint3D / joint4D.
# ---------------------------------------------------------------------------

_AXIS_KWARGS = ("xx", "yy", "zz", "ww")
_DIST_CLASSES = {
    2: PatchedTwoDInterpolatedDist,
    3: PatchedThreeDInterpolatedDist,
    4: PatchedFourDInterpolatedDist,
}


def _joint_nd(active_keys, fixed_params, data_file, ndim, fn_name, bounds=None):
    # active_keys only decides *which* parameters are active; the file's own
    # physical order decides axis ordering -- user input order is ignored.
    sliced_P, all_grids, ordered_keys = get_sorted_grid_and_prob(active_keys, fixed_params, data_file)
    check_ndim(ordered_keys, ndim, fn_name)

    grids_in_order = [all_grids[k] for k in ordered_keys]
    sliced_P, grids_in_order = slice_grid_to_bounds(sliced_P, grids_in_order, ordered_keys, bounds)
    sliced_P = prenormalize(sliced_P, grids_in_order)

    dist_names = list(ordered_keys)
    dist = _DIST_CLASSES[ndim](
        names=dist_names,
        prob=sliced_P,
        bounds=[(g.min(), g.max()) for g in grids_in_order],
        **dict(zip(_AXIS_KWARGS, grids_in_order)),
    )
    return [JointPrior(dist, name=name) for name in dist_names]


def joint2D(active_keys, fixed_params, data_file, bounds=None):
    """Build a 2D joint prior from `data_file`, over whichever 2 of
    (chirp_mass_source, mass_ratio, lambda_tilde, delta_lambda_tilde) remain
    active after applying `fixed_params`.

    active_keys  : list of parameter names you intend to keep active (only
                   used to sanity-check the result has exactly 2 dimensions
                   after fixing -- the file's own axes decide the rest).
    fixed_params : dict {name: value}; slices the grid at its nearest node
                   along that axis and removes it from the active set.
    data_file    : path to a .npz (or text .dat) grid file.
    bounds       : optional dict {name: (lo, hi)} to genuinely restrict an
                   active axis to a sub-range before building the prior.

    Returns a list of bilby.core.prior.JointPrior objects (one per active
    dimension, sharing the same underlying joint distribution) -- assign
    each into a PriorDict via `priors[jp.name] = jp`.
    """
    return _joint_nd(active_keys, fixed_params, data_file, 2, "joint2D", bounds=bounds)


def joint3D(active_keys, fixed_params, data_file, bounds=None):
    """Same as joint2D, but for the 3D case (3 parameters remain active)."""
    return _joint_nd(active_keys, fixed_params, data_file, 3, "joint3D", bounds=bounds)


def joint4D(active_keys, fixed_params, data_file, bounds=None):
    """Same as joint2D, but for the full 4D case: all of (chirp_mass_source,
    mass_ratio, lambda_tilde, delta_lambda_tilde) active, nothing fixed --
    this is the standard way to use the Goldilocks prior."""
    return _joint_nd(active_keys, fixed_params, data_file, 4, "joint4D", bounds=bounds)
