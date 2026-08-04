"""catch22 time-series features, implemented in Rust.

The 22 catch22 features, plus the two catch24 additions (mean and standard
deviation) and the slope of a linear fit, for 25 in total. Feature order is
given by :data:`FEATURE_NAMES` and is stable.

    >>> import numpy as np, pycatch22_rs
    >>> x = np.random.default_rng(0).standard_normal(500)
    >>> values = pycatch22_rs.compute_all(x)          # all 25, z-scored pipeline
    >>> centroid = pycatch22_rs.SP_Summaries_welch_rect_centroid(x)
    >>> batch = pycatch22_rs.compute_batch(np.random.default_rng(0).standard_normal((64, 500)))

By default the 22 catch22 features are computed on the z-scored series and the
remaining three on the raw one, which is what the reference C implementation
does. Pass ``normalize=False`` to compute on the series exactly as given.
"""

from __future__ import annotations

import importlib.metadata as _metadata

import numpy as np

from . import pycatch22_rs as _rust

FEATURE_NAMES: tuple[str, ...] = _rust.FEATURE_NAMES
N_FEATURES: int = _rust.N_FEATURES
#: Number of features computed on the z-scored series; the rest use the raw one.
N_NORMALIZED: int = _rust.N_NORMALIZED

try:
    __version__ = _metadata.version("pycatch22-rs")
except _metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "unknown"


def _as_series(x) -> np.ndarray:
    """Coerces input to a contiguous 1-D float64 array, copying only if needed."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {array.shape}")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _as_batch(x) -> np.ndarray:
    """Coerces input to a contiguous 2-D float64 array, copying only if needed."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array of series, got shape {array.shape}")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def compute_all(x, *, normalize: bool = True) -> np.ndarray:
    """Computes all 25 features for one series, returning a ``(25,)`` array."""
    return _rust.compute_all(_as_series(x), normalize=normalize)


def compute_batch(x, *, normalize: bool = True) -> np.ndarray:
    """Computes all 25 features for each row of a 2-D array.

    Returns an ``(n_series, 25)`` array. The GIL is released and rows are
    processed in parallel, so this is substantially faster than looping over
    :func:`compute_all` in Python.
    """
    return _rust.compute_batch(_as_batch(x), normalize=normalize)


def compute(x, n: int, *, normalize: bool = False) -> float:
    """Computes a single feature by index, as ordered by :data:`FEATURE_NAMES`."""
    return _rust.compute(_as_series(x), n, normalize=normalize)


def zscore(x) -> np.ndarray:
    """Z-scores a series using the sample standard deviation (``ddof=1``).

    The ``ddof=1`` matches the reference implementation. It matters: several
    features are not scale-invariant, so normalising with the population
    standard deviation shifts their values.
    """
    return _rust.zscore(_as_series(x))


def catch22_all(data, catch24: bool = False) -> dict[str, list]:
    """Compatibility helper shaped like ``pycatch22.catch22_all``.

    Returns ``{"names": [...], "values": [...]}``. With ``catch24=True`` the
    mean and standard deviation are included. The slope is not part of catch24
    and is never returned here; use :func:`compute_all` for all 25.
    """
    values = compute_all(data, normalize=True)
    count = 24 if catch24 else 22
    return {
        "names": list(FEATURE_NAMES[:count]),
        "values": [float(v) for v in values[:count]],
    }


# The 25 features, individually callable under their canonical catch22 names.
DN_OutlierInclude_n_001_mdrmd = _rust.DN_OutlierInclude_n_001_mdrmd
DN_OutlierInclude_p_001_mdrmd = _rust.DN_OutlierInclude_p_001_mdrmd
DN_HistogramMode_5 = _rust.DN_HistogramMode_5
DN_HistogramMode_10 = _rust.DN_HistogramMode_10
CO_Embed2_Dist_tau_d_expfit_meandiff = _rust.CO_Embed2_Dist_tau_d_expfit_meandiff
CO_f1ecac = _rust.CO_f1ecac
CO_FirstMin_ac = _rust.CO_FirstMin_ac
CO_HistogramAMI_even_2_5 = _rust.CO_HistogramAMI_even_2_5
CO_trev_1_num = _rust.CO_trev_1_num
FC_LocalSimple_mean1_tauresrat = _rust.FC_LocalSimple_mean1_tauresrat
FC_LocalSimple_mean3_stderr = _rust.FC_LocalSimple_mean3_stderr
IN_AutoMutualInfoStats_40_gaussian_fmmi = _rust.IN_AutoMutualInfoStats_40_gaussian_fmmi
MD_hrv_classic_pnn40 = _rust.MD_hrv_classic_pnn40
SB_BinaryStats_diff_longstretch0 = _rust.SB_BinaryStats_diff_longstretch0
SB_BinaryStats_mean_longstretch1 = _rust.SB_BinaryStats_mean_longstretch1
SB_MotifThree_quantile_hh = _rust.SB_MotifThree_quantile_hh
SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 = _rust.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 = _rust.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
SP_Summaries_welch_rect_area_5_1 = _rust.SP_Summaries_welch_rect_area_5_1
SP_Summaries_welch_rect_centroid = _rust.SP_Summaries_welch_rect_centroid
SB_TransitionMatrix_3ac_sumdiagcov = _rust.SB_TransitionMatrix_3ac_sumdiagcov
PD_PeriodicityWang_th0_01 = _rust.PD_PeriodicityWang_th0_01
DN_Mean = _rust.DN_Mean
DN_Spread_Std = _rust.DN_Spread_Std
SlopeOfLinearFit = _rust.SlopeOfLinearFit

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "N_NORMALIZED",
    "compute",
    "compute_all",
    "compute_batch",
    "zscore",
    "catch22_all",
    *FEATURE_NAMES,
]
