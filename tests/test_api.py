"""Surface behaviour of the Python API: shapes, dtypes, layouts, errors."""

from __future__ import annotations

import numpy as np
import pytest

import pycatch22_rs


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(20260804)
    return rng.standard_normal(400)


def test_feature_names_are_complete_and_unique():
    assert len(pycatch22_rs.FEATURE_NAMES) == pycatch22_rs.N_FEATURES == 25
    assert len(set(pycatch22_rs.FEATURE_NAMES)) == 25
    assert pycatch22_rs.FEATURE_NAMES[0] == "DN_OutlierInclude_n_001_mdrmd"
    assert pycatch22_rs.FEATURE_NAMES[22] == "DN_Mean"
    assert pycatch22_rs.FEATURE_NAMES[24] == "SlopeOfLinearFit"


def test_every_feature_name_is_callable(series):
    for name in pycatch22_rs.FEATURE_NAMES:
        function = getattr(pycatch22_rs, name)
        value = function(series)
        assert isinstance(value, float)


def test_compute_all_shape_and_dtype(series):
    values = pycatch22_rs.compute_all(series)
    assert values.shape == (25,)
    assert values.dtype == np.float64


def test_named_functions_match_compute_all(series):
    values = pycatch22_rs.compute_all(series, normalize=True)
    z = pycatch22_rs.zscore(series)

    for index, name in enumerate(pycatch22_rs.FEATURE_NAMES):
        function = getattr(pycatch22_rs, name)
        # Features 0..21 are defined on the z-scored series, the rest on raw.
        source = z if index < pycatch22_rs.N_NORMALIZED else series
        direct = function(source)
        if np.isnan(values[index]):
            assert np.isnan(direct), name
        else:
            assert direct == values[index], name


def test_compute_by_index_matches_named(series):
    for index, name in enumerate(pycatch22_rs.FEATURE_NAMES):
        by_index = pycatch22_rs.compute(series, index)
        by_name = getattr(pycatch22_rs, name)(series)
        if np.isnan(by_index):
            assert np.isnan(by_name), name
        else:
            assert by_index == by_name, name


def test_batch_matches_looping(series):
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((32, 200))

    parallel = pycatch22_rs.compute_batch(batch)
    looped = np.array([pycatch22_rs.compute_all(row) for row in batch])

    assert parallel.shape == (32, 25)
    np.testing.assert_array_equal(parallel, looped)


def test_batch_respects_normalize_flag():
    rng = np.random.default_rng(2)
    batch = rng.standard_normal((8, 128)) * 5.0 + 3.0

    normalized = pycatch22_rs.compute_batch(batch, normalize=True)
    raw = pycatch22_rs.compute_batch(batch, normalize=False)

    # DN_Mean/DN_Spread_Std come from the raw series either way.
    np.testing.assert_array_equal(normalized[:, 22], raw[:, 22])
    np.testing.assert_array_equal(normalized[:, 23], raw[:, 23])
    # A scale-dependent feature must differ between the two.
    assert not np.allclose(normalized[:, 2], raw[:, 2])


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int32])
def test_accepts_other_dtypes(series, dtype):
    converted = series.astype(dtype)
    values = pycatch22_rs.compute_all(converted)
    assert values.shape == (25,)
    assert np.isfinite(values[22])


def test_accepts_python_list():
    values = pycatch22_rs.compute_all([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0])
    assert values.shape == (25,)


def test_non_contiguous_input_matches_contiguous(series):
    strided = series[::2]
    assert not strided.flags.c_contiguous

    np.testing.assert_array_equal(
        pycatch22_rs.compute_all(strided),
        pycatch22_rs.compute_all(np.ascontiguousarray(strided)),
    )


def test_fortran_ordered_batch_matches_c_ordered():
    rng = np.random.default_rng(3)
    batch = rng.standard_normal((16, 100))
    fortran = np.asfortranarray(batch)
    assert fortran.flags.f_contiguous

    np.testing.assert_array_equal(
        pycatch22_rs.compute_batch(fortran),
        pycatch22_rs.compute_batch(batch),
    )


def test_input_is_not_modified(series):
    original = series.copy()
    pycatch22_rs.compute_all(series)
    pycatch22_rs.zscore(series)
    np.testing.assert_array_equal(series, original)


def test_zscore_uses_sample_std(series):
    z = pycatch22_rs.zscore(series)
    assert isinstance(z, np.ndarray)
    assert z.dtype == np.float64
    assert abs(z.mean()) < 1e-12
    # ddof=1, matching the reference implementation
    assert abs(z.std(ddof=1) - 1.0) < 1e-12


def test_rejects_non_finite_input():
    with pytest.raises(ValueError, match="not finite"):
        pycatch22_rs.compute_all(np.array([1.0, np.nan, 3.0, 4.0]))
    with pytest.raises(ValueError, match="not finite"):
        pycatch22_rs.compute_all(np.array([1.0, np.inf, 3.0, 4.0]))


def test_rejects_too_short_input():
    with pytest.raises(ValueError, match="smaller than minimum"):
        pycatch22_rs.compute_all(np.array([1.0, 2.0, 3.0]))


def test_rejects_constant_series_when_normalizing():
    # Z-scoring divides by a zero standard deviation.
    with pytest.raises(ValueError):
        pycatch22_rs.compute_all(np.full(32, 2.5), normalize=True)


def test_rejects_out_of_range_feature_index(series):
    with pytest.raises(ValueError, match="out of range"):
        pycatch22_rs.compute(series, 25)


def test_rejects_wrong_dimensionality(series):
    with pytest.raises(ValueError, match="1-D"):
        pycatch22_rs.compute_all(series.reshape(2, -1))
    with pytest.raises(ValueError, match="2-D"):
        pycatch22_rs.compute_batch(series)


def test_catch22_all_compat_shape(series):
    result = pycatch22_rs.catch22_all(series)
    assert list(result) == ["names", "values"]
    assert len(result["names"]) == len(result["values"]) == 22

    with_catch24 = pycatch22_rs.catch22_all(series, catch24=True)
    assert len(with_catch24["names"]) == 24
    assert with_catch24["names"][22:] == ["DN_Mean", "DN_Spread_Std"]


def test_batch_of_one_matches_single(series):
    batch = pycatch22_rs.compute_batch(series.reshape(1, -1))
    single = pycatch22_rs.compute_all(series)
    np.testing.assert_array_equal(batch[0], single)
