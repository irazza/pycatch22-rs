//! Python bindings for the catch22 feature set.
//!
//! Arrays cross the boundary as borrowed numpy views rather than being copied
//! element by element into a `Vec`, and the batch entry point releases the GIL
//! so a 2-D input can be processed in parallel.

use numpy::ndarray::Axis;
use numpy::{Ix1, Ix2, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

/// Number of features: the 22 catch22 features, the 2 catch24 additions
/// (mean and standard deviation), and the slope of a linear fit.
pub const N_FEATURES: usize = catch22::N_CATCH22;

fn to_py_err(err: catch22::Catch22Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Borrows a 1-D array as a contiguous slice, copying only if the caller passed
/// a strided view (e.g. `x[::2]`) that has no contiguous representation.
fn as_slice<'a>(array: &'a PyReadonlyArray<'_, f64, Ix1>, scratch: &'a mut Vec<f64>) -> &'a [f64] {
    match array.as_slice() {
        Ok(slice) => slice,
        Err(_) => {
            scratch.clear();
            scratch.extend(array.as_array().iter().copied());
            scratch.as_slice()
        }
    }
}

/// Computes every feature on `x` as given, without normalising.
///
/// Most callers want [`compute_all`] with `normalize=True`, which reproduces
/// the reference pipeline.
#[pyfunction]
#[pyo3(signature = (x, *, normalize = true))]
fn compute_all<'py>(
    py: Python<'py>,
    x: PyReadonlyArray<'py, f64, Ix1>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut scratch = Vec::new();
    let series = as_slice(&x, &mut scratch);

    let values = if normalize {
        catch22::compute_all_normalized(series).map_err(to_py_err)?
    } else {
        catch22::compute_all(series).map_err(to_py_err)?
    };

    Ok(values.to_pyarray(py))
}

/// Computes every feature for each row of a 2-D array.
///
/// Releases the GIL and processes rows in parallel; the result has shape
/// `(n_series, 25)`.
#[pyfunction]
#[pyo3(signature = (x, *, normalize = true))]
fn compute_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray<'py, f64, Ix2>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let input = x.as_array();
    let n_series = input.len_of(Axis(0));

    // Collect rows as owned slices up front: `ArrayView` is not `Sync` in a way
    // rayon can use for arbitrary strides, and this also normalises Fortran
    // ordering into something contiguous exactly once.
    let rows: Vec<Vec<f64>> = input
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().collect())
        .collect();

    let results: Result<Vec<[f64; N_FEATURES]>, catch22::Catch22Error> = py.detach(move || {
        rows.par_iter()
            .map(|row| {
                if normalize {
                    catch22::compute_all_normalized(row)
                } else {
                    catch22::compute_all(row)
                }
            })
            .collect()
    });

    let results = results.map_err(to_py_err)?;

    let mut flat = Vec::with_capacity(n_series * N_FEATURES);
    for row in &results {
        flat.extend_from_slice(row);
    }

    let array = PyArray1::from_vec(py, flat);
    array.reshape([n_series, N_FEATURES])
}

/// Computes a single feature by index (0..25), as ordered by `FEATURE_NAMES`.
#[pyfunction]
#[pyo3(signature = (x, n, *, normalize = false))]
fn compute<'py>(x: PyReadonlyArray<'py, f64, Ix1>, n: usize, normalize: bool) -> PyResult<f64> {
    let mut scratch = Vec::new();
    let series = as_slice(&x, &mut scratch);

    if normalize {
        if n >= catch22::N_NORMALIZED {
            // Features 22..24 are defined on the raw series; normalising first
            // would make the mean 0 and the standard deviation 1 by
            // construction.
            return catch22::compute(series, n).map_err(to_py_err);
        }
        let z = catch22::zscore(series);
        return catch22::compute(&z, n).map_err(to_py_err);
    }

    catch22::compute(series, n).map_err(to_py_err)
}

/// Z-scores a series using the sample standard deviation (`ddof=1`), matching
/// the reference implementation's `zscore_norm2`.
#[pyfunction]
#[pyo3(signature = (x))]
fn zscore<'py>(
    py: Python<'py>,
    x: PyReadonlyArray<'py, f64, Ix1>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut scratch = Vec::new();
    let series = as_slice(&x, &mut scratch);
    if series.len() < 2 {
        return Err(PyValueError::new_err(
            "zscore needs at least 2 samples".to_string(),
        ));
    }
    Ok(PyArray1::from_vec(py, catch22::zscore(series)))
}

/// Defines a `#[pyfunction]` per feature, exported under its canonical catch22
/// name, so `pycatch22_rs.DN_HistogramMode_5(x)` works alongside `compute_all`.
macro_rules! feature_functions {
    ($(($rust_name:ident, $py_name:literal, $index:expr)),* $(,)?) => {
        $(
            #[pyfunction]
            #[pyo3(name = $py_name, signature = (x, *, normalize = false))]
            fn $rust_name(x: PyReadonlyArray<'_, f64, Ix1>, normalize: bool) -> PyResult<f64> {
                let mut scratch = Vec::new();
                let series = as_slice(&x, &mut scratch);
                compute_one($index, series, normalize)
            }
        )*

        fn register_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( m.add_function(wrap_pyfunction!($rust_name, m)?)?; )*
            Ok(())
        }
    };
}

fn compute_one(index: usize, series: &[f64], normalize: bool) -> PyResult<f64> {
    if normalize && index < catch22::N_NORMALIZED {
        let z = catch22::zscore(series);
        return catch22::compute(&z, index).map_err(to_py_err);
    }
    catch22::compute(series, index).map_err(to_py_err)
}

feature_functions![
    (feat00, "DN_OutlierInclude_n_001_mdrmd", 0),
    (feat01, "DN_OutlierInclude_p_001_mdrmd", 1),
    (feat02, "DN_HistogramMode_5", 2),
    (feat03, "DN_HistogramMode_10", 3),
    (feat04, "CO_Embed2_Dist_tau_d_expfit_meandiff", 4),
    (feat05, "CO_f1ecac", 5),
    (feat06, "CO_FirstMin_ac", 6),
    (feat07, "CO_HistogramAMI_even_2_5", 7),
    (feat08, "CO_trev_1_num", 8),
    (feat09, "FC_LocalSimple_mean1_tauresrat", 9),
    (feat10, "FC_LocalSimple_mean3_stderr", 10),
    (feat11, "IN_AutoMutualInfoStats_40_gaussian_fmmi", 11),
    (feat12, "MD_hrv_classic_pnn40", 12),
    (feat13, "SB_BinaryStats_diff_longstretch0", 13),
    (feat14, "SB_BinaryStats_mean_longstretch1", 14),
    (feat15, "SB_MotifThree_quantile_hh", 15),
    (feat16, "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", 16),
    (feat17, "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", 17),
    (feat18, "SP_Summaries_welch_rect_area_5_1", 18),
    (feat19, "SP_Summaries_welch_rect_centroid", 19),
    (feat20, "SB_TransitionMatrix_3ac_sumdiagcov", 20),
    (feat21, "PD_PeriodicityWang_th0_01", 21),
    (feat22, "DN_Mean", 22),
    (feat23, "DN_Spread_Std", 23),
    (feat24, "SlopeOfLinearFit", 24),
];

#[pymodule]
#[pyo3(name = "pycatch22_rs")]
fn py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("N_FEATURES", N_FEATURES)?;
    m.add("N_NORMALIZED", catch22::N_NORMALIZED)?;
    m.add(
        "FEATURE_NAMES",
        PyTuple::new(m.py(), catch22::FEATURE_NAMES)?,
    )?;

    m.add_function(wrap_pyfunction!(compute_all, m)?)?;
    m.add_function(wrap_pyfunction!(compute_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute, m)?)?;
    m.add_function(wrap_pyfunction!(zscore, m)?)?;
    register_features(m)?;

    Ok(())
}
