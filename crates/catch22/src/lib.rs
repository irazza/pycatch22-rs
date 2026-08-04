// The feature implementations are a deliberate line-by-line transliteration of
// the original catch22 C sources, so that any of them can be diffed against its
// reference alongside `tools/c_reference/upstream/C`. Index-based loops,
// explicit `return`s and C-shaped arithmetic are kept on purpose: rewriting
// them in idiomatic Rust would reorder floating-point operations and make the
// correspondence with the reference much harder to audit.
#![allow(
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::manual_memcpy,
    clippy::assign_op_pattern,
    clippy::let_and_return,
    clippy::len_zero,
    clippy::manual_is_multiple_of,
    clippy::neg_multiply,
    clippy::unnecessary_cast,
    clippy::absurd_extreme_comparisons,
    clippy::type_complexity
)]

mod catch22;
mod statistics;

pub const N_CATCH22: usize = 25;
const MIN_INPUT_LEN: usize = 4;

// Performance-focused implementation notes:
// - Input validation is centralized in `compute_basic_stats_checked`, which also computes
//   mean/std/slope in a single pass to avoid repeated traversals of the input.
// - `compute_all` reuses a single autocorrelation (and derived tau) for multiple features,
//   and routes FC_LocalSimple_mean1_tauresrat through that shared autocorr to avoid extra FFTs.
// - PD_PeriodicityWang now uses an FFT-based autocovariance (O(n log n)) instead of per-lag
//   autocovariance (O(n^2)) in the hot path.
// - Unchecked variants exist for benchmarking or trusted call sites to skip validation costs.
#[derive(Debug, Clone, Copy)]
struct BasicStats {
    mean: f64,
    std_dev: f64,
    slope: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Catch22Error {
    InputTooShort { len: usize, min_len: usize },
    NonFiniteValue { index: usize, value: f64 },
    InvalidFeatureIndex { index: usize, max: usize },
}

impl std::fmt::Display for Catch22Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Catch22Error::InputTooShort { len, min_len } => {
                write!(f, "input length {len} is smaller than minimum {min_len}")
            }
            Catch22Error::NonFiniteValue { index, value } => {
                write!(f, "input value at index {index} is not finite: {value}")
            }
            Catch22Error::InvalidFeatureIndex { index, max } => {
                write!(f, "feature index {index} is out of range (max {max})")
            }
        }
    }
}

impl std::error::Error for Catch22Error {}

fn validate_input(x: &[f64]) -> Result<(), Catch22Error> {
    if x.len() < MIN_INPUT_LEN {
        return Err(Catch22Error::InputTooShort {
            len: x.len(),
            min_len: MIN_INPUT_LEN,
        });
    }

    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(Catch22Error::NonFiniteValue { index, value });
        }
    }

    Ok(())
}

/// Validates the input and returns mean, standard deviation and slope.
///
/// These delegate to the same functions that back features 22, 23 and 24 rather
/// than fusing their own one-pass versions. A fused Welford pass is better
/// conditioned, but it rounds differently from C's plain sum, which made
/// `compute_all(x)[22]` and `dn_mean(x)` disagree in the last bits.
fn compute_basic_stats_checked(x: &[f64]) -> Result<BasicStats, Catch22Error> {
    validate_input(x)?;
    Ok(compute_basic_stats_unchecked(x))
}

fn compute_basic_stats_unchecked(x: &[f64]) -> BasicStats {
    debug_assert!(x.len() >= MIN_INPUT_LEN);

    BasicStats {
        mean: statistics::mean(x),
        std_dev: statistics::std_dev(x),
        slope: statistics::slope(x),
    }
}

// ---------------------------------------------------------------------------
// The 25 features, individually callable.
//
// Names follow the canonical catch22 spelling (lower-cased), so
// `DN_HistogramMode_5` is `dn_histogram_mode_5` here. The index each one
// occupies in `compute`/`compute_all` is the position it has in `FEATURES`
// below, and that order is API: do not reshuffle it.
//
// These take the series exactly as given. Features 0..=21 are meant to be fed a
// z-scored series (see `compute_all_normalized`); 22..=24 are meant to be fed
// the raw one.
// ---------------------------------------------------------------------------

/// DN_OutlierInclude_n_001_mdrmd
pub fn dn_outlier_include_n_001_mdrmd(x: &[f64]) -> f64 {
    catch22::dn_outlier_include_np_001_mdrmd(x, false)
}

/// DN_OutlierInclude_p_001_mdrmd
pub fn dn_outlier_include_p_001_mdrmd(x: &[f64]) -> f64 {
    catch22::dn_outlier_include_np_001_mdrmd(x, true)
}

/// DN_HistogramMode_5
pub fn dn_histogram_mode_5(x: &[f64]) -> f64 {
    catch22::dn_histogram_mode_n(x, 5)
}

/// DN_HistogramMode_10
pub fn dn_histogram_mode_10(x: &[f64]) -> f64 {
    catch22::dn_histogram_mode_n(x, 10)
}

/// CO_Embed2_Dist_tau_d_expfit_meandiff
pub fn co_embed2_dist_tau_d_expfit_meandiff(x: &[f64]) -> f64 {
    catch22::co_embed2_dist_tau_d_expfit_meandiff(x)
}

/// CO_f1ecac
pub fn co_f1ecac(x: &[f64]) -> f64 {
    catch22::co_f1ecac(x)
}

/// CO_FirstMin_ac
pub fn co_first_min_ac(x: &[f64]) -> f64 {
    catch22::co_first_min_ac(x)
}

/// CO_HistogramAMI_even_2_5
pub fn co_histogram_ami_even_2_5(x: &[f64]) -> f64 {
    catch22::co_histogram_ami_even_tau_bins(x, 2, 5)
}

/// CO_trev_1_num
pub fn co_trev_1_num(x: &[f64]) -> f64 {
    catch22::co_trev_1_num(x)
}

/// FC_LocalSimple_mean1_tauresrat
pub fn fc_local_simple_mean1_tauresrat(x: &[f64]) -> f64 {
    catch22::fc_local_simple_mean_tauresrat(x, 1)
}

/// FC_LocalSimple_mean3_stderr
pub fn fc_local_simple_mean3_stderr(x: &[f64]) -> f64 {
    catch22::fc_local_simple_mean_stderr(x, 3)
}

/// IN_AutoMutualInfoStats_40_gaussian_fmmi
pub fn in_auto_mutual_info_stats_40_gaussian_fmmi(x: &[f64]) -> f64 {
    catch22::in_auto_mutual_info_stats_tau_gaussian_fmmi(x, 40.0)
}

/// MD_hrv_classic_pnn40
pub fn md_hrv_classic_pnn40(x: &[f64]) -> f64 {
    catch22::md_hrv_classic_pnn(x, 40)
}

/// SB_BinaryStats_diff_longstretch0
pub fn sb_binary_stats_diff_longstretch0(x: &[f64]) -> f64 {
    catch22::sb_binary_stats_diff_longstretch0(x)
}

/// SB_BinaryStats_mean_longstretch1
pub fn sb_binary_stats_mean_longstretch1(x: &[f64]) -> f64 {
    catch22::sb_binary_stats_mean_longstretch1(x)
}

/// SB_MotifThree_quantile_hh
pub fn sb_motif_three_quantile_hh(x: &[f64]) -> f64 {
    catch22::sb_motif_three_quantile_hh(x)
}

/// SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
pub fn sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(x: &[f64]) -> f64 {
    catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 1, "rsrangefit")
}

/// SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
pub fn sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(x: &[f64]) -> f64 {
    catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 2, "dfa")
}

/// SP_Summaries_welch_rect_area_5_1
pub fn sp_summaries_welch_rect_area_5_1(x: &[f64]) -> f64 {
    catch22::sp_summaries_welch_rect(x, "area_5_1")
}

/// SP_Summaries_welch_rect_centroid
pub fn sp_summaries_welch_rect_centroid(x: &[f64]) -> f64 {
    catch22::sp_summaries_welch_rect(x, "centroid")
}

/// SB_TransitionMatrix_3ac_sumdiagcov
pub fn sb_transition_matrix_3ac_sumdiagcov(x: &[f64]) -> f64 {
    catch22::sb_transition_matrix_3ac_sumdiagcov(x)
}

/// PD_PeriodicityWang_th0_01
pub fn pd_periodicity_wang_th0_01(x: &[f64]) -> f64 {
    catch22::pd_periodicity_wang_th0_01(x)
}

/// DN_Mean — the first of the two catch24 additions. Expects the raw series.
pub fn dn_mean(x: &[f64]) -> f64 {
    statistics::mean(x)
}

/// DN_Spread_Std — the second catch24 addition, the sample standard deviation.
/// Expects the raw series.
pub fn dn_spread_std(x: &[f64]) -> f64 {
    statistics::std_dev(x)
}

/// Least-squares slope against `x = 1..n`. Not part of catch22 or catch24; it
/// is this crate's 25th feature. Expects the raw series.
pub fn slope_of_linear_fit(x: &[f64]) -> f64 {
    statistics::slope(x)
}

/// The 25 features in index order. `FEATURES[i]` is the function `compute(x, i)`
/// dispatches to, and `FEATURE_NAMES[i]` is its canonical name.
pub const FEATURES: [fn(&[f64]) -> f64; N_CATCH22] = [
    dn_outlier_include_n_001_mdrmd,
    dn_outlier_include_p_001_mdrmd,
    dn_histogram_mode_5,
    dn_histogram_mode_10,
    co_embed2_dist_tau_d_expfit_meandiff,
    co_f1ecac,
    co_first_min_ac,
    co_histogram_ami_even_2_5,
    co_trev_1_num,
    fc_local_simple_mean1_tauresrat,
    fc_local_simple_mean3_stderr,
    in_auto_mutual_info_stats_40_gaussian_fmmi,
    md_hrv_classic_pnn40,
    sb_binary_stats_diff_longstretch0,
    sb_binary_stats_mean_longstretch1,
    sb_motif_three_quantile_hh,
    sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1,
    sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1,
    sp_summaries_welch_rect_area_5_1,
    sp_summaries_welch_rect_centroid,
    sb_transition_matrix_3ac_sumdiagcov,
    pd_periodicity_wang_th0_01,
    dn_mean,
    dn_spread_std,
    slope_of_linear_fit,
];

/// Canonical catch22 names, in the same order as [`FEATURES`].
pub const FEATURE_NAMES: [&str; N_CATCH22] = [
    "DN_OutlierInclude_n_001_mdrmd",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_diff_longstretch0",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "DN_Mean",
    "DN_Spread_Std",
    "SlopeOfLinearFit",
];

/// The number of features that operate on the z-scored series. The remaining
/// [`N_CATCH22`] - [`N_NORMALIZED`] are the catch24 additions plus the slope,
/// which are computed on the raw series.
pub const N_NORMALIZED: usize = 22;

pub fn compute(x: &[f64], n: usize) -> Result<f64, Catch22Error> {
    validate_input(x)?;
    if n >= N_CATCH22 {
        return Err(Catch22Error::InvalidFeatureIndex {
            index: n,
            max: N_CATCH22 - 1,
        });
    }

    Ok(compute_unchecked(x, n))
}

/// Computes all 25 features on `x` exactly as given, doing no normalisation.
/// Most callers want [`compute_all_normalized`] instead.
pub fn compute_all(x: &[f64]) -> Result<[f64; N_CATCH22], Catch22Error> {
    let stats = compute_basic_stats_checked(x)?;
    Ok(compute_all_inner(x, stats))
}

/// The full pipeline, matching what upstream's `C/main.c` does: features
/// 0..=21 are computed on the z-scored series, and 22..=24 (mean, standard
/// deviation, slope) on the raw one — the catch24 convention.
///
/// A constant series has zero standard deviation and therefore z-scores to
/// `NaN`; that is reported as [`Catch22Error::NonFiniteValue`] rather than
/// propagated.
pub fn compute_all_normalized(x: &[f64]) -> Result<[f64; N_CATCH22], Catch22Error> {
    let raw_stats = compute_basic_stats_checked(x)?;

    let z = zscore(x);
    let z_stats = compute_basic_stats_checked(&z)?;
    let mut out = compute_all_inner(&z, z_stats);

    out[22] = raw_stats.mean;
    out[23] = raw_stats.std_dev;
    out[24] = raw_stats.slope;

    Ok(out)
}

pub fn compute_unchecked(x: &[f64], n: usize) -> f64 {
    debug_assert!(n < N_CATCH22);
    FEATURES[n](x)
}

pub fn compute_all_unchecked(x: &[f64]) -> [f64; N_CATCH22] {
    let stats = compute_basic_stats_unchecked(x);
    compute_all_inner(x, stats)
}

fn compute_all_inner(x: &[f64], stats: BasicStats) -> [f64; N_CATCH22] {
    let mut out = [0.0; N_CATCH22];

    let autocorr = statistics::autocorr(x);
    let tau = statistics::first_zero_from_autocorr(&autocorr, x.len());

    out[0] = catch22::dn_outlier_include_np_001_mdrmd(x, false);
    out[1] = catch22::dn_outlier_include_np_001_mdrmd(x, true);
    out[2] = catch22::dn_histogram_mode_n(x, 5);
    out[3] = catch22::dn_histogram_mode_n(x, 10);
    out[4] = catch22::co_embed2_dist_tau_d_expfit_meandiff_with_tau(x, tau);
    out[5] = catch22::co_f1ecac_from_autocorr(x, &autocorr);
    out[6] = catch22::co_first_min_ac_from_autocorr(x, &autocorr);
    out[7] = catch22::co_histogram_ami_even_tau_bins(x, 2, 5);
    out[8] = catch22::co_trev_1_num(x);
    out[9] = catch22::fc_local_simple_mean_tauresrat_from_autocorr(x, &autocorr, 1);
    out[10] = catch22::fc_local_simple_mean_stderr(x, 3);
    out[11] = catch22::in_auto_mutual_info_stats_tau_gaussian_fmmi(x, 40.0);
    out[12] = catch22::md_hrv_classic_pnn(x, 40);
    out[13] = catch22::sb_binary_stats_diff_longstretch0(x);
    out[14] = catch22::sb_binary_stats_mean_longstretch1(x);
    out[15] = catch22::sb_motif_three_quantile_hh(x);
    out[16] = catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 1, "rsrangefit");
    out[17] = catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 2, "dfa");
    out[18] = catch22::sp_summaries_welch_rect(x, "area_5_1");
    out[19] = catch22::sp_summaries_welch_rect(x, "centroid");
    out[20] = catch22::sb_transition_matrix_3ac_sumdiagcov_with_tau(x, tau);
    out[21] = catch22::pd_periodicity_wang_th0_01(x);
    out[22] = stats.mean;
    out[23] = stats.std_dev;
    out[24] = stats.slope;

    out
}

/// Z-scores a series the way the original C does it (`zscore_norm2` in
/// `C/stats.c`): the mean is the plain arithmetic mean, and the standard
/// deviation is the *sample* standard deviation, i.e. divided by `n - 1`.
///
/// The `n - 1` matters. Several catch22 features are not scale-invariant —
/// the histogram modes, the AMI bin edges, the 0.01 threshold grid in
/// `DN_OutlierInclude`, and `MD_hrv_classic_pnn40` all compare against absolute
/// magnitudes — so normalising with the population standard deviation shifts
/// their values away from the reference implementation.
pub fn zscore(x: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = (x.iter().map(|val| (val - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt();
    x.iter().map(|val| (val - mean) / std).collect()
}
