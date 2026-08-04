//! Degenerate inputs and the error contract.
// Feature indices are identities here, not cursors: `FEATURES[i]` and
// `FEATURE_NAMES[i]` must line up, so the loops index deliberately.
#![allow(clippy::needless_range_loop)]

use catch22::{
    Catch22Error, FEATURES, N_CATCH22, compute, compute_all, compute_all_normalized, zscore,
};

#[test]
fn rejects_input_shorter_than_minimum() {
    for len in 0..4 {
        let series = vec![1.0; len];
        assert!(matches!(
            compute_all(&series).unwrap_err(),
            Catch22Error::InputTooShort { .. }
        ));
        assert!(matches!(
            compute(&series, 0).unwrap_err(),
            Catch22Error::InputTooShort { .. }
        ));
    }
}

#[test]
fn rejects_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let series = vec![1.0, 2.0, bad, 4.0];
        assert!(matches!(
            compute_all(&series).unwrap_err(),
            Catch22Error::NonFiniteValue { index: 2, .. }
        ));
    }
}

#[test]
fn rejects_out_of_range_feature_index() {
    let series = vec![1.0, 2.0, 3.0, 4.0];
    assert!(matches!(
        compute(&series, N_CATCH22).unwrap_err(),
        Catch22Error::InvalidFeatureIndex { .. }
    ));
}

/// A constant series has zero standard deviation, so the normalised pipeline
/// cannot z-score it. That must surface as an error rather than a panic or a
/// silently wrong number — this used to divide by zero inside
/// SB_TransitionMatrix.
#[test]
fn constant_series_is_rejected_by_the_normalized_pipeline() {
    for value in [0.0, 1.0, -7.5, 1e12] {
        let series = vec![value; 32];
        assert!(matches!(
            compute_all_normalized(&series).unwrap_err(),
            Catch22Error::NonFiniteValue { .. }
        ));
    }
}

/// The un-normalised path is defined on a constant series, and must not panic.
#[test]
fn constant_series_does_not_panic_unnormalized() {
    let series = vec![3.0; 32];
    let values = compute_all(&series).expect("constant series is finite");
    assert_eq!(values[22], 3.0, "mean of a constant series");
    assert_eq!(values[23], 0.0, "standard deviation of a constant series");
    assert!(
        values[20].is_nan(),
        "SB_TransitionMatrix is NaN on a constant series, as in C"
    );
}

/// No input in this set may panic through any entry point.
#[test]
fn awkward_inputs_do_not_panic() {
    let cases: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![1e-300, 2e-300, 3e-300, 4e-300],
        vec![1e300, -1e300, 1e300, -1e300],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0],
        (0..40)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect(),
        (0..40).map(|i| i as f64).collect(),
        (0..40).map(|i| -(i as f64)).collect(),
        vec![0.0; 39].into_iter().chain([1.0]).collect(),
    ];

    for series in cases {
        // Squaring a value near the top of the exponent range legitimately
        // overflows to infinity, in C too, so only inputs with headroom are
        // held to a finite result.
        let has_headroom = series.iter().all(|v| v.abs() < 1e150);

        for feature in 0..N_CATCH22 {
            let value = FEATURES[feature](&series);
            if has_headroom {
                assert!(
                    !value.is_infinite(),
                    "feature {feature} returned an infinity for {series:?}"
                );
            }
        }
        let _ = compute_all(&series).expect("finite input");
    }
}

#[test]
fn zscore_matches_reference_definition() {
    let series = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let z = zscore(&series);

    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let sample_std =
        (series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64).sqrt();

    for (value, zed) in series.iter().zip(&z) {
        assert!(((value - mean) / sample_std - zed).abs() < 1e-15);
    }
}

/// `FC_LocalSimple_mean3_stderr` takes the standard deviation of a single
/// residual when the series is exactly four samples long. C divides by
/// `n - 1 == 0` there and returns NaN; returning 0.0 instead would look like a
/// legitimate value.
#[test]
fn single_residual_stderr_is_nan_like_c() {
    let series = vec![1.0, 2.0, 3.0, 4.0];
    assert!(FEATURES[10](&series).is_nan());
}

/// `DN_OutlierInclude` walks a 0.01-spaced threshold grid up to the maximum of
/// the series, so a raw series with an enormous range would ask for an
/// impossible allocation. C overflows an `int` and invokes undefined behaviour;
/// this must refuse cleanly instead of aborting the process it is embedded in.
#[test]
fn extreme_amplitude_does_not_abort() {
    let series = vec![1e300, -1e300, 1e300, -1e300];
    assert!(FEATURES[0](&series).is_nan());
    assert!(FEATURES[1](&series).is_nan());

    // A z-scored series, which is what catch22 actually feeds it, stays well
    // inside the bound no matter how extreme the raw input was.
    let z = zscore(&series);
    assert!(FEATURES[0](&z).is_finite());
}

/// Short series drive `CO_Embed2_Dist`'s tau to zero through the `size / 10`
/// cap. C carries on and embeds the series against itself rather than bailing
/// out, so this must be a real number.
#[test]
fn embed2_dist_handles_zero_tau() {
    let series = vec![1.0, -2.0, 3.5, 4.0, -1.5];
    let value = FEATURES[4](&series);
    assert!(value.is_finite() && value != 0.0, "got {value}");
}
