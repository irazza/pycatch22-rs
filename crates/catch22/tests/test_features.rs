// Feature indices are identities here, not cursors: `FEATURES[i]` and
// `FEATURE_NAMES[i]` must line up, so the loops index deliberately.
#![allow(clippy::needless_range_loop)]

use catch22::{
    Catch22Error, N_CATCH22, compute, compute_all, compute_all_unchecked, compute_unchecked, zscore,
};

#[test]
fn test_catch22() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n_features = 22;

    let features = (0..n_features)
        .map(|i| compute(&time_series, i).unwrap())
        .collect::<Vec<_>>();
    println!("Catch22 features: {:?}", features);
}

#[test]
fn test_compute_all_matches_compute() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let all = compute_all(&time_series).unwrap();

    for i in 0..N_CATCH22 {
        let single = compute(&time_series, i).unwrap();
        let combined = all[i];
        assert!(
            (combined - single).abs() < 1e-12 || (combined.is_nan() && single.is_nan()),
            "feature {i} mismatch: compute_all={combined}, compute={single}"
        );
    }
}

#[test]
fn test_compute_unchecked_matches_checked() {
    let time_series = (0..100)
        .map(|i| (i as f64 * 0.01).sin() + (i as f64 * 0.0001))
        .collect::<Vec<_>>();

    for i in 0..N_CATCH22 {
        let checked = compute(&time_series, i).unwrap();
        let unchecked = compute_unchecked(&time_series, i);
        assert!(
            (checked - unchecked).abs() < 1e-12 || (checked.is_nan() && unchecked.is_nan()),
            "feature {i} mismatch: checked={checked}, unchecked={unchecked}"
        );
    }
}

#[test]
fn test_compute_all_unchecked_matches_checked() {
    let time_series = (0..100)
        .map(|i| (i as f64 * 0.01).sin() + (i as f64 * 0.0001))
        .collect::<Vec<_>>();
    let checked = compute_all(&time_series).unwrap();
    let unchecked = compute_all_unchecked(&time_series);

    for i in 0..N_CATCH22 {
        assert!(
            (checked[i] - unchecked[i]).abs() < 1e-12
                || (checked[i].is_nan() && unchecked[i].is_nan()),
            "feature {i} mismatch: checked={}, unchecked={}",
            checked[i],
            unchecked[i]
        );
    }
}

#[test]
fn test_invalid_index() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0];
    let err = compute(&time_series, N_CATCH22).unwrap_err();
    assert!(matches!(err, Catch22Error::InvalidFeatureIndex { .. }));
}

#[test]
fn test_non_finite_input() {
    let time_series = vec![1.0, 2.0, f64::NAN, 4.0];
    let err = compute_all(&time_series).unwrap_err();
    assert!(matches!(err, Catch22Error::NonFiniteValue { .. }));
}

#[test]
fn test_non_finite_infinite_input() {
    let time_series = vec![1.0, f64::INFINITY, 3.0, 4.0];
    let err = compute_all(&time_series).unwrap_err();
    assert!(matches!(err, Catch22Error::NonFiniteValue { .. }));
}

#[test]
fn test_input_too_short() {
    let time_series = vec![1.0, 2.0, 3.0];
    let err = compute_all(&time_series).unwrap_err();
    assert!(matches!(err, Catch22Error::InputTooShort { .. }));
}

#[test]
fn test_compute_input_too_short() {
    let time_series = vec![1.0, 2.0, 3.0];
    let err = compute(&time_series, 0).unwrap_err();
    assert!(matches!(err, Catch22Error::InputTooShort { .. }));
}

#[test]
fn test_zscore_mean_std() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0];
    let z = zscore(&time_series);

    // `zscore` matches C's `zscore_norm2`, which divides by the *sample*
    // standard deviation, so it is the ddof=1 std of the result that is 1.
    let mean = z.iter().sum::<f64>() / z.len() as f64;
    let variance = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (z.len() - 1) as f64;
    let std = variance.sqrt();

    assert!(mean.abs() < 1e-12, "zscore mean was {mean}");
    assert!((std - 1.0).abs() < 1e-12, "zscore sample std was {std}");
}

/// Guards the `n - 1` in `zscore` against a silent regression to the
/// population standard deviation: the two differ by sqrt((n-1)/n), which is
/// small enough to look like noise but shifts every scale-dependent feature.
#[test]
fn test_zscore_uses_sample_std() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0];
    let z = zscore(&time_series);

    let mean = 2.5;
    let sample_std = ((1.0f64 - mean).powi(2)
        + (2.0f64 - mean).powi(2)
        + (3.0f64 - mean).powi(2)
        + (4.0f64 - mean).powi(2))
        / 3.0;
    let expected_first = (1.0 - mean) / sample_std.sqrt();

    assert!(
        (z[0] - expected_first).abs() < 1e-15,
        "expected {expected_first}, got {}",
        z[0]
    );
}
