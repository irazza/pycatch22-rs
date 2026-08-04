//! The four ways to reach a feature value must agree.
//!
//! `compute_all` shares one autocorrelation across features 4, 5, 6, 9 and 20
//! and reuses a single mean/std/slope pass; the named functions and
//! `compute(x, n)` each compute from scratch. Those paths are easy to let drift
//! apart, so they are pinned against each other here.
// Feature indices are identities here, not cursors: `FEATURES[i]` and
// `FEATURE_NAMES[i]` must line up, so the loops index deliberately.
#![allow(clippy::needless_range_loop)]

use catch22::{
    FEATURES, N_CATCH22, compute, compute_all, compute_all_normalized, compute_all_unchecked,
    compute_unchecked, zscore,
};

fn sample_series() -> Vec<Vec<f64>> {
    let mut out = vec![
        (0..100)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.001)
            .collect::<Vec<f64>>(),
        (0..256).map(|i| ((i * 37) % 19) as f64).collect(),
        (0..64).map(|i| i as f64).collect(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![1.0, 2.0, 3.0, 4.0],
    ];

    // A deterministic pseudo-random walk, for something less structured.
    let mut state = 12345u64;
    let mut walk = Vec::with_capacity(512);
    let mut value = 0.0;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        value += ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        walk.push(value);
    }
    out.push(walk);
    out
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

#[test]
fn compute_all_matches_individual_features() {
    for series in sample_series() {
        let all = compute_all(&series).expect("valid series");
        for feature in 0..N_CATCH22 {
            let individual = FEATURES[feature](&series);
            assert!(
                same(all[feature], individual),
                "feature {feature} ({}): compute_all={}, direct={}",
                catch22::FEATURE_NAMES[feature],
                all[feature],
                individual
            );
        }
    }
}

#[test]
fn compute_matches_compute_unchecked() {
    for series in sample_series() {
        for feature in 0..N_CATCH22 {
            let checked = compute(&series, feature).expect("valid series");
            let unchecked = compute_unchecked(&series, feature);
            assert!(same(checked, unchecked), "feature {feature}");
        }
    }
}

#[test]
fn compute_all_matches_unchecked() {
    for series in sample_series() {
        let checked = compute_all(&series).expect("valid series");
        let unchecked = compute_all_unchecked(&series);
        for feature in 0..N_CATCH22 {
            assert!(
                same(checked[feature], unchecked[feature]),
                "feature {feature}"
            );
        }
    }
}

/// `compute_all_normalized` is the reference pipeline: z-score, then features
/// 0..=21 on the z-scored series and the rest on the raw one.
#[test]
fn normalized_pipeline_splits_raw_and_zscored() {
    for series in sample_series() {
        let normalized = compute_all_normalized(&series).expect("valid series");
        let z = zscore(&series);

        for feature in 0..catch22::N_NORMALIZED {
            let expected = FEATURES[feature](&z);
            assert!(
                same(normalized[feature], expected),
                "feature {feature} should be computed on the z-scored series"
            );
        }
        for feature in catch22::N_NORMALIZED..N_CATCH22 {
            let expected = FEATURES[feature](&series);
            assert!(
                same(normalized[feature], expected),
                "feature {feature} should be computed on the raw series"
            );
        }
    }
}

/// Feature order is API — downstream code indexes into these arrays.
#[test]
fn feature_tables_line_up() {
    assert_eq!(catch22::FEATURE_NAMES.len(), N_CATCH22);
    assert_eq!(FEATURES.len(), N_CATCH22);

    let mut names: Vec<&str> = catch22::FEATURE_NAMES.to_vec();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), N_CATCH22, "feature names must be unique");

    assert_eq!(catch22::FEATURE_NAMES[0], "DN_OutlierInclude_n_001_mdrmd");
    assert_eq!(catch22::FEATURE_NAMES[22], "DN_Mean");
    assert_eq!(catch22::FEATURE_NAMES[24], "SlopeOfLinearFit");
}

/// Repeated calls must not drift. The FFT plan and spline basis caches are
/// keyed by length and shared across calls, so a bug there would show up as a
/// second call disagreeing with the first.
#[test]
fn caches_do_not_change_results() {
    for series in sample_series() {
        let first = compute_all_normalized(&series).expect("valid series");
        for _ in 0..3 {
            let again = compute_all_normalized(&series).expect("valid series");
            for feature in 0..N_CATCH22 {
                assert!(
                    same(first[feature], again[feature]),
                    "feature {feature} changed between calls"
                );
            }
        }
    }
}

/// Series of different lengths interleaved, to catch a cache keyed or sized
/// incorrectly.
#[test]
fn interleaved_lengths_do_not_interfere() {
    let series = sample_series();
    let alone: Vec<_> = series
        .iter()
        .map(|s| compute_all_normalized(s).expect("valid series"))
        .collect();

    for _ in 0..3 {
        for (index, s) in series.iter().enumerate() {
            let interleaved = compute_all_normalized(s).expect("valid series");
            for feature in 0..N_CATCH22 {
                assert!(
                    same(alone[index][feature], interleaved[feature]),
                    "series {index} feature {feature} differs when interleaved"
                );
            }
        }
    }
}
