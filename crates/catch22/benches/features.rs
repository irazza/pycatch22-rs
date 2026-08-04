use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use catch22::{compute_all, compute_all_unchecked, compute_unchecked, zscore};

fn build_series(len: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        let x = i as f64;
        // Deterministic, non-constant series with small trend.
        data.push((x * 0.01).sin() + (x * 0.0001));
    }
    data
}

fn bench_compute_all(c: &mut Criterion) {
    let data_1k = build_series(1_000);
    let data_10k = build_series(10_000);
    let data_1k_z = zscore(&data_1k);
    let data_10k_z = zscore(&data_10k);

    c.bench_function("compute_all_1k", |b| {
        b.iter(|| compute_all_unchecked(black_box(&data_1k)))
    });

    c.bench_function("compute_all_checked_1k", |b| {
        b.iter(|| compute_all(black_box(&data_1k)).unwrap())
    });

    c.bench_function("compute_all_10k", |b| {
        b.iter(|| compute_all_unchecked(black_box(&data_10k)))
    });

    c.bench_function("compute_all_checked_10k", |b| {
        b.iter(|| compute_all(black_box(&data_10k)).unwrap())
    });

    c.bench_function("compute_all_zscore_1k", |b| {
        b.iter(|| compute_all_unchecked(black_box(&data_1k_z)))
    });

    c.bench_function("compute_all_zscore_10k", |b| {
        b.iter(|| compute_all_unchecked(black_box(&data_10k_z)))
    });

    const HEAVY_FEATURES: &[usize] = &[0, 1, 4, 5, 6, 9, 11, 16, 17, 18, 19, 20, 21];

    for &feature in HEAVY_FEATURES {
        let name_1k = format!("feature_{feature:02}_1k");
        c.bench_function(&name_1k, |b| {
            b.iter(|| compute_unchecked(black_box(&data_1k), feature))
        });

        let name_10k = format!("feature_{feature:02}_10k");
        c.bench_function(&name_10k, |b| {
            b.iter(|| compute_unchecked(black_box(&data_10k), feature))
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(60)
        .measurement_time(Duration::from_secs(3));
    targets = bench_compute_all
}
criterion_main!(benches);
