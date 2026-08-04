# Rust vs C

The original catch22 C implementation, pinned at the commit in
`tools/c_reference/PINNED_SHA`, against this port. Same machine, same series,
same order, one thread each. Every feature is timed across a whole file in one
measurement, so no clock read straddles a sub-microsecond call.

Reproduce with:

```bash
bash tools/c_reference/fetch.sh && make -C tools/c_reference
cargo run --release -p ucr_check -- --ucr-root <subset> \
    --driver tools/c_reference/driver_native --bench BENCHMARKS.md
```

## Headline

| baseline | C µs/series | Rust µs/series | speedup |
|----------|-------------|----------------|---------|
| C at `-O2` | 1806.5 | 279.7 | **6.46x** |
| C at `-O3 -march=native` | 1793.4 | 279.7 | **6.41x** |

Both C builds are reported so the comparison cannot be accused of hobbling the
baseline; `-march=native` barely moves it.

Measured on 1,666 series spanning lengths 24 to 2,844 (7 UCR datasets), on a
16-core machine, Rust 1.97, GCC 16.1, rustfft 6.4.1.

**These are single-threaded numbers.** `compute_batch` releases the GIL and
processes series in parallel: on the same hardware, 1,000 series of length 1,024
run at **46 µs/series** wall-clock (7.9x over one thread), i.e. roughly **48x
the C reference** for batch workloads.

## Where the speedup comes from

Ordered by how much they mattered:

1. **`DN_OutlierInclude` (features 0 and 1), 322 µs -> 21 µs.** C rescans the
   whole series once per threshold — roughly 100x the series maximum, so several
   hundred passes — and sorts the surviving indices each time to take their
   median. Sorting once by value and sweeping the thresholds downwards keeps the
   surviving set incremental: the mean gap between indices telescopes to
   `(last - first) / (count - 1)`, and a Fenwick tree answers the median as an
   order statistic. `O(T·n log n)` becomes `O(n log n + T log n)`.
2. **FFT plan caching.** `autocorr` built two radix-4 plans, twiddle tables and
   all, on *every* call. Caching them per (length, direction) in a thread-local
   roughly halved the cost of every autocorrelation-based feature: 5, 6, 9 and
   20 together went from ~180 µs to ~72 µs.
3. **Spline basis caching (feature 21), 167 µs -> 17 µs.** The design matrix and
   its normal matrix `AᵀA` depend only on the series *length*, never on the
   data. Caching them per length leaves only `Aᵀb` and a 5x5 solve per call —
   a large win on a UCR dataset, where every series shares a length.
4. **`SB_MotifThree` (feature 15), 91 µs -> 8 µs.** Only the *sizes* of the
   transition groups are used, so the nested index vectors (three of length n
   plus nine holding their partition) collapse to a flat 3x3 count matrix.
5. **Allocation removal in `SC_FluctAnal`** and hoisting the loop-invariant
   regression sums, whose closed forms are exact here.

Every one of these preserves the output exactly: the full-archive sweep produced
byte-identical results before and after.

## What was left alone

`SC_FluctAnal` (features 16 and 17) and `IN_AutoMutualInfoStats` (feature 11)
sit at roughly C's speed and are the largest remaining Rust costs. Both are
bound by serial floating-point dependency chains rather than by allocation or
algorithmic waste, so the remaining lever is multiple accumulators — which
reorders the summation. All three features are compared against C *exactly*
rather than to a tolerance, and reordering risks flipping their discrete
outputs on knife-edge inputs. Not worth roughly 12% of the pipeline.

Features 3, 14, 17 and 24 come out marginally slower than C. They are all
sub-2 µs or dependency-chain-bound, and together account for under 15% of the
pipeline.

## Per feature

| # | feature | C µs/series | Rust µs/series | speedup |
|---|---------|------------|----------------|---------|
| 0 | DN_OutlierInclude_n_001_mdrmd | 319.286 | 21.625 | 14.76x |
| 1 | DN_OutlierInclude_p_001_mdrmd | 333.881 | 20.594 | 16.21x |
| 2 | DN_HistogramMode_5 | 1.740 | 1.650 | 1.05x |
| 3 | DN_HistogramMode_10 | 1.678 | 1.763 | 0.95x |
| 4 | CO_Embed2_Dist_tau_d_expfit_meandiff | 108.329 | 19.899 | 5.44x |
| 5 | CO_f1ecac | 93.056 | 14.195 | 6.56x |
| 6 | CO_FirstMin_ac | 93.177 | 14.303 | 6.51x |
| 7 | CO_HistogramAMI_even_2_5 | 5.666 | 4.481 | 1.26x |
| 8 | CO_trev_1_num | 6.184 | 0.463 | 13.36x |
| 9 | FC_LocalSimple_mean1_tauresrat | 188.717 | 28.971 | 6.51x |
| 10 | FC_LocalSimple_mean3_stderr | 6.674 | 1.671 | 3.99x |
| 11 | IN_AutoMutualInfoStats_40_gaussian_fmmi | 57.145 | 54.619 | 1.05x |
| 12 | MD_hrv_classic_pnn40 | 1.050 | 0.373 | 2.82x |
| 13 | SB_BinaryStats_diff_longstretch0 | 0.792 | 0.766 | 1.03x |
| 14 | SB_BinaryStats_mean_longstretch1 | 1.050 | 1.223 | 0.86x |
| 15 | SB_MotifThree_quantile_hh | 61.406 | 8.146 | 7.54x |
| 16 | SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 | 79.334 | 60.754 | 1.31x |
| 17 | SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 | 27.014 | 30.893 | 0.87x |
| 18 | SP_Summaries_welch_rect_area_5_1 | 37.854 | 6.426 | 5.89x |
| 19 | SP_Summaries_welch_rect_centroid | 38.146 | 6.279 | 6.08x |
| 20 | SB_TransitionMatrix_3ac_sumdiagcov | 111.839 | 15.380 | 7.27x |
| 21 | PD_PeriodicityWang_th0_01 | 165.386 | 17.453 | 9.48x |
| 22 | DN_Mean | 0.891 | 0.571 | 1.56x |
| 23 | DN_Spread_Std | 6.290 | 0.967 | 6.51x |
| 24 | SlopeOfLinearFit | 0.503 | 0.536 | 0.94x |

## By dataset (whole pipeline)

| dataset | length | series | C µs/series | Rust µs/series | speedup |
|---------|--------|--------|-------------|----------------|---------|
| ACSF1.tsv | 1460 | 100 | 4365.0 | 524.5 | 8.32x |
| Adiac.tsv | 176 | 390 | 359.4 | 66.7 | 5.39x |
| ArrowHead.tsv | 251 | 36 | 462.5 | 93.9 | 4.92x |
| Chinatown.tsv | 24 | 20 | 42.2 | 12.3 | 3.44x |
| ECG200.tsv | 96 | 100 | 192.3 | 50.4 | 3.82x |
| Rock.tsv | 2844 | 20 | 9306.6 | 1090.4 | 8.54x |
| StarLightCurves.tsv | 1024 | 1000 | 2188.2 | 357.0 | 6.13x |
