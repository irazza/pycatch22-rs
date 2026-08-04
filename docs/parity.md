# Parity with the original catch22 C implementation

This document records exactly what was compared, against what, and at what
tolerance. The machine-generated per-feature table lives in
[`parity-report.md`](parity-report.md). Reproduce both with:

```bash
bash tools/c_reference/fetch.sh
make -C tools/c_reference
cargo run --release -p ucr_check -- --ucr-root ~/DATA/ucr --report docs/parity-report.md
```

## Reference

The oracle is the original C implementation,
[`DynamicsAndNeuralSystems/catch22`](https://github.com/DynamicsAndNeuralSystems/catch22),
pinned at the commit in `tools/c_reference/PINNED_SHA`. Upstream's `C/main.c` is
interactive and reads one series per file, so `tools/c_reference/driver.c` drives
the same feature functions over a UCR `.tsv` instead. It mirrors what upstream's
`main.c` does: features 0–21 are computed on the series after `zscore_norm2`,
and features 22–24 (mean, standard deviation, slope) on the raw series, which is
the catch24 convention.

Values cross the boundary as raw little-endian `f64`, so nothing round-trips
through decimal text.

## Corpus

The full UCR archive: 128 datasets, 256 `.tsv` files, **191,158 series**,
4.78 million feature values.

## Tolerance

- Features whose value is an integer or a ratio of small integers — 6, 11, 13,
  14, 16, 17, 21 — must match **exactly**.
- Every other feature must satisfy `|rust - c| <= 1e-12 + 1e-9 * |c|`.

## Result

**21 of 25 features match on every one of the 191,158 series.**

The remaining four differ on a combined 39 of 4.78 million values (0.0008%),
all of them in the autocorrelation-threshold family:

| # | feature | differing series | share |
|---|---------|------------------|-------|
| 4 | CO_Embed2_Dist_tau_d_expfit_meandiff | 1 | 0.0005% |
| 6 | CO_FirstMin_ac | 1 | 0.0005% |
| 9 | FC_LocalSimple_mean1_tauresrat | 36 | 0.019% |
| 20 | SB_TransitionMatrix_3ac_sumdiagcov | 1 | 0.0005% |

### Why these four, and why this is not a defect

All four consume an integer derived from a *threshold crossing* of the
autocorrelation: `CO_FirstMin_ac` takes the first strict local minimum, and the
other three take `first_zero` (directly, or as the `tau` used to embed or
downsample). Each is a discrete decision made by comparing floating-point
numbers that are mathematically equal, or exactly zero.

**Example 1 — `ElectricDevices_TRAIN` row 4859** (features 4, 6 and 20: one
series, one shared cause). The series takes only four distinct values. Computed
in exact rational arithmetic, its autocovariance at lags 2 and 3 differs by
`2.7e-17`, while one ULP at that magnitude is `5.6e-17`. The two quantities
therefore **round to the same double**, and `CO_FirstMin_ac`'s test
`ac[i] < ac[i+1]` is decided purely by which way each FFT happened to round.
C answers 3, this implementation answers 2, and a hypothetical
infinitely-accurate implementation would answer 5. There is no "correct" double
here to converge on.

**Example 2 — `ScreenType_TRAIN` row 179** (feature 9). The residuals of the
one-step mean forecast are white, so their autocorrelation at lags 2 through 6
is exactly zero mathematically and lands at `~1e-18` in practice:

```
lag 1:  1.860e-01
lag 2:  1.781e-18   <- first_zero asks whether this is positive
lag 3: -2.085e-18
lag 4:  1.251e-17
lag 5:  7.819e-19
lag 6:  7.797e-18
lag 7: -1.098e-01
```

`first_zero` scans for the first non-positive value, so it is being asked for
the sign of zero. C's FFT rounds lag 2 to `<= 0` and stops there; rustfft keeps
it positive and stops at lag 7. The feature is intrinsically unstable on such
series, independently of language.

Matching C on these cases would require reproducing its FFT bit-for-bit — its
own naive recursive transform, in its own operation order — which costs the
speed this port exists to provide. The trade was made deliberately: match to
tolerance, keep the fast transform, and state the exception here.

## Defects found and fixed

The sweep was first run *before* any fix, to establish a baseline. At that point
only **14 of 25** features matched. The causes:

1. **`welch()` produced an empty spectrum.** The segment count was
   `floor(size / windowWidth) - 1`, which is 0 for the full-length rectangular
   window catch22 uses, so the accumulation loop never ran. C uses
   `floor(size / (windowWidth / 2.0)) - 1`, which is 1.
2. **`welch()` filled its FFT buffer at the wrong index** (`f[i]`, the segment
   index, instead of `f[j]`), and never re-zeroed the zero-pad region between
   segments.
3. **The power spectrum and the frequency vector were swapped at the call
   site**: `welch` returned `(freq, power)` and `SP_Summaries_welch_rect`
   destructured it as `(power, freq)`. They are now returned as a named `Welch`
   struct so this cannot recur.
4. **`area_5_1` summed nothing.** Its guard, `w[i] >= 5.0 && w[i] <= 1.0`,
   cannot be satisfied. C sums the lowest fifth of the spectrum unconditionally.

   Together, 1–4 made features 18 and 19 return `NaN` for every input — the
   empty spectrum divides by a zero normalisation factor.

5. **`zscore` divided by the population standard deviation.** C's
   `zscore_norm2` divides by the *sample* standard deviation (`n - 1`). Every
   feature that is not scale-invariant was therefore computed on a slightly
   mis-scaled series: features 2, 3, 4, 8 and 10 mismatched on 100% of series,
   and 0, 1, 7 and 12 on a few percent.
6. **`DN_OutlierInclude` swallowed a load-bearing NaN.** C computes
   `mean(Dt_exc, highSize-1)`, which is `0.0/0` → NaN when exactly one sample
   clears the threshold; that NaN is what sets `fbi` and caps the trim limit.
   Substituting `0.0` silently disabled the cap.
7. **`compute_all` panicked on a constant series.** `SB_TransitionMatrix`'s
   public entry point guards on `is_constant`, but `compute_all` called the
   shared-`tau` variant directly, bypassing it; `tau` is then 0 and the
   downsampling step divides by zero. C returns `NaN`. The guard now lives in
   the shared variant so both paths agree.

### Found by the edge-case fixtures

The UCR archive has no series shorter than 24 samples and none with an extreme
dynamic range, so these needed synthetic inputs (`edge:*` in the fixture):

8. **`FC_LocalSimple_mean3_stderr` returned 0.0 where C returns NaN.** With a
   four-sample series the forecast leaves exactly one residual, and C's
   `stddev` divides by `n - 1 == 0`. `std_dev` short-circuited to 0.0 for fewer
   than two samples; it now reproduces C's NaN (and its `-0.0` for an empty
   slice).
9. **`CO_Embed2_Dist` bailed out when tau was 0.** The `size / 10` cap drives
   tau to zero for any series shorter than 10 samples. C carries on and embeds
   the series against itself, which is perfectly well defined; an added
   `if tau == 0 { return 0.0 }` guard was returning a constant instead.
10. **`compute_all` and `dn_mean` disagreed in the last bits.** `compute_all`
    computed the mean and standard deviation with a fused Welford pass while
    features 22 and 23 used the plain two-pass formulas. Welford is better
    conditioned, but C uses the plain sum, so both paths now route through the
    same functions.
11. **Unsigned underflow on out-of-range histogram bins.** `histbinassign`
    yields 0 for a value below no edge at all, which happens when the top bin
    edge rounds down onto the value itself at extreme magnitudes.
    `CO_HistogramAMI` then computed `bins1[i] - 1` in `usize` and panicked; C
    does this in `int` and goes negative. Now done in `i64`.
12. **Unlabelled positions in `SB_MotifThree`.** `coarsegrain` can leave a
    position in no quantile band (`th[0] -= 1.0` is a no-op once 1.0 falls below
    the value's ULP). The original index-list formulation skipped those
    implicitly; the flattened counting version has to skip them explicitly.
13. **`DN_OutlierInclude` attempted an impossible allocation.** Its threshold
    grid steps by 0.01 up to the series maximum, so a raw series with a huge
    range asks for an astronomically large one. C converts the count to `int`
    and overflows into undefined behaviour, so there is no reference result to
    reproduce; this now returns NaN rather than aborting the process it is
    embedded in. Unreachable through the normalised pipeline, where the maximum
    of a z-scored series is at most sqrt(n - 1).

## Deliberate divergences from C

- **Non-finite input is rejected, not propagated.** C checks for `NaN` at the
  top of each feature and returns `NaN`. This implementation validates once and
  returns `Catch22Error::NonFiniteValue` (a Python `ValueError`). A constant
  series z-scores to all-`NaN`, so it lands here; the sweep audits those rows
  separately rather than scoring them as mismatches. The UCR archive contains no
  such rows.
- **Input shorter than 4 samples is rejected** rather than read out of bounds.
