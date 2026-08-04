use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::vec;

use rustfft::{Fft, FftDirection, algorithm::Radix4, num_complex::Complex};

thread_local! {
    /// Radix-4 plans, keyed by (length, inverse). Building a plan computes the
    /// twiddle table, which is why this is cached: the transform lengths here
    /// come from a handful of series lengths, but `autocorr` is called for
    /// every series, and it used to build two plans per call.
    static FFT_PLANS: RefCell<HashMap<(usize, bool), Rc<Radix4<f64>>>> =
        RefCell::new(HashMap::new());

    /// Scratch space for in-place transforms, grown to the largest size seen so
    /// far. `Fft::process` allocates this internally on every call.
    static FFT_SCRATCH: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Returns a cached radix-4 plan. The plan is deterministic in its inputs, so
/// caching changes no arithmetic — the same transform runs, just without
/// rebuilding the twiddles first.
fn fft_plan(len: usize, inverse: bool) -> Rc<Radix4<f64>> {
    FFT_PLANS.with(|cache| {
        Rc::clone(cache.borrow_mut().entry((len, inverse)).or_insert_with(|| {
            let direction = if inverse {
                FftDirection::Inverse
            } else {
                FftDirection::Forward
            };
            Rc::new(Radix4::new(len, direction))
        }))
    })
}

fn fft_in_place(plan: &Radix4<f64>, buffer: &mut [Complex<f64>]) {
    FFT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let needed = plan.get_inplace_scratch_len();
        if scratch.len() < needed {
            scratch.resize(needed, Complex::new(0.0, 0.0));
        }
        plan.process_with_scratch(buffer, &mut scratch[..needed]);
    });
}

pub fn min_(a: &[f64]) -> f64 {
    let mut min = a[0];
    for i in 1..a.len() {
        if a[i] < min {
            min = a[i];
        }
    }
    return min;
}

pub fn max_(a: &[f64]) -> f64 {
    let mut max = a[0];
    for i in 1..a.len() {
        if a[i] > max {
            max = a[i];
        }
    }
    return max;
}

pub fn is_constant(a: &[f64]) -> bool {
    a.iter().all(|&x| x == a[0])
}

pub fn mean(a: &[f64]) -> f64 {
    if a.len() == 0 {
        return 0.0;
    }
    a.iter().sum::<f64>() / a.len() as f64
}

pub fn median(a: &[f64]) -> f64 {
    if a.len() == 0 {
        return 0.0;
    }

    let mut a = a.to_vec();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    let n = a.len();
    if n % 2 == 0 {
        (a[n / 2] + a[n / 2 - 1]) / 2.0
    } else {
        a[n / 2]
    }
}

/// Sample standard deviation, matching C's `stddev` including its degenerate
/// cases: a single sample divides 0 by 0 and yields NaN, and an empty slice
/// divides 0 by -1 and yields -0.0. `FC_LocalSimple_mean3_stderr` reaches the
/// one-sample case whenever the series is exactly `train_length + 1` long.
pub fn std_dev(a: &[f64]) -> f64 {
    if a.is_empty() {
        return -0.0;
    }
    if a.len() == 1 {
        return f64::NAN;
    }

    // Two passes, as in C: the plain mean, then the sum of squared deviations.
    // A one-pass Welford update is better conditioned but rounds differently,
    // and DN_Spread_Std is compared against C bit for bit.
    let m = mean(a);
    let mut sum_sq = 0.0;
    for &value in a {
        let delta = value - m;
        sum_sq += delta * delta;
    }

    (sum_sq / (a.len() - 1) as f64).sqrt()
}

pub fn slope(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f64;
    let x_mean = (n_f + 1.0) / 2.0;
    let x2_mean = (n_f + 1.0) * (2.0 * n_f + 1.0) / 6.0;

    let mut y_sum = 0.0;
    let mut xy_sum = 0.0;
    for (i, &value) in a.iter().enumerate() {
        let x = (i + 1) as f64;
        y_sum += value;
        xy_sum += x * value;
    }

    let y_mean = y_sum / n_f;
    let xy_mean = xy_sum / n_f;
    let slope = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean);
    assert!(slope.is_finite());
    slope
}

pub fn histcounts(a: &[f64], n_bins: usize) -> (Vec<usize>, Vec<f64>) {
    let mut n_bins = n_bins;

    let max_val = max_(a);
    let min_val = min_(a);

    if n_bins <= 0 {
        n_bins = ((max_val - min_val) / (3.5 * std_dev(a) * (a.len() as f64).powf(-1.0 / 3.0)))
            .ceil() as usize;
    }

    let bin_step = (max_val - min_val) / n_bins as f64;

    let mut bin_counts = vec![0; n_bins];

    for i in 0..a.len() {
        let mut bin_ind = ((a[i] - min_val) / bin_step) as usize;
        bin_ind = bin_ind.min(n_bins - 1);

        bin_counts[bin_ind] += 1;
    }
    let mut bin_edges = vec![0.0; n_bins + 1];

    for i in 0..n_bins + 1 {
        bin_edges[i] = min_val + i as f64 * bin_step;
    }

    (bin_counts, bin_edges)
}

pub fn autocorr(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }

    let n = a.len().next_power_of_two() << 1;
    let m = mean(a);

    let mut buffer = vec![Complex::new(0.0, 0.0); n];

    for (i, value) in a.iter().enumerate() {
        buffer[i].re = value - m;
    }

    let fft = fft_plan(n, false);
    let ifft = fft_plan(n, true);

    fft_in_place(&fft, &mut buffer);
    for value in buffer.iter_mut() {
        *value = *value * value.conj();
    }
    fft_in_place(&ifft, &mut buffer);

    let norm = buffer[0];
    let mut out = Vec::with_capacity(a.len());
    for value in buffer.iter().take(a.len()) {
        out.push((value / norm).re);
    }

    out
}

pub fn autocovariance(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }

    let n = a.len().next_power_of_two() << 1;
    let mut buffer = vec![Complex::new(0.0, 0.0); n];

    for (i, value) in a.iter().enumerate() {
        buffer[i].re = *value;
    }

    let fft = fft_plan(n, false);
    let ifft = fft_plan(n, true);

    fft_in_place(&fft, &mut buffer);
    for value in buffer.iter_mut() {
        *value = *value * value.conj();
    }
    fft_in_place(&ifft, &mut buffer);

    let inv_n = 1.0 / n as f64;
    let mut out = Vec::with_capacity(a.len());
    for (lag, value) in buffer.iter().take(a.len()).enumerate() {
        let sum = value.re * inv_n;
        let denom = (a.len() - lag) as f64;
        out.push(sum / denom);
    }

    out
}

pub fn first_zero(a: &[f64], max_tau: usize) -> usize {
    let autocorr = autocorr(a);
    first_zero_from_autocorr(&autocorr, max_tau)
}

pub fn first_zero_from_autocorr(autocorr: &[f64], max_tau: usize) -> usize {
    let mut zero_cross_ind = 0;
    let max_tau = max_tau.min(autocorr.len());

    while zero_cross_ind < max_tau && autocorr[zero_cross_ind] > 0.0 {
        zero_cross_ind += 1;
    }

    zero_cross_ind
}

pub fn num_bins_auto(a: &[f64]) -> usize {
    let max_val = max_(a);
    let min_val = min_(a);

    if std_dev(a) < 0.001 {
        return 0;
    }

    let n_bins = ((max_val - min_val) / (3.5 * std_dev(a) * (a.len() as f64).powf(-1.0 / 3.0)))
        .ceil() as usize;
    return n_bins;
}

pub fn histbinassign(a: &[f64], bin_edges: &[f64]) -> Vec<usize> {
    let mut bin_identity = vec![0; a.len()];

    for i in 0..a.len() {
        for j in 0..bin_edges.len() {
            if a[i] < bin_edges[j] {
                bin_identity[i] = j;
                break;
            }
        }
    }

    return bin_identity;
}

pub fn histcount_edges(a: &[f64], bin_edges: &[f64]) -> Vec<usize> {
    let mut histcounts = vec![0; bin_edges.len()];

    for i in 0..a.len() {
        for j in 0..bin_edges.len() {
            if a[i] <= bin_edges[j] {
                histcounts[j] += 1;
                break;
            }
        }
    }

    return histcounts;
}
/// Direct per-lag autocovariance. Only used to pin `autocovariance` (which
/// computes every lag at once via FFT) against the straightforward definition.
#[cfg(test)]
pub fn autocov_lag(a: &[f64], lag: usize) -> f64 {
    cov_(&a[..a.len() - lag], &a[lag..])
}
#[cfg(test)]
fn cov_(a: &[f64], b: &[f64]) -> f64 {
    let mut covariance = 0.0;
    for i in 0..a.len() {
        covariance += a[i] * b[i];
    }

    return covariance / a.len() as f64;
}

pub fn autocorr_lag(a: &[f64], lag: usize) -> f64 {
    let mean_a = mean(&a[..a.len() - lag]);
    let mean_b = mean(&a[lag..]);

    corr(&a[..a.len() - lag], &a[lag..], mean_a, mean_b)
}

pub fn corr(a: &[f64], b: &[f64], mean_a: f64, mean_b: f64) -> f64 {
    let mut nom = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;

    for i in 0..b.len() {
        nom += (a[i] - mean_a) * (b[i] - mean_b);
        denom_a += (a[i] - mean_a) * (a[i] - mean_a);
        denom_b += (b[i] - mean_b) * (b[i] - mean_b);
    }

    return nom / (denom_a * denom_b).sqrt();
}

pub fn coarsegrain(a: &[f64], num_groups: usize) -> Vec<usize> {
    let mut labels = vec![0; a.len()];
    let mut th = vec![0.0; num_groups + 1];
    let ls = linspace(0.0, 1.0, num_groups + 1);

    let mut sorted = a.to_vec();
    sorted.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());

    for i in 0..num_groups + 1 {
        th[i] = quantile_sorted(&sorted, ls[i]);
    }

    th[0] -= 1.0;

    for i in 0..num_groups {
        for j in 0..a.len() {
            if a[j] > th[i] && a[j] <= th[i + 1] {
                labels[j] = i + 1;
            }
        }
    }
    labels
}

pub fn linspace(start: f64, end: f64, num_groups: usize) -> Vec<f64> {
    let mut out = vec![0.0; num_groups];
    let mut start = start;
    let step_size = (end - start) / (num_groups - 1) as f64;
    for i in 0..num_groups {
        out[i] = start;
        start += step_size;
    }
    return out;
}

fn quantile_sorted(a: &[f64], quantile: f64) -> f64 {
    let q = 0.5 / a.len() as f64;

    if quantile < q {
        return a[0];
    } else if quantile > (1.0 - q) {
        return a[a.len() - 1];
    }

    let quant_idx = a.len() as f64 * quantile - 0.5;
    let idx_left = quant_idx.floor() as usize;
    let idx_right = quant_idx.ceil() as usize;
    let value = a[idx_left]
        + (quant_idx - idx_left as f64) * (a[idx_right] - a[idx_left])
            / (idx_right - idx_left) as f64;
    value
}

pub fn f_entropy(a: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..a.len() {
        if a[i] > 0.0 {
            f += a[i] * a[i].ln();
        }
    }
    return -1.0 * f;
}

pub fn linreg(n: usize, x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut sumx = 0.0;
    let mut sumx2 = 0.0;
    let mut sumxy = 0.0;
    let mut sumy = 0.0;

    for i in 0..n {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
    }

    let denom = n as f64 * sumx2 - sumx * sumx;

    if denom == 0.0 {
        return (0.0, 0.0);
    }

    return (
        (n as f64 * sumxy - sumx * sumy) / denom,
        (sumy * sumx2 - sumx * sumxy) / denom,
    );
}

pub fn norm(a: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * a[i];
    }
    return sum.sqrt();
}

/// One-sided Welch power spectral density estimate.
///
/// Direct port of `welch()` in the original `C/SP_Summaries.c`. The returned
/// fields are named rather than positional because the two vectors are easy to
/// swap at the call site, and swapping them silently produces plausible-looking
/// nonsense.
pub struct Welch {
    /// Power spectral density, `nfft / 2 + 1` points.
    pub power: Vec<f64>,
    /// Frequencies matching `power`, in Hz.
    pub freq: Vec<f64>,
}

pub fn welch(a: &[f64], fs: f64, window: &[f64]) -> Welch {
    let window_width = window.len();
    let dt = 1.0 / fs;
    let df = 1.0 / (window_width.next_power_of_two() as f64) / dt;
    let m = mean(a);
    let nfft = a.len().next_power_of_two();

    // C: k = floor(size / (windowWidth / 2.0)) - 1. Dividing by the window
    // width instead of by half of it yields k = 0 for a full-length window,
    // which skips the accumulation loop entirely and leaves the spectrum empty.
    let k = ((a.len() as f64 / (window_width as f64 / 2.0)).floor() - 1.0).max(0.0) as usize;

    let kmu = k as f64 * norm(window).powi(2);

    let mut p = vec![0.0; nfft];
    let mut buffer = vec![Complex::new(0.0, 0.0); nfft];
    let fft = fft_plan(nfft, false);

    for i in 0..k {
        let offset = (i as f64 * window_width as f64 / 2.0) as usize;

        // Windowed, mean-subtracted segment, zero-padded out to nfft. The
        // buffer has to be refilled (including the pad) on every iteration
        // because `fft.process` transforms in place.
        for (j, slot) in buffer.iter_mut().enumerate().take(window_width) {
            *slot = Complex::new(window[j] * a[offset + j] - m, 0.0);
        }
        for slot in buffer.iter_mut().skip(window_width) {
            *slot = Complex::new(0.0, 0.0);
        }

        fft_in_place(&fft, &mut buffer);

        for (acc, value) in p.iter_mut().zip(buffer.iter()) {
            *acc += value.norm_sqr();
        }
    }

    let n_out = nfft / 2 + 1;
    let mut power = vec![0.0; n_out];
    for i in 0..n_out {
        power[i] = p[i] / kmu * dt;
        if i > 0 && i < n_out - 1 {
            power[i] *= 2.0;
        }
    }

    let freq = (0..n_out).map(|x| x as f64 * df).collect::<Vec<f64>>();

    Welch { power, freq }
}

pub fn cov(a: &[f64], b: &[f64]) -> f64 {
    let mut covariance = 0.0;

    let mean_x = mean(a);
    let mean_y = mean(b);

    for i in 0..a.len() {
        covariance += (a[i] - mean_x) * (b[i] - mean_y);
    }

    return covariance / (a.len() - 1) as f64;
}

pub fn covariance_matrix(a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    let mut covariance_m = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let column_i: Vec<f64> = a.iter().map(|row| row[i]).collect();
            let column_j: Vec<f64> = a.iter().map(|row| row[j]).collect();
            covariance_m[i][j] = cov(&column_i, &column_j);
        }
    }

    covariance_m
}

/// The parts of a spline fit that depend only on the series *length*: the
/// design matrix, its normal matrix, and the piecewise coefficient mixing
/// matrix. None of these look at the data, so for a batch of equal-length
/// series (which is what a UCR dataset is) they are built once and reused.
struct SplineBasis {
    /// Design matrix `A`, `n` rows of `N_SPLINE + 1` columns, row-major.
    design: Vec<f64>,
    /// Normal matrix `A^T A`, `(N_SPLINE + 1)` square, row-major.
    ata: Vec<f64>,
    /// Mixes the solved coefficients into per-piece polynomial coefficients.
    c: Vec<Vec<f64>>,
    /// Index at which the second piece starts.
    break1: usize,
}

const N_SPLINE: usize = 4;
const SPLINE_PIECES: usize = 2;

thread_local! {
    static SPLINE_BASES: RefCell<HashMap<usize, Rc<SplineBasis>>> =
        RefCell::new(HashMap::new());
}

fn spline_basis(n: usize) -> Rc<SplineBasis> {
    SPLINE_BASES.with(|cache| {
        Rc::clone(
            cache
                .borrow_mut()
                .entry(n)
                .or_insert_with(|| Rc::new(build_spline_basis(n))),
        )
    })
}

fn build_spline_basis(series_len: usize) -> SplineBasis {
    let deg = 3;
    let pieces = SPLINE_PIECES;
    let breaks = [
        0,
        (series_len as f64 / 2.0).floor() as usize - 1,
        series_len - 1,
    ];
    let h0 = [breaks[1] - breaks[0], breaks[2] - breaks[1]];

    let h_copy = [h0[0], h0[1], h0[0], h0[1]];

    let hl = [h_copy[3], h_copy[2], h_copy[1]];

    let hl_cs = hl
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<usize>>();

    let bl = hl_cs
        .iter()
        .map(|x| breaks[0] as f64 - *x as f64)
        .collect::<Vec<f64>>();

    let hr = [h_copy[0], h_copy[1], h_copy[2]];

    let hr_cs = hr
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<usize>>();

    let br = hr_cs
        .iter()
        .map(|x| breaks[2] as f64 + *x as f64)
        .collect::<Vec<f64>>();

    let mut breaks_ext = vec![0.0; 3 * deg];

    for i in 0..deg {
        breaks_ext[i] = bl[deg - i - 1] as f64;
        breaks_ext[i + deg] = breaks[i] as f64;
        breaks_ext[i + 2 * deg] = br[i] as f64;
    }

    let mut h_ext = vec![0.0; 3 * deg - 1];
    for i in 0..3 * deg - 1 {
        h_ext[i] = breaks_ext[i + 1] - breaks_ext[i];
    }

    let n_spline = 4;
    let pieces_ext = 3 * deg - 1;

    let mut coefs = vec![vec![0.0; n_spline + 1]; n_spline * pieces_ext];

    for i in (0..n_spline * pieces_ext).step_by(n_spline) {
        coefs[i][0] = 1.0;
    }

    let mut ii = vec![vec![0.0; pieces_ext]; deg + 1];

    for i in 0..pieces_ext {
        ii[0][i] = i.min(pieces_ext - 1) as f64;
        ii[1][i] = (i + 1).min(pieces_ext - 1) as f64;
        ii[2][i] = (i + 2).min(pieces_ext - 1) as f64;
        ii[3][i] = (i + 3).min(pieces_ext - 1) as f64;
    }

    let mut h = vec![0.0; (deg + 1) * pieces_ext];
    for i in 0..n_spline * pieces_ext {
        let ii_flat = ii[i % n_spline][i / n_spline] as usize;
        h[i] = h_ext[ii_flat];
    }

    let mut q = vec![vec![0.0; pieces_ext]; n_spline];

    for i in 1..n_spline {
        for j in 0..i {
            for k in 0..n_spline * pieces_ext {
                coefs[k][j] *= h[k] / (i - j) as f64;
            }
        }

        for j in 0..n_spline * pieces_ext {
            q[j % n_spline][j / n_spline] = 0.0;
            for k in 0..n_spline {
                q[j % n_spline][j / n_spline] += coefs[j][k];
            }
        }

        for j in 0..pieces_ext {
            for k in 1..n_spline {
                q[k][j] += q[k - 1][j];
            }
        }

        for j in 0..n_spline * pieces_ext {
            if j % n_spline == 0 {
                coefs[j][i] = 0.0;
            } else {
                coefs[j][i] = q[j % n_spline - 1][j / n_spline];
            }
        }

        let mut fmax = vec![0.0; pieces_ext * n_spline];
        for j in 0..pieces_ext {
            for k in 0..n_spline {
                fmax[j * n_spline + k] = q[n_spline - 1][j];
            }
        }

        for j in 0..i + 1 {
            for k in 0..n_spline * pieces_ext {
                coefs[k][j] /= fmax[k];
            }
        }

        // diff to adjacent antiderivatives
        for j in 0..(n_spline * pieces_ext) - deg {
            for k in 0..i + 1 {
                coefs[j][k] -= coefs[deg + j][k];
            }
        }
        for j in (0..n_spline * pieces_ext).step_by(n_spline) {
            coefs[j][i] = 0.0;
        }
    }

    let mut scale = vec![1.0; n_spline * pieces_ext];
    for i in 0..n_spline - 1 {
        for j in 0..n_spline * pieces_ext {
            scale[j] /= h[j];
        }
        for j in 0..n_spline * pieces_ext {
            coefs[j][(n_spline - 1) - (i + 1)] *= scale[j];
        }
    }

    let mut jj = vec![vec![0; pieces]; n_spline];
    for i in 0..n_spline {
        for j in 0..pieces {
            if i == 0 {
                jj[i][j] = n_spline * (1 + j);
            } else {
                jj[i][j] = deg;
            }
        }
    }

    for i in 1..n_spline {
        for j in 0..pieces {
            jj[i][j] += jj[i - 1][j];
        }
    }

    let mut coefs_out = vec![vec![0.0; n_spline]; n_spline * pieces];

    for i in 0..n_spline * pieces {
        let jj_flat = jj[i % n_spline][i / n_spline] - 1;
        for j in 0..n_spline {
            coefs_out[i][j] = coefs[jj_flat][j];
        }
    }

    let mut xs_b = vec![0; series_len * n_spline];
    let mut index_b = vec![0; series_len * n_spline];

    let mut break_ind = 1;

    for i in 0..series_len {
        if i >= breaks[break_ind] && break_ind < breaks.len() - 1 {
            break_ind += 1;
        }
        for j in 0..n_spline {
            xs_b[i * n_spline + j] = i - breaks[break_ind - 1];
            index_b[i * n_spline + j] = j + (break_ind - 1) * n_spline;
        }
    }

    let mut v_b = vec![0.0; series_len * n_spline];
    for i in 0..series_len * n_spline {
        v_b[i] = coefs_out[index_b[i]][0];
    }

    for i in 1..n_spline {
        for j in 0..series_len * n_spline {
            v_b[j] = v_b[j] * xs_b[j] as f64 + coefs_out[index_b[j]][i];
        }
    }

    let mut a_ = vec![0.0; series_len * (n_spline + 1)];
    let mut break_ind = 0;
    for i in 0..series_len * n_spline {
        if i / n_spline >= breaks[1] {
            break_ind = 1;
        }
        a_[(i % n_spline) + break_ind + (i / n_spline) * (n_spline + 1)] = v_b[i];
    }

    let mut c = vec![vec![0.0; n_spline * pieces]; pieces + n_spline - 1];
    for i in 0..n_spline * n_spline * pieces {
        let crow = i % n_spline + (i / n_spline) % 2;
        let ccol = i / n_spline;
        let coef_row = i % (n_spline * 2);
        let coef_col = i / (n_spline * 2);
        c[crow][ccol] = coefs_out[coef_row][coef_col];
    }

    // A^T A only involves the design matrix, so it is folded into the basis
    // too; per call that leaves just A^T b and a 5x5 solve.
    let cols = n_spline + 1;
    let mut ata = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..series_len {
                sum += a_[k * cols + i] * a_[k * cols + j];
            }
            ata[i * cols + j] = sum;
        }
    }

    SplineBasis {
        design: a_,
        ata,
        c,
        break1: breaks[1],
    }
}

/// Cubic spline fit with two pieces, as used by PD_PeriodicityWang.
pub fn splinefit(a: &[f64]) -> Vec<f64> {
    let n_spline = N_SPLINE;
    let pieces = SPLINE_PIECES;
    let cols = n_spline + 1;
    let basis = spline_basis(a.len());

    // A^T b, accumulated over k in increasing order to match the original
    // transpose-then-multiply formulation term for term.
    let mut atb = vec![0.0; cols];
    for (k, &value) in a.iter().enumerate() {
        for i in 0..cols {
            atb[i] += basis.design[k * cols + i] * value;
        }
    }

    let x = gauss_elimination(cols, &basis.ata, atb);

    let mut coefs_spline = vec![vec![0.0; n_spline]; pieces];
    for i in 0..n_spline * pieces {
        let coef_col = i / pieces;
        let coef_row = i % pieces;
        for j in 0..cols {
            coefs_spline[coef_row][coef_col] += basis.c[j][i] * x[j];
        }
    }

    let break1 = basis.break1;
    let mut y_out = vec![0.0; a.len()];
    for i in 0..a.len() {
        let second_half = if i < break1 { 0 } else { 1 };
        y_out[i] = coefs_spline[second_half][0];
    }

    for i in 1..n_spline {
        for j in 0..a.len() {
            let second_half = if j < break1 { 0 } else { 1 };
            y_out[j] = y_out[j] * (j - break1 * second_half) as f64 + coefs_spline[second_half][i];
        }
    }

    return y_out;
}

pub fn gauss_elimination(size_a2: usize, a: &[f64], b: Vec<f64>) -> Vec<f64> {
    let mut x = vec![0.0; size_a2];

    let mut a_elim = vec![vec![0.0; size_a2]; size_a2];
    let mut b_elim = vec![0.0; size_a2];

    for i in 0..size_a2 {
        for j in 0..size_a2 {
            a_elim[i][j] = a[i * size_a2 + j];
        }
        b_elim[i] = b[i];
    }

    for i in 0..size_a2 {
        for j in i + 1..size_a2 {
            let factor = a_elim[j][i] / a_elim[i][i];
            b_elim[j] = b_elim[j] - factor * b_elim[i];

            for k in i..size_a2 {
                a_elim[j][k] = a_elim[j][k] - factor * a_elim[i][k];
            }
        }
    }

    let mut b_mines_a_temp;
    for i in (0..size_a2).rev() {
        b_mines_a_temp = b_elim[i];
        for j in i + 1..size_a2 {
            b_mines_a_temp -= x[j] * a_elim[i][j];
        }
        x[i] = b_mines_a_temp / a_elim[i][i];
    }

    return x;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocovariance_matches_autocov_lag() {
        let data = vec![1.0, -2.0, 3.5, 4.0, -1.5, 2.25, 0.75, -3.0, 1.2, 0.1];
        let autocov = autocovariance(&data);
        for lag in 0..data.len() {
            let expected = autocov_lag(&data, lag);
            let actual = autocov[lag];
            assert!(
                (actual - expected).abs() < 1e-10,
                "lag {lag}: expected {expected}, got {actual}"
            );
        }
    }
}
