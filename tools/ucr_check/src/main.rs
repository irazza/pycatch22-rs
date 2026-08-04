//! Compares the Rust catch22 kernel against the original C implementation over
//! the UCR archive.
//!
//! For every `.tsv` under `--ucr-root` this runs the pinned C driver
//! (`tools/c_reference/driver_O2`), computes the same 25 features in Rust, and
//! diffs them. Features that are integer-valued by construction must match
//! exactly; the rest must agree to a relative tolerance.
//!
//! The UCR archive itself is never committed — point `--ucr-root` at your local
//! copy (e.g. `~/DATA/ucr`).
// Feature indices are identities here, not cursors: `FEATURES[i]` and
// `FEATURE_NAMES[i]` must line up, so the loops index deliberately.
#![allow(clippy::needless_range_loop)]

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const N_FEATURES: usize = catch22::N_CATCH22;

/// Features whose value is an integer (a count, an index, or a ratio of small
/// integers) and must therefore agree with C bit-for-bit.
const EXACT_FEATURES: &[usize] = &[6, 11, 13, 14, 16, 17, 21];

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-12;

const FEATURE_NAMES: [&str; N_FEATURES] = [
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

struct Config {
    ucr_root: PathBuf,
    driver: PathBuf,
    work_dir: PathBuf,
    report: Option<PathBuf>,
    jobs: usize,
    per_feature: bool,
    limit: Option<usize>,
    bench: Option<PathBuf>,
}

#[derive(Default, Clone)]
struct FeatureStats {
    compared: u64,
    mismatches: u64,
    worst_rel: f64,
    worst_example: Option<String>,
    nan_only_rust: u64,
    nan_only_c: u64,
}

#[derive(Default)]
struct Summary {
    per_feature: Vec<FeatureStats>,
    rows: u64,
    files: u64,
    skipped_rows: u64,
    /// Rows whose raw series is constant. Z-scoring divides by a zero standard
    /// deviation, so C feeds NaN into every feature and gets NaN back, while
    /// the Rust API rejects non-finite input up front. Counted and audited
    /// separately instead of being scored as a mismatch.
    degenerate_rows: u64,
    degenerate_unexpected: u64,
}

impl Summary {
    fn new() -> Self {
        Summary {
            per_feature: vec![FeatureStats::default(); N_FEATURES],
            ..Default::default()
        }
    }

    fn merge(&mut self, other: Summary) {
        self.rows += other.rows;
        self.files += other.files;
        self.skipped_rows += other.skipped_rows;
        self.degenerate_rows += other.degenerate_rows;
        self.degenerate_unexpected += other.degenerate_unexpected;
        for (dst, src) in self.per_feature.iter_mut().zip(other.per_feature) {
            dst.compared += src.compared;
            dst.mismatches += src.mismatches;
            dst.nan_only_rust += src.nan_only_rust;
            dst.nan_only_c += src.nan_only_c;
            if src.worst_rel > dst.worst_rel {
                dst.worst_rel = src.worst_rel;
                dst.worst_example = src.worst_example;
            }
        }
    }
}

fn parse_args() -> Config {
    // The archive is not in the repository; point at a local copy with
    // --ucr-root, or set UCR_ROOT.
    let mut ucr_root = match std::env::var_os("UCR_ROOT") {
        Some(root) => PathBuf::from(root),
        None => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join("DATA/ucr"),
            None => PathBuf::from("DATA/ucr"),
        },
    };
    let mut driver = PathBuf::from("tools/c_reference/driver_O2");
    let mut work_dir = std::env::temp_dir().join("ucr_check");
    let mut report = None;
    let mut jobs = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut per_feature = false;
    let mut limit = None;
    let mut bench = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ucr-root" => {
                ucr_root = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--driver" => {
                driver = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--work-dir" => {
                work_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--report" => {
                report = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--jobs" => {
                jobs = args[i + 1].parse().expect("--jobs takes a number");
                i += 1;
            }
            "--limit" => {
                limit = Some(args[i + 1].parse().expect("--limit takes a number"));
                i += 1;
            }
            "--bench" => {
                bench = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--per-feature" => per_feature = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }

    Config {
        ucr_root,
        driver,
        work_dir,
        report,
        jobs,
        per_feature,
        limit,
        bench,
    }
}

/// Parses a UCR row: column 0 is the class label, the rest is the series.
fn parse_row(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .skip(1)
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn compute_rust(raw: &[f64], z: &[f64], per_feature: bool) -> [f64; N_FEATURES] {
    if !per_feature {
        // The production path: z-scores internally, then computes 0..=21 on the
        // z-scored series and 22..=24 on the raw one.
        return catch22::compute_all_normalized(raw).expect("validated above");
    }

    // Same pipeline, but routed through the individually callable entry points
    // so `--per-feature` exercises them rather than the shared-autocorr path.
    let mut out = [0.0; N_FEATURES];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i < catch22::N_NORMALIZED {
            catch22::FEATURES[i](z)
        } else {
            catch22::FEATURES[i](raw)
        };
    }
    out
}

fn compare(rust: f64, c: f64, exact: bool) -> Option<f64> {
    if rust.is_nan() && c.is_nan() {
        return None;
    }
    if rust == c {
        return None;
    }
    if rust.is_nan() != c.is_nan() {
        return Some(f64::INFINITY);
    }

    let diff = (rust - c).abs();
    if exact {
        return Some(if c == 0.0 { diff } else { diff / c.abs() });
    }
    if diff <= ATOL + RTOL * c.abs() {
        return None;
    }
    Some(if c.abs() > 0.0 {
        diff / c.abs()
    } else {
        f64::INFINITY
    })
}

fn check_file(path: &Path, cfg: &Config, tag: &str) -> Result<Summary, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;

    let out_bin = cfg.work_dir.join(format!("{tag}.bin"));
    let status = Command::new(&cfg.driver)
        .arg(path)
        .arg(&out_bin)
        .output()
        .map_err(|e| format!("running C driver: {e}"))?;
    if !status.status.success() {
        return Err(format!(
            "C driver failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&status.stderr)
        ));
    }

    let bytes = fs::read(&out_bin).map_err(|e| format!("reading {}: {e}", out_bin.display()))?;
    let _ = fs::remove_file(&out_bin);

    let mut c_values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        c_values.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }

    let mut summary = Summary::new();
    summary.files = 1;

    let mut c_row = 0usize;
    for (row_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw = parse_row(line);
        // The C driver skips rows shorter than 4 samples; mirror that so rows
        // stay aligned between the two sides.
        if raw.len() < 4 {
            summary.skipped_rows += 1;
            continue;
        }

        let offset = c_row * N_FEATURES;
        if offset + N_FEATURES > c_values.len() {
            return Err(format!(
                "{}: C produced {} rows, Rust reached row {}",
                path.display(),
                c_values.len() / N_FEATURES,
                c_row
            ));
        }
        let c_row_values = &c_values[offset..offset + N_FEATURES];
        c_row += 1;

        // A constant series has zero standard deviation, so z-scoring produces
        // NaN. C propagates that into NaN (or its constant-input early return)
        // for every feature, while the Rust API refuses non-finite input up
        // front. Audit that C really did degenerate, then move on — comparing
        // an error against a NaN is not meaningful.
        let z = catch22::zscore(&raw);
        if !z.iter().all(|v| v.is_finite()) {
            summary.degenerate_rows += 1;
            if c_row_values[..22].iter().any(|v| !v.is_nan() && *v != 0.0) {
                summary.degenerate_unexpected += 1;
            }
            continue;
        }

        let rust_values = compute_rust(&raw, &z, cfg.per_feature);
        summary.rows += 1;

        for feature in 0..N_FEATURES {
            let exact = EXACT_FEATURES.contains(&feature);
            let stats = &mut summary.per_feature[feature];
            stats.compared += 1;

            let rust = rust_values[feature];
            let c = c_row_values[feature];
            if rust.is_nan() && !c.is_nan() {
                stats.nan_only_rust += 1;
            }
            if c.is_nan() && !rust.is_nan() {
                stats.nan_only_c += 1;
            }

            if let Some(rel) = compare(rust, c, exact) {
                stats.mismatches += 1;
                if rel > stats.worst_rel || stats.worst_example.is_none() {
                    stats.worst_rel = rel;
                    stats.worst_example = Some(format!(
                        "{}:row{row_index} rust={rust:.17e} c={c:.17e}",
                        path.file_name().unwrap().to_string_lossy()
                    ));
                }
            }
        }
    }

    Ok(summary)
}

// ---------------------------------------------------------------------------
// Benchmark mode
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Timings {
    /// Nanoseconds spent on each feature, summed over every series.
    feature_ns: Vec<f64>,
    /// Nanoseconds for the whole pipeline (z-score + all 25 features).
    pipeline_ns: f64,
    rows: u64,
}

impl Timings {
    fn new() -> Self {
        Timings {
            feature_ns: vec![0.0; N_FEATURES],
            ..Default::default()
        }
    }

    fn add(&mut self, other: &Timings) {
        for (dst, src) in self.feature_ns.iter_mut().zip(&other.feature_ns) {
            *dst += src;
        }
        self.pipeline_ns += other.pipeline_ns;
        self.rows += other.rows;
    }
}

/// Times the Rust side the same way the C driver does: everything is loaded and
/// z-scored up front, then each feature is timed across the whole file in one
/// go, so a single clock read never straddles a sub-microsecond call.
fn bench_rust(series: &[Vec<f64>]) -> Timings {
    let mut timings = Timings::new();
    timings.rows = series.len() as u64;

    let zscored: Vec<Vec<f64>> = series.iter().map(|row| catch22::zscore(row)).collect();

    for feature in 0..N_FEATURES {
        let inputs = if feature < catch22::N_NORMALIZED {
            &zscored
        } else {
            series
        };
        let start = std::time::Instant::now();
        for row in inputs {
            std::hint::black_box(catch22::FEATURES[feature](std::hint::black_box(row)));
        }
        timings.feature_ns[feature] = start.elapsed().as_nanos() as f64;
    }

    let start = std::time::Instant::now();
    for row in series {
        std::hint::black_box(catch22::compute_all_normalized(std::hint::black_box(row)).unwrap());
    }
    timings.pipeline_ns = start.elapsed().as_nanos() as f64;

    timings
}

fn bench_c(path: &Path, cfg: &Config, tag: &str) -> Result<Timings, String> {
    let csv_path = cfg.work_dir.join(format!("{tag}-timing.csv"));
    let output = Command::new(&cfg.driver)
        .arg(path)
        .arg("--bench")
        .arg(&csv_path)
        .output()
        .map_err(|e| format!("running C driver: {e}"))?;
    if !output.status.success() {
        return Err(format!("C driver failed on {}", path.display()));
    }

    let text = fs::read_to_string(&csv_path).map_err(|e| format!("{e}"))?;
    let _ = fs::remove_file(&csv_path);

    let mut timings = Timings::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let index: i64 = fields[0].parse().map_err(|_| "bad index".to_string())?;
        let ns: f64 = fields[2].parse().map_err(|_| "bad ns".to_string())?;
        let rows: u64 = fields[3].parse().map_err(|_| "bad rows".to_string())?;
        timings.rows = rows;
        if index >= 0 {
            timings.feature_ns[index as usize] = ns;
        } else {
            timings.pipeline_ns = ns;
        }
    }

    Ok(timings)
}

fn run_bench(files: &[PathBuf], cfg: &Config) -> String {
    let mut rust_total = Timings::new();
    let mut c_total = Timings::new();
    let mut per_file = Vec::new();

    for (index, path) in files.iter().enumerate() {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("skipping {}: {err}", path.display());
                continue;
            }
        };
        let series: Vec<Vec<f64>> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(parse_row)
            .filter(|row| row.len() >= 4 && catch22::zscore(row).iter().all(|v| v.is_finite()))
            .collect();
        if series.is_empty() {
            continue;
        }

        let length = series[0].len();
        eprintln!(
            "[{}/{}] {} ({} series of length {})",
            index + 1,
            files.len(),
            path.file_name().unwrap().to_string_lossy(),
            series.len(),
            length
        );

        let rust = bench_rust(&series);
        let c = match bench_c(path, cfg, "bench") {
            Ok(c) => c,
            Err(err) => {
                eprintln!("  C bench failed: {err}");
                continue;
            }
        };

        per_file.push((
            path.file_name().unwrap().to_string_lossy().into_owned(),
            length,
            series.len(),
            rust.pipeline_ns / series.len() as f64,
            c.pipeline_ns / series.len() as f64,
        ));

        rust_total.add(&rust);
        c_total.add(&c);
    }

    let mut out = String::new();
    let _ = writeln!(out, "# Rust vs C: single-threaded benchmark\n");
    let _ = writeln!(
        out,
        "Same machine, same series, same order. The C reference is the pinned upstream\n\
         implementation built with `{}`. Every feature is timed across a whole file in\n\
         one measurement, so no clock read straddles a sub-microsecond call.\n\n\
         Series benchmarked: {} across {} files.\n",
        cfg.driver.file_name().unwrap().to_string_lossy(),
        rust_total.rows,
        per_file.len(),
    );

    let rust_pipeline = rust_total.pipeline_ns / rust_total.rows as f64;
    let c_pipeline = c_total.pipeline_ns / c_total.rows as f64;
    let _ = writeln!(
        out,
        "## Whole pipeline (z-score + all 25 features)\n\n\
         | | µs / series | speedup |\n|---|---|---|\n\
         | C | {:.1} | 1.00x |\n| Rust | {:.1} | **{:.2}x** |\n",
        c_pipeline / 1000.0,
        rust_pipeline / 1000.0,
        c_pipeline / rust_pipeline,
    );

    let _ = writeln!(out, "## Per feature\n");
    let _ = writeln!(
        out,
        "| # | feature | C µs/series | Rust µs/series | speedup |"
    );
    let _ = writeln!(
        out,
        "|---|---------|------------|----------------|---------|"
    );
    for feature in 0..N_FEATURES {
        let c_us = c_total.feature_ns[feature] / c_total.rows as f64 / 1000.0;
        let rust_us = rust_total.feature_ns[feature] / rust_total.rows as f64 / 1000.0;
        let speedup = if rust_us > 0.0 { c_us / rust_us } else { 0.0 };
        let _ = writeln!(
            out,
            "| {feature} | {} | {c_us:.3} | {rust_us:.3} | {speedup:.2}x |",
            FEATURE_NAMES[feature]
        );
    }

    let _ = writeln!(out, "\n## By dataset (whole pipeline)\n");
    let _ = writeln!(
        out,
        "| dataset | length | series | C µs/series | Rust µs/series | speedup |"
    );
    let _ = writeln!(
        out,
        "|---------|--------|--------|-------------|----------------|---------|"
    );
    for (name, length, count, rust_ns, c_ns) in &per_file {
        let _ = writeln!(
            out,
            "| {name} | {length} | {count} | {:.1} | {:.1} | {:.2}x |",
            c_ns / 1000.0,
            rust_ns / 1000.0,
            c_ns / rust_ns
        );
    }

    out
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "tsv") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn render_report(summary: &Summary, cfg: &Config) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Rust vs C parity over the UCR archive\n");
    let _ = writeln!(
        out,
        "- root: `{}`\n- files: {}\n- series compared: {}\n- rows skipped (<4 samples): {}\n\
         - degenerate (constant) rows, audited separately: {} (unexpected C output on {})\n- mode: {}\n\
         - tolerance: exact for {:?}, otherwise |rust-c| <= {ATOL:e} + {RTOL:e}*|c|\n",
        cfg.ucr_root.display(),
        summary.files,
        summary.rows,
        summary.skipped_rows,
        summary.degenerate_rows,
        summary.degenerate_unexpected,
        if cfg.per_feature {
            "per-feature entry points"
        } else {
            "compute_all"
        },
        EXACT_FEATURES,
    );

    let failing: Vec<usize> = (0..N_FEATURES)
        .filter(|&f| summary.per_feature[f].mismatches > 0)
        .collect();

    let _ = writeln!(
        out,
        "**{} / {N_FEATURES} features match.**\n",
        N_FEATURES - failing.len()
    );

    let _ = writeln!(
        out,
        "| # | feature | mismatches | % | worst rel err | example |"
    );
    let _ = writeln!(
        out,
        "|---|---------|-----------|---|---------------|---------|"
    );
    for feature in 0..N_FEATURES {
        let stats = &summary.per_feature[feature];
        let pct = if stats.compared > 0 {
            100.0 * stats.mismatches as f64 / stats.compared as f64
        } else {
            0.0
        };
        let _ = writeln!(
            out,
            "| {feature} | {} | {} | {pct:.2}% | {:.3e} | {} |",
            FEATURE_NAMES[feature],
            stats.mismatches,
            stats.worst_rel,
            stats.worst_example.as_deref().unwrap_or("-"),
        );
    }

    out
}

fn main() {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.work_dir).expect("creating work dir");

    if !cfg.driver.exists() {
        eprintln!(
            "C driver not found at {}. Run:\n  bash tools/c_reference/fetch.sh && make -C tools/c_reference",
            cfg.driver.display()
        );
        std::process::exit(2);
    }

    let mut files = collect_files(&cfg.ucr_root);
    if let Some(limit) = cfg.limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        eprintln!("no .tsv files under {}", cfg.ucr_root.display());
        std::process::exit(2);
    }
    if let Some(bench_path) = &cfg.bench {
        let report = run_bench(&files, &cfg);
        println!("{report}");
        fs::write(bench_path, &report).expect("writing benchmark report");
        eprintln!("benchmark written to {}", bench_path.display());
        return;
    }

    eprintln!(
        "comparing {} files with {} job(s)",
        files.len(),
        cfg.jobs.max(1)
    );

    let next = AtomicUsize::new(0);
    let total = Mutex::new(Summary::new());
    let errors = Mutex::new(Vec::<String>::new());
    let done = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for worker in 0..cfg.jobs.max(1) {
            let next = &next;
            let total = &total;
            let errors = &errors;
            let done = &done;
            let files = &files;
            let cfg = &cfg;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= files.len() {
                        break;
                    }
                    let path = &files[index];
                    match check_file(path, cfg, &format!("w{worker}")) {
                        Ok(summary) => total.lock().unwrap().merge(summary),
                        Err(err) => errors.lock().unwrap().push(err),
                    }
                    let seen = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if seen.is_multiple_of(16) || seen == files.len() {
                        eprint!("\r{seen}/{} files", files.len());
                        let _ = std::io::stderr().flush();
                    }
                }
            });
        }
    });
    eprintln!();

    let errors = errors.into_inner().unwrap();
    for err in &errors {
        eprintln!("error: {err}");
    }

    let summary = total.into_inner().unwrap();
    let report = render_report(&summary, &cfg);
    println!("{report}");

    if let Some(path) = &cfg.report {
        fs::write(path, &report).expect("writing report");
        eprintln!("report written to {}", path.display());
    }

    let failing = (0..N_FEATURES).any(|f| summary.per_feature[f].mismatches > 0);
    if failing || !errors.is_empty() {
        std::process::exit(1);
    }
}
