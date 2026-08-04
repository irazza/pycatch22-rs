# pycatch22-rs

A Rust implementation of the [catch22](https://github.com/DynamicsAndNeuralSystems/catch22)
time-series feature set, with Python bindings.

25 features: the 22 catch22 features, the two catch24 additions (mean and
standard deviation), and the slope of a linear fit.

- **Verified against the original C** over the entire UCR archive — 128
  datasets, 191,158 series, 4.78 million feature values. See
  [`docs/parity.md`](docs/parity.md) for exactly what was compared, at what
  tolerance, and where the two still differ.
- **~6x faster than the original C**, single-threaded, on the same machine and
  data, with C built at `-O3 -march=native`. See
  [`BENCHMARKS.md`](BENCHMARKS.md).
- **Zero-copy numpy interface**, with a batch entry point that releases the GIL
  and processes series in parallel.

## Installation

Requires Python 3.11 or later.

```bash
pip install pycatch22-rs
```

From source:

```bash
git clone https://github.com/irazza/pycatch22-rs
cd pycatch22-rs
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

## Usage

```python
import numpy as np
import pycatch22_rs

x = np.random.default_rng(0).standard_normal(500)

# All 25 features at once.
values = pycatch22_rs.compute_all(x)            # -> ndarray, shape (25,)
dict(zip(pycatch22_rs.FEATURE_NAMES, values))

# Any single feature, by name.
pycatch22_rs.SP_Summaries_welch_rect_centroid(x)
pycatch22_rs.CO_f1ecac(x)

# ...or by index, in FEATURE_NAMES order.
pycatch22_rs.compute(x, 19)

# A batch of series: GIL released, rows processed in parallel.
batch = np.random.default_rng(0).standard_normal((1000, 500))
pycatch22_rs.compute_batch(batch)               # -> ndarray, shape (1000, 25)
```

### Normalisation

By default, `compute_all` and `compute_batch` reproduce the reference pipeline:
features 0–21 are computed on the **z-scored** series, and 22–24 (mean, standard
deviation, slope) on the **raw** series, which is the catch24 convention. Pass
`normalize=False` to compute everything on the series exactly as given.

The z-score uses the **sample** standard deviation (`ddof=1`), matching the
reference implementation's `zscore_norm2`. This is not cosmetic: several
features are not scale-invariant, so normalising with the population standard
deviation shifts their values.

The individually named functions take the series as given and do no
normalisation, so z-score first if you want the reference values:

```python
z = pycatch22_rs.zscore(x)
pycatch22_rs.DN_HistogramMode_5(z)
```

### Input handling

Lists, non-float64 dtypes, and non-contiguous views are all accepted and
converted once. Input is never modified.

Series shorter than 4 samples, and series containing `NaN` or infinity, raise
`ValueError`. This is a deliberate departure from C, which propagates `NaN`
through its feature functions. Note that a **constant** series z-scores to
`NaN` (its standard deviation is zero), so `compute_all(constant, normalize=True)`
raises as well.

## Rust

The kernel is a standalone crate with no Python dependency:

```toml
[dependencies]
catch22 = { path = "crates/catch22" }
```

```rust
let values = catch22::compute_all_normalized(&series)?;   // [f64; 25]
let centroid = catch22::sp_summaries_welch_rect_centroid(&z);
```

`FEATURES` and `FEATURE_NAMES` are parallel arrays over the same 25 indices.

## Development

```bash
cargo test --workspace          # Rust: parity, consistency, edge cases
maturin develop --release
pytest tests/                   # Python: API surface and parity
```

### Checking against the C reference

Parity is enforced at two levels. The committed fixture in `tests/data/` holds
C's expected output for a sample of series plus hand-picked edge cases, and is
checked by both test suites on every run. The full archive sweep is opt-in and
needs a local copy of the UCR archive:

```bash
bash tools/c_reference/fetch.sh          # clone catch22 C at the pinned commit
make -C tools/c_reference                # build the reference driver
cargo run --release -p ucr_check -- --ucr-root ~/DATA/ucr --report docs/parity-report.md
```

The same tool produces the benchmark table:

```bash
cargo run --release -p ucr_check -- --ucr-root <subset> \
    --driver tools/c_reference/driver_native --bench BENCHMARKS.md
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
