#!/usr/bin/env python3
"""Generates the golden fixture used by the Rust and Python test suites.

Samples series from the UCR archive, adds hand-picked edge cases, runs the
pinned C reference over them, and writes the series together with C's expected
feature values. This is what lets CI check parity without needing the 836 MB
archive or a C toolchain.

Usage:
    python tools/make_fixtures.py --ucr-root ~/DATA/ucr \
        --driver tools/c_reference/driver_O2 \
        --out tests/data/golden.tsv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N_FEATURES = 25
MAX_LEN = 512
PER_DATASET = 2


def edge_cases() -> list[tuple[str, np.ndarray]]:
    """Degenerate-but-valid inputs that the archive does not cover well."""
    rng = np.random.default_rng(20260804)
    cases: list[tuple[str, np.ndarray]] = [
        ("edge:minimum_length", np.array([1.0, 2.0, 3.0, 4.0])),
        ("edge:length_five", np.array([1.0, -2.0, 3.5, 4.0, -1.5])),
        ("edge:ramp", np.arange(64, dtype=float)),
        ("edge:descending_ramp", np.arange(64, 0, -1, dtype=float)),
        ("edge:two_valued", np.tile([0.0, 1.0], 64)),
        ("edge:mostly_zero", np.concatenate([np.zeros(120), [5.0, -5.0, 3.0, 0.0]])),
        ("edge:all_negative", -np.abs(rng.standard_normal(128)) - 1.0),
        ("edge:tiny_amplitude", rng.standard_normal(128) * 1e-8),
        ("edge:huge_amplitude", rng.standard_normal(128) * 1e8),
        ("edge:sine", np.sin(np.arange(256) * 0.1)),
        ("edge:sawtooth", np.arange(256, dtype=float) % 17),
        ("edge:single_spike", np.concatenate([np.zeros(99), [100.0], np.zeros(100)])),
        ("edge:step", np.concatenate([np.zeros(64), np.ones(64)])),
        ("edge:white_noise", rng.standard_normal(256)),
        ("edge:random_walk", np.cumsum(rng.standard_normal(256))),
    ]
    return cases


def sample_ucr(root: Path, per_dataset: int, stride: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    datasets = sorted(p for p in root.iterdir() if p.is_dir())
    for index, dataset in enumerate(datasets):
        # Take a spread of datasets rather than all 128, to keep the fixture
        # small; the full sweep (tools/ucr_check) covers the rest.
        if index % stride != 0:
            continue
        tsv = dataset / f"{dataset.name}_TRAIN.tsv"
        if not tsv.exists():
            continue
        with tsv.open() as handle:
            for row_index, line in enumerate(handle):
                if row_index >= per_dataset:
                    break
                values = np.array([float(v) for v in line.split()[1:]], dtype=np.float64)
                values = values[:MAX_LEN]
                if len(values) < 4 or not np.isfinite(values).all():
                    continue
                if np.std(values, ddof=1) == 0.0:
                    continue
                out.append((f"{dataset.name}:{row_index}", values))
    return out


def c_reference(driver: Path, series: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Runs the C driver over the sample and returns an (n, 25) array."""
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp) / "input.tsv"
        binary = Path(tmp) / "out.bin"
        with tsv.open("w") as handle:
            for _, values in series:
                # Column 0 is the class label, which the driver discards.
                handle.write("0\t" + "\t".join(repr(float(v)) for v in values) + "\n")

        result = subprocess.run(
            [str(driver), str(tsv), str(binary)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"C driver failed: {result.stderr}")

        expected = np.fromfile(binary, dtype="<f8").reshape(-1, N_FEATURES)

    if len(expected) != len(series):
        raise SystemExit(f"C produced {len(expected)} rows for {len(series)} series")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucr-root", type=Path, default=Path.home() / "DATA/ucr")
    parser.add_argument("--driver", type=Path, default=Path("tools/c_reference/driver_O2"))
    parser.add_argument("--out", type=Path, default=Path("tests/data/golden.tsv"))
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    if not args.driver.exists():
        raise SystemExit(
            f"{args.driver} not found; run tools/c_reference/fetch.sh && make -C tools/c_reference"
        )

    series = edge_cases()
    if args.ucr_root.exists():
        series += sample_ucr(args.ucr_root, PER_DATASET, args.stride)
    else:
        print(f"warning: {args.ucr_root} not found, writing edge cases only", file=sys.stderr)

    expected = c_reference(args.driver, series)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write(
            "# golden fixture: name, n, 25 expected feature values from the C "
            "reference, then the series\n"
            "# features 0..21 are computed on the z-scored series, 22..24 on the raw one\n"
        )
        for (name, values), row in zip(series, expected):
            fields = [name, str(len(values))]
            fields += [f"{v:.17g}" for v in row]
            fields += [f"{v:.17g}" for v in values]
            handle.write("\t".join(fields) + "\n")

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {len(series)} series to {args.out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
