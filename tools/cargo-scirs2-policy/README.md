# cargo-scirs2-policy

A Cargo subcommand that enforces the COOLJAPAN Pure Rust and SciRS2 workspace
compliance policies.  It lints `Cargo.toml` files for banned dependencies,
scans source code for prohibited `use` statements and `.unwrap()` calls, checks
`#[deprecated]` attribute hygiene, audits the transitive dependency footprint,
and provides benchmark regression detection.

Part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing
library for Rust.

---

## Installation

```
cargo install cargo-scirs2-policy
```

The tool is invoked as a Cargo subcommand:

```
cargo scirs2-policy --help
```

---

## Subcommands

> **Path and format validation**: `check`, `dep-audit`, `check-semver`, `save-api-snapshot`, and `version-policy` all validate that `--workspace` exists and is a directory *before* scanning anything, failing with exit code `1` and a clear error message on a typo'd or missing path. (Previously a bad `--workspace` path was silently scanned as zero crates, which made every one of these subcommands report a false-clean pass — e.g. `check` printing "No policy violations found." — instead of failing loudly.) Likewise, `--format` on `check` and `check-semver` is a `clap`-validated enum (`text` or `json`); an unrecognized value now fails parsing immediately with a clear clap error instead of silently falling back to the text report.

> **Windows exempt-path matching (0.6.3)**: the source scanner's (`src/rules/source_scan.rs`) exempt-path matching previously compared against `Path::to_string_lossy()` directly — on Windows this renders path separators as `\`, so `/`-separated directory-segment exemption checks never matched. Paths are now normalized to `/` before comparison. Verified by code review; not exercised under Windows CI.

### `check` — full policy compliance scan

Runs all registered rules against the workspace and reports violations.

```
# Text output (default)
cargo scirs2-policy check --workspace /path/to/scirs

# Scan the current directory
cargo scirs2-policy check --workspace .

# Emit machine-readable JSON
cargo scirs2-policy check --workspace . --format json
```

Exit code is `0` when no violations are found, `1` otherwise.

### `rules` — list available rules

Prints every registered rule ID and its description.

```
cargo scirs2-policy rules
```

### `duplicates` — detect multi-version dependencies

Parses `Cargo.lock` and lists packages that appear with more than one version.
This is informational and always exits with code `0`.

```
cargo scirs2-policy duplicates --workspace .
```

### `dep-audit` — dependency footprint audit

Reports the total count of unique packages in `Cargo.lock`, the number of
direct workspace dependencies, and any banned packages present.

```
# Basic audit
cargo scirs2-policy dep-audit --workspace .

# Compare against a known-good baseline count
cargo scirs2-policy dep-audit --workspace . --baseline-count 850

# Fail the build when banned packages are present
cargo scirs2-policy dep-audit --workspace . --strict
```

### `bench-snapshot` — capture a Criterion benchmark snapshot

Walks a Criterion output directory for `estimates.json` files and serialises
the measurements to a JSON snapshot file for later comparison.

```
cargo scirs2-policy bench-snapshot \
    --criterion-dir target/criterion \
    --output /tmp/baseline.json
```

### `bench-diff` — detect benchmark regressions

Compares two snapshots and reports any benchmarks that regressed beyond a
configurable threshold.  Exits with code `1` when regressions are found.

```
# Default threshold: 10%
cargo scirs2-policy bench-diff \
    --baseline /tmp/baseline.json \
    --current  /tmp/current.json

# Custom threshold: 5%
cargo scirs2-policy bench-diff \
    --baseline /tmp/baseline.json \
    --current  /tmp/current.json \
    --threshold 0.05

# Print the full diff including improvements
cargo scirs2-policy bench-diff \
    --baseline /tmp/baseline.json \
    --current  /tmp/current.json \
    --full
```

### `check-semver` — deprecation and SemVer policy

Scans `#[deprecated]` attributes across the workspace and validates that they
carry the required `since` and `note` fields, and reports items that have
exceeded the deprecation window and are ready for removal.

```
# Deprecation-only scan
cargo scirs2-policy check-semver --workspace .

# Include API compatibility check against a saved snapshot
cargo scirs2-policy check-semver --workspace . \
    --api-snapshot /tmp/api_snapshot.json

# JSON output
cargo scirs2-policy check-semver --workspace . --format json
```

### `save-api-snapshot` — save the public API surface

Captures the current public API surface (all `pub` items found in source) to a
JSON file.  Use the snapshot later with `check-semver --api-snapshot` to detect
backward-incompatible removals.

```
cargo scirs2-policy save-api-snapshot \
    --workspace . \
    --output /tmp/api_snapshot.json
```

### `version-policy` — print the current version policy

Shows the active SemVer commitment level, deprecation window, and LTS branch
configuration derived from the workspace `Cargo.toml`.

```
cargo scirs2-policy version-policy --workspace .
```

---

## Policy rules

| Rule ID | Severity | Description |
|---------|----------|-------------|
| `BANNED_DEP_001` | ERROR | Direct dependency on a banned crate (`zip`, `flate2`, `zstd`, `bzip2`, `lz4`, `snap`, `brotli`, `miniz_oxide`, `bincode`, `openblas-src`, `blas-src`, `z3`, `ndarray-npy`) |
| `SOURCE_SCAN_001` | WARNING | `use rand::` in non-core source files — use `scirs2-core` RNG instead |
| `SOURCE_SCAN_002` | INFO | `use ndarray::` in non-core source files |
| `UNWRAP_001` | WARNING | `.unwrap()` call outside a `#[cfg(test)]` block |
| `DEPRECATION_001` | WARNING | `#[deprecated]` attribute missing `since` version |
| `DEPRECATION_002` | WARNING | `#[deprecated]` attribute missing `note` / migration guidance |
| `DEPRECATION_003` | INFO | Item deprecated 2+ minor versions ago — ready for removal |
| `DEPRECATION_004` | WARNING | `since` version is newer than the current crate version |
| `API_COMPAT_001–003` | ERROR/WARNING | Public API item removed or changed relative to snapshot |

Violations at severity `ERROR` cause a non-zero exit code.

---

## Banned dependencies and replacements

The COOLJAPAN Pure Rust Policy prohibits C/Fortran-linked or non-preferred
crates.  Use these replacements:

| Banned crate | Replacement |
|---|---|
| `zip` | `oxiarc-archive` |
| `flate2` | `oxiarc-deflate` / `oxiarc-*` |
| `zstd` | `oxiarc-zstd` |
| `bzip2` | `oxiarc-bzip2` |
| `lz4` | `oxiarc-lz4` |
| `snap` | `oxiarc-snappy` |
| `brotli` | `oxiarc-brotli` |
| `miniz_oxide` | `oxiarc-deflate` |
| `bincode` | `oxicode` |
| `openblas-src` / `blas-src` | `oxiblas` |
| `z3` | `oxiz` |
| `rustfft` | `oxifft` |

---

## Typical CI integration

```yaml
- name: Policy compliance
  run: cargo scirs2-policy check --workspace . --format json

- name: Dependency audit
  run: cargo scirs2-policy dep-audit --workspace . --strict

- name: Benchmark regression
  run: |
    cargo scirs2-policy bench-snapshot \
        --criterion-dir target/criterion \
        --output /tmp/current.json
    cargo scirs2-policy bench-diff \
        --baseline baselines/latest.json \
        --current  /tmp/current.json \
        --threshold 0.10
```

---

## License

Licensed under Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE)).

---

SciRS2 project: <https://github.com/cool-japan/scirs>
