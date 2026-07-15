# scirs2-validation

[![Stable](https://img.shields.io/badge/status-stable-brightgreen)]()
[![crates.io](https://img.shields.io/crates/v/scirs2-validation.svg)](https://crates.io/crates/scirs2-validation)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-validation)](https://docs.rs/scirs2-validation)

Shared statistical validation framework for the [SciRS2](https://github.com/cool-japan/scirs) ecosystem.

`publish = false` — this is an internal, workspace-only crate (v0.6.1). It ships a complete, fully-tested (25 `#[test]`s + 1 doctest, all passing; 0 stubs) reference-value framework, but as of this release it is not yet wired in as a dependency by any other workspace crate — see [TODO.md](TODO.md) for details.

## Overview

`scirs2-validation` provides a dependency-light, standalone framework for validating statistical distribution implementations against mathematically-derived reference values. It is designed to be used by any library in the COOLJAPAN ecosystem — `scirs2-stats`, `NumRS2`, `QuantRS2`, and others — without introducing circular dependencies.

The crate ships with:

- **Pre-computed reference values** for 15+ standard probability distributions derived from exact analytical formulas.
- **Generic validation traits** that any distribution type can implement.
- **Property-test helpers** that verify fundamental mathematical properties (PDF integrates to 1, CDF is monotone, PPF is the inverse of CDF, etc.).
- **Report generation** in human-readable tabular format and JSON.

## Features

### Reference Values (`reference_values`)

Pre-computed `(x, f(x))` pairs for PDF, CDF, and PPF, plus exact analytical moments for 15 built-in parameterizations:

| Distribution | Parameters |
|---|---|
| Normal | (0,1) and (2,3) |
| Exponential | rate=1 |
| Uniform | (0,1) |
| Beta | (2,5) |
| Gamma | (2,1) |
| Chi-squared | df=4 |
| Student's t | df=5 |
| Cauchy | (0,1) |
| Poisson | lambda=3 |
| Binomial | (10, 0.3) |
| Weibull | (k=2, scale=1) |
| Log-normal | (0,1) |
| Laplace | (0,1) |
| Pareto | (alpha=1, scale=2) |

All values are verifiable from the distribution's closed-form definition — no external numerical tools were used.

### Validation Trait (`validators`)

Implement `ValidatableDistribution` for your distribution type to gain access to all validation functions:

```rust
pub trait ValidatableDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn ppf(&self, p: f64) -> Option<f64> { None }
}
```

### Validation Functions

| Function | Description |
|---|---|
| `validate_distribution` | Compare PDF/CDF/PPF/moments against a `DistributionReference` |
| `validate_pdf_integral` | Numerical integration check: PDF integrates to 1 via trapezoidal rule |
| `validate_cdf_monotone` | Verify CDF is non-decreasing at a set of x-values |
| `validate_ppf_roundtrip` | Round-trip check: `cdf(ppf(p)) ≈ p` for all test probabilities |
| `validate_cdf_bounds` | CDF approaches 0 in the lower tail and 1 in the upper tail |
| `validate_pdf_nonnegative` | PDF is non-negative at all evaluation points |

### Report Generation (`report`)

- `generate_report` — ASCII table showing per-distribution PASS/FAIL status, error counts, and moment errors, plus a detail section for any failures.
- `generate_json_report` — JSON output (no serde required; NaN/Inf serialized as `null`).
- `ValidationReport` — Struct collecting multiple `ValidationResult`s with summary counts (`passed`, `failed`, `total`) and a `Display` impl.

### Optional Feature: `serialization`

Enable `serde` / `serde_json` support for richer serialization workflows:

```toml
[dependencies]
scirs2-validation = { version = "0.6.1", features = ["serialization"] }
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-validation = "0.6.1"
```

### Validating a Distribution Implementation

```rust
use scirs2_validation::validators::{ValidatableDistribution, validate_distribution};
use scirs2_validation::reference_values::NORMAL_STANDARD;

struct MyNormal {
    mu: f64,
    sigma: f64,
}

impl ValidatableDistribution for MyNormal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * std::f64::consts::PI).sqrt())
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / (self.sigma * std::f64::consts::SQRT_2);
        0.5 * (1.0 + libm::erf(z))
    }

    fn mean(&self) -> f64 { self.mu }
    fn variance(&self) -> f64 { self.sigma * self.sigma }
}

fn main() {
    let dist = MyNormal { mu: 0.0, sigma: 1.0 };
    let result = validate_distribution(&dist, &NORMAL_STANDARD, 1e-12, 1e-12);

    println!("{}", result);
    assert!(result.passed, "Distribution failed validation: {}", result);
}
```

### Batch Validation with a Report

```rust
use scirs2_validation::reference_values::all_references;
use scirs2_validation::report::{generate_report, ValidationReport};
use scirs2_validation::validators::{validate_distribution, ValidatableDistribution};

// ... implement ValidatableDistribution for each distribution type ...

fn validate_all(dists: &[(&dyn ValidatableDistribution, usize)]) {
    let references = all_references();
    let results: Vec<_> = dists
        .iter()
        .zip(references.iter())
        .map(|((dist, _), reference)| {
            validate_distribution(*dist, reference, 1e-9, 1e-9)
        })
        .collect();

    let report = ValidationReport::new("My Library v1.0", results.clone());
    println!("{}", report);

    // Also print the full ASCII table
    println!("{}", generate_report(&results));
}
```

### Property Tests

```rust
use scirs2_validation::validators::{
    validate_pdf_integral, validate_cdf_monotone,
    validate_ppf_roundtrip, validate_cdf_bounds, validate_pdf_nonnegative,
};

fn check_properties(dist: &impl ValidatableDistribution) {
    // PDF must integrate to 1 over (-10, 10) with 10000 quadrature points
    assert!(validate_pdf_integral(dist, -10.0, 10.0, 10000, 1e-4));

    // CDF must be non-decreasing
    let xs: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.1).collect();
    assert!(validate_cdf_monotone(dist, &xs));

    // PDF must be non-negative
    assert!(validate_pdf_nonnegative(dist, &xs));

    // Tail behaviour
    assert!(validate_cdf_bounds(dist, -50.0, 50.0, 1e-6));

    // PPF round-trip
    let probs = vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    let cdf_fn = |x: f64| dist.cdf(x);
    let ppf_fn = |p: f64| dist.ppf(p);
    assert!(validate_ppf_roundtrip(&cdf_fn, &ppf_fn, &probs, 1e-10));
}
```

## Design Goals

- **Zero circular dependencies**: no dependency on `scirs2-stats` or any other SciRS2 crate.
- **No external numerical tools**: all reference values are derived analytically and can be verified by hand.
- **Minimal dependencies**: only `serde` / `serde_json` as optional, behind the `serialization` feature gate.
- **Pure Rust**: fully compatible with the COOLJAPAN Pure Rust policy; no C or Fortran dependencies.

## Related Crates

- [`scirs2-stats`](../scirs2-stats) - Statistical distributions and hypothesis tests
- [`scirs2-metrics`](../scirs2-metrics) - Distance metrics and evaluation measures
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
