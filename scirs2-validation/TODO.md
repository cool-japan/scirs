# scirs2-validation TODO

## Status: v0.4.3 Released (2026-05-03)

## Purpose

Shared statistical validation framework providing pre-computed analytical reference values, generic validation traits, property-test helpers, and report generation. Designed to be consumed by any COOLJAPAN ecosystem crate without introducing circular dependencies.

## Completed

### Reference Distributions (15+)
- [x] Normal (0,1) and (2,3)
- [x] Exponential (rate=1)
- [x] Uniform (0,1)
- [x] Beta (2,5)
- [x] Gamma (2,1)
- [x] Chi-squared (df=4)
- [x] Student's t (df=5)
- [x] Cauchy (0,1)
- [x] Poisson (lambda=3)
- [x] Binomial (10, 0.3)
- [x] Weibull (k=2, scale=1)
- [x] Log-normal (0,1)
- [x] Laplace (0,1)
- [x] Pareto (alpha=1, scale=2)

### Validation Functions
- [x] `validate_distribution` — PDF/CDF/PPF/moment comparison vs `DistributionReference`
- [x] `validate_pdf_integral` — trapezoidal rule integration check
- [x] `validate_cdf_monotone` — non-decreasing CDF verification
- [x] `validate_ppf_roundtrip` — `cdf(ppf(p)) ≈ p` check
- [x] `validate_cdf_bounds` — tail behaviour verification
- [x] `validate_pdf_nonnegative` — non-negativity check

### Reporting
- [x] ASCII tabular report via `generate_report`
- [x] JSON report via `generate_json_report` (no serde required for default)
- [x] `ValidationReport` aggregator with pass/fail summary
- [x] Optional `serialization` feature for richer serde-based output

## v0.4.3 Quality Gate

- ~25 `#[test]` functions covering distribution validation suite
- cargo check + clippy: clean
- Pure Rust (no C/Fortran deps); core has zero non-stdlib runtime dependencies (serde optional)
- Distribution validation suite consumed by `scirs2-stability-tests` and downstream crates (78 tests across 15+ dists at workspace level)

## Notes

- Crate is `publish = false` — used internally by SciRS2 ecosystem.
- All reference values are derived analytically and verifiable by hand (no external numerical tools).
- Wave 8 distribution accuracy fixes (Beta CDF Lentz fraction, F-dist via regularized beta, ChiSquare even-df Poisson sum, Normal PPF Acklam, Pareto PDF strict boundary) are validated through this framework.
