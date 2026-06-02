//! Distribution validation test suite
//!
//! Split into three focused modules to keep each file under 2000 lines:
//!
//! - `distribution_validation_reference`: Core SciPy reference values for all distributions
//! - `distribution_validation_sanity`: Cross-distribution checks, logistic, chi2/t additional,
//!   and theoretical mean/variance verification
//! - `distribution_validation_properties`: PDF integrates-to-one, CDF monotonicity, PPF
//!   inverse, and miscellaneous additional reference values

// Each module is a standalone integration test binary under tests/
// Rust's integration test harness picks them up automatically as separate test binaries.
// This file intentionally contains no test functions — it serves as documentation only.
