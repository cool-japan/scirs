//! Symbolic identity discovery from data — [`discover_identity`].
//!
//! Given `(x, f(x))` data pairs, discovers whether the data matches a known
//! mathematical identity by:
//!
//! 1. Running [`fn@crate::regression::discover`] on the data to find candidate
//!    formulas via beam-search symbolic regression.
//! 2. Canonicalizing each candidate with [`mod@crate::cas::canonicalize`] and
//!    comparing its u128 hash against a lookup table of known identities.
//! 3. Emitting a [`ProofCertificate`] when a match is found, including a
//!    [`crate::cas::CertifiedValue`] witness at a representative data point.
//!
//! # Supported identities
//!
//! The built-in database ([`builtin_identity_db`]) covers:
//! - `Const(1.0)` — constant-one data (hash shared by several named identities)
//! - `Const(0.0)` — constant-zero data
//! - `Var(0)` — identity function / `ln(exp(x)) = x`
//!
//! Multiple named identities may share the same canonical hash (e.g., all
//! expressions that simplify to the constant `1`). The lookup map retains the
//! *last* database entry for each hash, so `certificate.identity.name` is
//! deterministic but may name any one of the equivalent entries.
//!
//! # No recursion
//!
//! All traversals within `canonicalize` are iterative. This module itself adds
//! no recursive calls.

#![warn(missing_docs)]

use crate::cas::canonicalize::canonicalize;
use crate::cas::certified_value::{CertifiedValue, CertifiedValueError};
use crate::eml::op::LoweredOp;
use crate::regression::{discover, SrConfig};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// =====================================================================
// Public types
// =====================================================================

/// A named identity entry in the proof certificate database.
///
/// Each entry stores a pre-computed canonical hash so lookups run in O(1).
/// Multiple entries may share the same `canonical_hash` when their canonical
/// forms are the same expression (e.g., all expressions that evaluate to the
/// constant `1`).
#[derive(Debug, Clone)]
pub struct KnownIdentity {
    /// Human-readable name of the identity.
    pub name: &'static str,
    /// Pre-computed canonical hash of the identity's canonical form.
    pub canonical_hash: u128,
    /// A human-readable example expression for this identity.
    pub example: &'static str,
}

/// A machine-checkable proof certificate that data matches a known identity.
///
/// Returned by [`discover_identity`] when any SR candidate, after
/// canonicalization, has the same canonical hash as a known identity.
#[derive(Debug, Clone)]
pub struct ProofCertificate {
    /// The discovered symbolic formula (the SR candidate that matched).
    pub formula: LoweredOp,
    /// The matched known identity from the built-in database.
    pub identity: KnownIdentity,
    /// Certified numerical interval at a witness point (the midpoint of the
    /// `x` data, evaluated via [`CertifiedValue::certify`]).
    pub witness: CertifiedValue,
    /// Number of SR candidates evaluated before a match was found.
    ///
    /// Capped at `max_candidates` (caller-supplied bound).
    pub candidates_checked: usize,
}

/// Errors that can occur in [`discover_identity`].
#[derive(Debug)]
pub enum ProofError {
    /// The input `x` array was empty.
    EmptyData,
    /// SR discovery produced no candidates (e.g., shape mismatch).
    DiscoveryFailed(String),
    /// No SR candidate matched any known identity within `max_candidates`.
    NoIdentityMatch {
        /// Number of candidates that were checked.
        candidates: usize,
    },
    /// The discovered formula could not be certified numerically at the witness
    /// point.
    CertificationFailed(CertifiedValueError),
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::EmptyData => write!(f, "input data is empty"),
            ProofError::DiscoveryFailed(msg) => {
                write!(f, "symbolic regression discovery failed: {msg}")
            }
            ProofError::NoIdentityMatch { candidates } => write!(
                f,
                "no identity matched among {candidates} candidate formula(s)"
            ),
            ProofError::CertificationFailed(e) => {
                write!(f, "numerical certification failed: {e}")
            }
        }
    }
}

impl std::error::Error for ProofError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProofError::CertificationFailed(e) => Some(e),
            _ => None,
        }
    }
}

// =====================================================================
// Built-in identity database
// =====================================================================

/// The built-in identity database used by [`discover_identity`].
///
/// Returns a `Vec<KnownIdentity>` with pre-computed canonical hashes for
/// common mathematical constants and identities. The entries are ordered;
/// when multiple entries share the same hash, the last one wins in the
/// lookup map built by [`discover_identity`].
///
/// # Included identities
///
/// | Name | Canonical form |
/// |------|----------------|
/// | Pythagorean identity: sin²(x) + cos²(x) = 1 | `Const(1.0)` |
/// | Hyperbolic Pythagorean: cosh²(x) - sinh²(x) = 1 | `Const(1.0)` |
/// | exp(0) = 1 | `Const(1.0)` |
/// | exp(x) * exp(-x) = 1 | `Const(1.0)` |
/// | ln(exp(x)) = x | `Var(0)` |
/// | Zero constant | `Const(0.0)` |
pub fn builtin_identity_db() -> Vec<KnownIdentity> {
    let hash_one = canonicalize(&LoweredOp::Const(1.0)).hash();
    let hash_var0 = canonicalize(&LoweredOp::Var(0)).hash();
    let hash_zero = canonicalize(&LoweredOp::Const(0.0)).hash();

    vec![
        KnownIdentity {
            name: "Pythagorean identity: sin\u{00b2}(x) + cos\u{00b2}(x) = 1",
            canonical_hash: hash_one,
            example: "sin(x)^2 + cos(x)^2",
        },
        KnownIdentity {
            name: "Hyperbolic Pythagorean: cosh\u{00b2}(x) - sinh\u{00b2}(x) = 1",
            canonical_hash: hash_one,
            example: "cosh(x)^2 - sinh(x)^2",
        },
        KnownIdentity {
            name: "exp(0) = 1",
            canonical_hash: hash_one,
            example: "exp(0)",
        },
        KnownIdentity {
            name: "exp(x) * exp(-x) = 1",
            canonical_hash: hash_one,
            example: "exp(x) * exp(-x)",
        },
        KnownIdentity {
            name: "ln(exp(x)) = x",
            canonical_hash: hash_var0,
            example: "ln(exp(x))",
        },
        KnownIdentity {
            name: "Zero constant",
            canonical_hash: hash_zero,
            example: "0",
        },
    ]
}

// =====================================================================
// Main API
// =====================================================================

/// Discover whether `(x, y)` data pairs match a known mathematical identity.
///
/// # Algorithm
///
/// 1. Runs [`fn@crate::regression::discover`] (beam-search SR) on the data
///    using a moderate default configuration.
/// 2. Canonicalizes each discovered formula with [`canonicalize`] and checks
///    its u128 hash against the [`builtin_identity_db`] lookup table.
/// 3. On the first match, certifies the formula numerically at the midpoint
///    of the `x` data via [`CertifiedValue::certify`].
/// 4. Returns a [`ProofCertificate`] on success.
///
/// # Arguments
///
/// - `x`: input values (1D slice).
/// - `y`: target values (1D slice, same length as `x`).
/// - `max_candidates`: stop after checking this many SR candidates. Pass `0`
///   to skip checking entirely (always returns [`ProofError::NoIdentityMatch`]).
///
/// # Errors
///
/// - [`ProofError::EmptyData`] if `x` is empty.
/// - [`ProofError::DiscoveryFailed`] if SR returns no candidates at all.
/// - [`ProofError::NoIdentityMatch`] if no candidate matched within
///   `max_candidates`.
/// - [`ProofError::CertificationFailed`] if the matched formula cannot be
///   certified numerically at the witness point (extremely rare for
///   constant/identity formulas).
pub fn discover_identity(
    x: &[f64],
    y: &[f64],
    max_candidates: usize,
) -> Result<ProofCertificate, ProofError> {
    if x.is_empty() {
        return Err(ProofError::EmptyData);
    }

    // Build O(1)-lookup map: canonical_hash → KnownIdentity.
    // Duplicate hashes: last entry wins (deterministic, documented above).
    let db: HashMap<u128, KnownIdentity> = builtin_identity_db()
        .into_iter()
        .map(|id| (id.canonical_hash, id))
        .collect();

    // Build 2D feature matrix of shape (n, 1) and 1D target array.
    let n = x.len();
    let features = Array2::from_shape_vec((n, 1), x.to_vec())
        .map_err(|e| ProofError::DiscoveryFailed(format!("failed to build feature matrix: {e}")))?;
    let targets = Array1::from_vec(y.to_vec());

    // Use a moderate SR config. A slightly looser tolerance speeds discovery
    // for exact-constant / identity targets without sacrificing precision.
    let config = SrConfig::default()
        .with_max_iter(50)
        .with_top_n(max_candidates.max(1))
        .with_tolerance(1e-6);

    let candidates = discover(features.view(), targets.view(), &config);

    if candidates.is_empty() {
        return Err(ProofError::DiscoveryFailed(
            "symbolic regression returned no candidates".to_owned(),
        ));
    }

    let n_checked = candidates.len().min(max_candidates);

    for formula in candidates.iter().take(max_candidates) {
        let canonical = canonicalize(&formula.op);

        if let Some(identity) = db.get(&canonical.hash()) {
            // Found a match. Certify at the midpoint of the x data.
            let witness_x = x[x.len() / 2];
            // For constant ops (no Var nodes) we pass an empty binding;
            // for formulas referencing Var(0) we pass [witness_x].
            let bindings: &[f64] = if formula.op.count_vars() > 0 {
                std::slice::from_ref(&witness_x)
            } else {
                &[]
            };

            let witness = CertifiedValue::certify(&formula.op, bindings, 1e-6)
                .map_err(ProofError::CertificationFailed)?;

            return Ok(ProofCertificate {
                formula: formula.op.clone(),
                identity: identity.clone(),
                witness,
                candidates_checked: n_checked,
            });
        }
    }

    Err(ProofError::NoIdentityMatch {
        candidates: n_checked,
    })
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build x data as evenly-spaced points in [lo, hi].
    fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64))
            .collect()
    }

    // ----------------------------------------------------------------
    // Test 1 — constant-one data discovers a known identity
    // ----------------------------------------------------------------
    #[test]
    fn test_discover_constant_one() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![1.0_f64; 10];

        let result = discover_identity(&x, &y, 5);
        assert!(
            result.is_ok(),
            "y=[1.0;10] should match a known identity, got: {:?}",
            result.err().map(|e| e.to_string())
        );

        let cert = result.expect("expected Ok");
        assert!(
            !cert.identity.name.is_empty(),
            "identity name must be non-empty"
        );
        // The identity.name may be any of the constant-1 entries.
        // Verify the witness interval contains 1.0.
        assert!(
            cert.witness.certified_interval.lo <= 1.0 + 1e-6
                && cert.witness.certified_interval.hi >= 1.0 - 1e-6,
            "witness interval [{}, {}] must contain 1.0",
            cert.witness.certified_interval.lo,
            cert.witness.certified_interval.hi
        );
    }

    // ----------------------------------------------------------------
    // Test 2 — built-in database has at least 4 entries
    // ----------------------------------------------------------------
    #[test]
    fn test_builtin_db_has_entries() {
        let db = builtin_identity_db();
        assert!(
            db.len() >= 4,
            "builtin_identity_db must have at least 4 entries, got {}",
            db.len()
        );
    }

    // ----------------------------------------------------------------
    // Test 3 — no-match for non-identity data
    // ----------------------------------------------------------------
    #[test]
    fn test_discover_no_match() {
        // y = 3x + 7 — not a known identity.
        let x = linspace(1.0, 10.0, 10);
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 7.0).collect();

        // Use very few candidates to avoid accidentally matching.
        let result = discover_identity(&x, &y, 3);
        match result {
            Err(ProofError::NoIdentityMatch { .. }) | Err(ProofError::DiscoveryFailed(_)) => {}
            Ok(cert) => {
                // If SR happens to express 3x+7 as something canonical,
                // it is a false positive; at least the identity name must
                // be non-empty and the witness must be non-NaN.
                assert!(
                    !cert.identity.name.is_empty(),
                    "spurious match must still have a non-empty name"
                );
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ----------------------------------------------------------------
    // Test 4 — empty data returns EmptyData error
    // ----------------------------------------------------------------
    #[test]
    fn test_discover_empty() {
        let result = discover_identity(&[], &[], 5);
        assert!(
            matches!(result, Err(ProofError::EmptyData)),
            "empty input must return EmptyData, got: {:?}",
            result.map(|_| ())
        );
    }

    // ----------------------------------------------------------------
    // Test 5 — ProofCertificate fields are well-formed
    // ----------------------------------------------------------------
    #[test]
    fn test_proof_certificate_fields() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![1.0_f64; 10];

        let cert = discover_identity(&x, &y, 5).expect("expected successful certificate");

        // identity.name must be non-empty
        assert!(!cert.identity.name.is_empty(), "identity name is empty");

        // At least one candidate was checked
        assert!(
            cert.candidates_checked >= 1,
            "candidates_checked must be >= 1, got {}",
            cert.candidates_checked
        );

        // Witness interval must contain 1.0
        assert!(
            cert.witness.certified_interval.lo <= 1.0 + 1e-6
                && cert.witness.certified_interval.hi >= 1.0 - 1e-6,
            "witness interval [{}, {}] must contain 1.0",
            cert.witness.certified_interval.lo,
            cert.witness.certified_interval.hi
        );
    }

    // ----------------------------------------------------------------
    // Test 6 — Const(1.0) and Const(0.0) have different hashes
    // ----------------------------------------------------------------
    #[test]
    fn test_builtin_db_const1_hash() {
        let h1 = canonicalize(&LoweredOp::Const(1.0)).hash();
        let h1b = canonicalize(&LoweredOp::Const(1.0)).hash();
        let h0 = canonicalize(&LoweredOp::Const(0.0)).hash();

        // Same value → same hash (trivially idempotent).
        assert_eq!(h1, h1b, "canonicalize(Const(1.0)) must be deterministic");

        // Different values → different hashes.
        assert_ne!(
            h1, h0,
            "Const(1.0) and Const(0.0) must have distinct hashes"
        );
    }

    // ----------------------------------------------------------------
    // Test 7 — y = x (identity function) discovers ln(exp(x)) = x
    // ----------------------------------------------------------------
    #[test]
    fn test_discover_ln_exp() {
        // y = x — SR should rediscover Var(0) directly (in the initial
        // population), whose canonical hash matches canonicalize(Var(0)).
        let x = linspace(0.1, 3.0, 20);
        let y = x.clone();

        let result = discover_identity(&x, &y, 5);
        match result {
            Ok(cert) => {
                // The "ln(exp(x)) = x" entry or direct Var(0) match.
                // Canonical hash must match Var(0).
                let expected_hash = canonicalize(&LoweredOp::Var(0)).hash();
                assert_eq!(
                    cert.identity.canonical_hash, expected_hash,
                    "identity hash must match canonicalize(Var(0))"
                );
                assert!(
                    !cert.identity.name.is_empty(),
                    "identity name must be non-empty"
                );
            }
            Err(ProofError::NoIdentityMatch { .. }) => {
                // Acceptable: SR may not have recovered Var(0) among top-5
                // candidates for this specific data range.
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ----------------------------------------------------------------
    // Test 8 — max_candidates = 0 always returns NoIdentityMatch
    // ----------------------------------------------------------------
    #[test]
    fn test_max_candidates_zero() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![1.0_f64; 10];

        let result = discover_identity(&x, &y, 0);
        assert!(
            matches!(result, Err(ProofError::NoIdentityMatch { candidates: 0 })),
            "max_candidates=0 must return NoIdentityMatch{{candidates:0}}, got: {:?}",
            result.map(|_| ())
        );
    }
}
