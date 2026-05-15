//! Symbolic closed-form RoPE attention logit module.
//!
//! Rotary Position Embedding (RoPE) rotates query and key vectors pair-wise
//! using sinusoidal position encodings.  The key insight — exploited here — is
//! that the dot-product of RoPE-rotated query and key depends **only** on the
//! relative position `(m − n)`, not on the absolute positions `m` and `n`
//! separately.
//!
//! ## Closed form
//!
//! For a head dimension `d_head` (must be even) the logit is:
//!
//! ```text
//! RoPE(q, m) · RoPE(k, n) =
//!   Σᵢ  [ (q_{2i}·k_{2i} + q_{2i+1}·k_{2i+1}) · cos((m−n)·θᵢ)
//!        + (q_{2i+1}·k_{2i} − q_{2i}·k_{2i+1}) · sin((m−n)·θᵢ) ]
//! ```
//!
//! where `θᵢ = theta_base^(−2i / d_head)` and `i ∈ {0, …, d_head/2 − 1}`.
//!
//! ## Variable layout
//!
//! | Index            | Meaning                            |
//! |------------------|------------------------------------|
//! | `Var(0)`         | relative position `(m − n)`        |
//! | `Var(1 + 4·i)`   | `q_{2i}`  (i-th pair, query even)  |
//! | `Var(2 + 4·i)`   | `q_{2i+1}`(i-th pair, query odd)   |
//! | `Var(3 + 4·i)`   | `k_{2i}`  (i-th pair, key even)    |
//! | `Var(4 + 4·i)`   | `k_{2i+1}`(i-th pair, key odd)     |
//!
//! Total variables: `1 + 4 · (d_head / 2)`.

use scirs2_symbolic::cas::{canonicalize, Canonical};
use scirs2_symbolic::eml::op::LoweredOp;
use std::sync::Arc;

/// Mapping from semantic names to [`LoweredOp`] variable indices.
#[derive(Debug, Clone)]
pub struct RopeVarMap {
    /// Variable index for the relative position `(m − n)`.  Always `0`.
    pub relative_position: usize,
    /// Pairs `(var_idx_q_{2i}, var_idx_q_{2i+1})` for each head-dimension pair `i`.
    pub q_pairs: Vec<(usize, usize)>,
    /// Pairs `(var_idx_k_{2i}, var_idx_k_{2i+1})` for each head-dimension pair `i`.
    pub k_pairs: Vec<(usize, usize)>,
}

/// A symbolically constructed RoPE attention logit.
///
/// The expression in `logit` is already canonicalized and evaluates, given a
/// variable assignment built from [`RopeVarMap`], to the dot-product of the
/// RoPE-rotated query and key vectors.
#[derive(Debug, Clone)]
pub struct RopeAttentionSymbolic {
    /// Head dimension (even, ≤ 256).
    pub d_head: usize,
    /// Base for the geometric frequency schedule (> 1.0, finite).
    pub theta_base: f64,
    /// Canonical closed-form logit expression.
    pub logit: Arc<LoweredOp>,
    /// Variable index map for evaluating `logit`.
    pub variables: RopeVarMap,
}

/// Errors that can arise while building a [`RopeAttentionSymbolic`].
#[derive(Debug, Clone, PartialEq)]
pub enum RopeAttentionError {
    /// `d_head` is not divisible by 2.
    OddDimension(usize),
    /// `d_head > 256`.
    DimensionTooLarge(usize),
    /// `theta_base ≤ 1.0` or not finite.
    InvalidBase(f64),
}

impl std::fmt::Display for RopeAttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OddDimension(d) => write!(f, "d_head must be even, got {d}"),
            Self::DimensionTooLarge(d) => write!(f, "d_head must be ≤ 256, got {d}"),
            Self::InvalidBase(b) => write!(f, "theta_base must be > 1.0 and finite, got {b}"),
        }
    }
}

impl std::error::Error for RopeAttentionError {}

/// Build the symbolic closed-form RoPE attention logit.
///
/// The returned [`RopeAttentionSymbolic`] contains a single [`LoweredOp`] that
/// computes the full relative-position dot-product for any query / key pair.
///
/// # Errors
///
/// Returns an error when:
/// - `d_head % 2 != 0`  ([`RopeAttentionError::OddDimension`])
/// - `d_head > 256`      ([`RopeAttentionError::DimensionTooLarge`])
/// - `theta_base <= 1.0` or not finite ([`RopeAttentionError::InvalidBase`])
///
/// # Variable layout
///
/// See the [module-level documentation](self) for the exact `Var(n)` → value
/// mapping.
pub fn rope_attention_logit(
    d_head: usize,
    theta_base: f64,
) -> Result<RopeAttentionSymbolic, RopeAttentionError> {
    // --- Validation -------------------------------------------------------
    if !d_head.is_multiple_of(2) {
        return Err(RopeAttentionError::OddDimension(d_head));
    }
    if d_head > 256 {
        return Err(RopeAttentionError::DimensionTooLarge(d_head));
    }
    if theta_base <= 1.0 || !theta_base.is_finite() {
        return Err(RopeAttentionError::InvalidBase(theta_base));
    }

    let n_pairs = d_head / 2;

    // --- Build terms iteratively ------------------------------------------
    // Each pair i contributes two terms:
    //   cos_term = (q_{2i}·k_{2i} + q_{2i+1}·k_{2i+1}) · cos(Var(0) · θᵢ)
    //   sin_term = (q_{2i+1}·k_{2i} − q_{2i}·k_{2i+1}) · sin(Var(0) · θᵢ)
    let mut terms: Vec<LoweredOp> = Vec::with_capacity(2 * n_pairs);

    for i in 0..n_pairs {
        let theta_i = theta_base.powf(-2.0 * (i as f64) / (d_head as f64));

        let q0_idx = 1 + 4 * i;
        let q1_idx = 2 + 4 * i;
        let k0_idx = 3 + 4 * i;
        let k1_idx = 4 + 4 * i;

        // angle = Var(0) * Const(theta_i)
        let angle = LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(theta_i)),
        );

        // dot_same = q_{2i}·k_{2i} + q_{2i+1}·k_{2i+1}
        let dot_same = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(q0_idx)),
                Box::new(LoweredOp::Var(k0_idx)),
            )),
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(q1_idx)),
                Box::new(LoweredOp::Var(k1_idx)),
            )),
        );

        // cos_term = dot_same · cos(angle)
        let cos_term = LoweredOp::Mul(
            Box::new(dot_same),
            Box::new(LoweredOp::Cos(Box::new(angle.clone()))),
        );

        // cross_diff = q_{2i+1}·k_{2i} − q_{2i}·k_{2i+1}
        let cross_diff = LoweredOp::Sub(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(q1_idx)),
                Box::new(LoweredOp::Var(k0_idx)),
            )),
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(q0_idx)),
                Box::new(LoweredOp::Var(k1_idx)),
            )),
        );

        // sin_term = cross_diff · sin(angle)
        let sin_term = LoweredOp::Mul(
            Box::new(cross_diff),
            Box::new(LoweredOp::Sin(Box::new(angle))),
        );

        terms.push(cos_term);
        terms.push(sin_term);
    }

    // Fold terms into a left-to-right sum (iterative, no recursion).
    // n_pairs >= 1 (d_head >= 2 and even), so terms is non-empty.
    // The fallback `Const(0.0)` is unreachable but needed for type-safety.
    let logit_raw = terms
        .into_iter()
        .reduce(|acc, t| LoweredOp::Add(Box::new(acc), Box::new(t)))
        .unwrap_or(LoweredOp::Const(0.0));

    // Canonicalize per the public-API policy.
    let logit = canonicalize(&logit_raw).into_op();

    // --- Build variable map -----------------------------------------------
    let q_pairs: Vec<(usize, usize)> = (0..n_pairs).map(|i| (1 + 4 * i, 2 + 4 * i)).collect();
    let k_pairs: Vec<(usize, usize)> = (0..n_pairs).map(|i| (3 + 4 * i, 4 + 4 * i)).collect();

    Ok(RopeAttentionSymbolic {
        d_head,
        theta_base,
        logit: Arc::new(logit),
        variables: RopeVarMap {
            relative_position: 0,
            q_pairs,
            k_pairs,
        },
    })
}

/// Build a variable bindings slice for evaluating the logit produced by
/// [`rope_attention_logit`].
///
/// - `rel_pos`  — relative position `m − n`
/// - `q`        — query vector of length `d_head`
/// - `k`        — key vector of length `d_head`
///
/// # Panics (in tests only)
///
/// Panics if `q.len() != d_head` or `k.len() != d_head`.  In production code
/// check lengths before calling.
pub fn build_vars(d_head: usize, rel_pos: f64, q: &[f64], k: &[f64]) -> Vec<f64> {
    let n_pairs = d_head / 2;
    let n_vars = 1 + 4 * n_pairs;
    let mut vars = vec![0.0_f64; n_vars];
    vars[0] = rel_pos;
    for i in 0..n_pairs {
        vars[1 + 4 * i] = q[2 * i];
        vars[2 + 4 * i] = q[2 * i + 1];
        vars[3 + 4 * i] = k[2 * i];
        vars[4 + 4 * i] = k[2 * i + 1];
    }
    vars
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_symbolic::cas::canonicalize;
    use scirs2_symbolic::eml::{eval_real, EvalCtx};

    // ------------------------------------------------------------------
    // Test 1: d_head=2 structural shape
    // ------------------------------------------------------------------
    #[test]
    fn test_d2_structural_shape() {
        let sym = rope_attention_logit(2, 10000.0).expect("d_head=2 should succeed");
        // n_pairs = 1 → exactly 1 pair in each map
        assert_eq!(sym.variables.q_pairs.len(), 1);
        assert_eq!(sym.variables.k_pairs.len(), 1);
        assert_eq!(sym.variables.q_pairs[0], (1, 2));
        assert_eq!(sym.variables.k_pairs[0], (3, 4));
        assert_eq!(sym.variables.relative_position, 0);
        assert_eq!(sym.d_head, 2);
    }

    // ------------------------------------------------------------------
    // Test 2: d_head=4 gives 2 pairs
    // ------------------------------------------------------------------
    #[test]
    fn test_d4_two_pairs() {
        let sym = rope_attention_logit(4, 10000.0).expect("d_head=4 should succeed");
        assert_eq!(sym.variables.q_pairs.len(), 2);
        assert_eq!(sym.variables.k_pairs.len(), 2);
        // Pair 0
        assert_eq!(sym.variables.q_pairs[0], (1, 2));
        assert_eq!(sym.variables.k_pairs[0], (3, 4));
        // Pair 1
        assert_eq!(sym.variables.q_pairs[1], (5, 6));
        assert_eq!(sym.variables.k_pairs[1], (7, 8));
    }

    // ------------------------------------------------------------------
    // Test 3: OddDimension error
    // ------------------------------------------------------------------
    #[test]
    fn test_odd_dimension_error() {
        let err = rope_attention_logit(3, 10000.0).unwrap_err();
        assert_eq!(err, RopeAttentionError::OddDimension(3));
    }

    // ------------------------------------------------------------------
    // Test 4: DimensionTooLarge error
    // ------------------------------------------------------------------
    #[test]
    fn test_dimension_too_large_error() {
        let err = rope_attention_logit(258, 10000.0).unwrap_err();
        assert_eq!(err, RopeAttentionError::DimensionTooLarge(258));
    }

    // ------------------------------------------------------------------
    // Test 5: InvalidBase error
    // ------------------------------------------------------------------
    #[test]
    fn test_invalid_base_error() {
        let err = rope_attention_logit(4, 0.5).unwrap_err();
        assert_eq!(err, RopeAttentionError::InvalidBase(0.5));
        // Also check exactly 1.0 (border)
        let err2 = rope_attention_logit(4, 1.0).unwrap_err();
        assert_eq!(err2, RopeAttentionError::InvalidBase(1.0));
    }

    // ------------------------------------------------------------------
    // Test 6: Numerical equivalence (d_head=4)
    // ------------------------------------------------------------------
    #[test]
    fn test_numerical_equivalence_d4() {
        let d_head = 4usize;
        let theta_base = 10000.0_f64;
        let n_pairs = d_head / 2;
        let sym = rope_attention_logit(d_head, theta_base).expect("valid");

        for seed in 0_u64..50 {
            // Deterministic pseudo-random values derived from seed
            let rel_pos = ((seed * 17 + 3) % 100) as f64 - 50.0;
            let q: Vec<f64> = (0..d_head)
                .map(|j| ((seed * 7 + j as u64 * 13 + 1) % 100) as f64 / 50.0 - 1.0)
                .collect();
            let k: Vec<f64> = (0..d_head)
                .map(|j| ((seed * 11 + j as u64 * 17 + 2) % 100) as f64 / 50.0 - 1.0)
                .collect();

            // Reference: compute directly from the closed-form formula
            let mut expected = 0.0_f64;
            for i in 0..n_pairs {
                let theta_i = theta_base.powf(-2.0 * (i as f64) / (d_head as f64));
                let angle = rel_pos * theta_i;
                let q0 = q[2 * i];
                let q1 = q[2 * i + 1];
                let k0 = k[2 * i];
                let k1 = k[2 * i + 1];
                expected += (q0 * k0 + q1 * k1) * angle.cos() + (q1 * k0 - q0 * k1) * angle.sin();
            }

            // Build variable vector for symbolic evaluation
            let vars = build_vars(d_head, rel_pos, &q, &k);
            let ctx = EvalCtx::new(&vars);
            let symbolic_val = eval_real(&sym.logit, &ctx).expect("eval_real should not fail");

            assert!(
                (expected - symbolic_val).abs() < 1e-10,
                "seed={seed}: expected={expected} symbolic={symbolic_val} diff={}",
                (expected - symbolic_val).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 7: q_pairs.len() == d_head/2 for various d_head values
    // ------------------------------------------------------------------
    #[test]
    fn test_pair_counts_for_various_d_head() {
        for &d in &[2_usize, 4, 8, 16] {
            let sym = rope_attention_logit(d, 10000.0).expect("valid");
            assert_eq!(
                sym.variables.q_pairs.len(),
                d / 2,
                "d_head={d}: q_pairs.len() mismatch"
            );
            assert_eq!(
                sym.variables.k_pairs.len(),
                d / 2,
                "d_head={d}: k_pairs.len() mismatch"
            );
            assert_eq!(sym.d_head, d);
        }
    }

    // ------------------------------------------------------------------
    // Test 8: logit varies with relative position
    // ------------------------------------------------------------------
    #[test]
    fn test_relative_position_dependence() {
        let sym = rope_attention_logit(2, 10000.0).expect("valid");
        let q = [1.0_f64, 0.0];
        let k = [1.0_f64, 0.0];
        // vars layout for d_head=2: [rel_pos, q0, q1, k0, k1]
        let eval_at = |rp: f64| -> f64 {
            let vars = [rp, q[0], q[1], k[0], k[1]];
            let ctx = EvalCtx::new(&vars);
            eval_real(&sym.logit, &ctx).expect("eval ok")
        };

        let v0 = eval_at(0.0);
        let v5 = eval_at(5.0);
        let v10 = eval_at(10.0);
        // For q=[1,0], k=[1,0]: logit = cos(rel_pos·θ₀), which varies.
        assert!(
            (v0 - v5).abs() > 1e-6 || (v0 - v10).abs() > 1e-6,
            "logit should vary with relative position: v0={v0} v5={v5} v10={v10}"
        );
    }

    // ------------------------------------------------------------------
    // Test 9: canonicalize is idempotent
    // ------------------------------------------------------------------
    #[test]
    fn test_canonicalize_idempotent() {
        let sym = rope_attention_logit(4, 10000.0).expect("valid");
        let logit = &*sym.logit;

        let can1 = canonicalize(logit).into_op();
        let can2 = canonicalize(&can1).into_op();

        // Verify numerical equivalence at a fixed test point
        let test_vars: Vec<f64> = (0..9).map(|i| i as f64 * 0.1 + 0.1).collect();
        let ctx = EvalCtx::new(&test_vars);
        let v1 = eval_real(&can1, &ctx).expect("can1 eval ok");
        let v2 = eval_real(&can2, &ctx).expect("can2 eval ok");
        assert!(
            (v1 - v2).abs() < 1e-12,
            "canonicalize should be idempotent: v1={v1} v2={v2} diff={}",
            (v1 - v2).abs()
        );
    }
}
