//! `Canonical` — EML encodings of elementary functions.
//!
//! Each constructor returns the canonical [`EmlTree`] for the corresponding
//! function. The encodings follow Odrzywolek 2026
//! ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852)) where every
//! elementary function reduces to the single binary operator
//! `eml(x, y) = exp(x) - ln(y)` plus the constant `1`.
//!
//! # Examples
//!
//! ```
//! use scirs2_symbolic::eml::{Canonical, EmlTree};
//!
//! let x = EmlTree::var(0);
//! let formula = Canonical::add(&Canonical::sin(&x), &Canonical::cos(&x));
//! // formula represents sin(x) + cos(x) as a canonical EML tree.
//! ```
//!
//! # Adapted from oxieml v0.1.0, src/canonical.rs
//!
//! Encodings derived from Odrzywolek 2026 (arXiv:2603.21852). The encoding
//! bodies are preserved verbatim from oxieml; the API surface is clean-room
//! (returns [`Result<EmlTree, EmlError>`](crate::error::EmlError) on invalid
//! input where oxieml uses panic).
//!
//! # Numerical correctness — deferred to Phase 0 item 5
//!
//! Tests in this module are limited to **structural** properties (depth,
//! size, determinism, hash equality). Numerical correctness via
//! `eval_real` is BLOCKED until Phase 0 item 5 (the `eml::eval` module)
//! lands; once it does, oxieml's golden numerical-roundtrip suite should
//! be ported in addition. See `tests/canonical_numerical.rs` (TBD).

use crate::eml::tree::EmlTree;
use crate::error::EmlError;

/// Namespace for the canonical EML constructors.
///
/// All methods are static; `Canonical` is a zero-sized type used purely
/// for API namespacing. See module docs for the per-function table.
pub struct Canonical;

impl Canonical {
    // ================================================================
    // Table 1 — Basic operations: exp / ln / e / neg
    // ================================================================

    /// `exp(x) = eml(x, 1)` — depth `+1`.
    ///
    /// `eml(x, 1) = exp(x) - ln(1) = exp(x)`.
    pub fn exp(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        EmlTree::eml(x, &one)
    }

    /// `ln(x) = eml(1, eml(eml(1, x), 1))` — depth `+3`.
    ///
    /// Construction:
    /// - Inner: `eml(1, x) = e - ln(x)`
    /// - Middle: `eml(e - ln(x), 1) = exp(e - ln(x)) = exp(e)/x`
    /// - Outer: `eml(1, exp(e)/x) = e - ln(exp(e)/x) = e - (e - ln(x)) = ln(x)`
    pub fn ln(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let inner = EmlTree::eml(&one, x); // e - ln(x)
        let middle = EmlTree::eml(&inner, &one); // exp(e - ln(x))
        EmlTree::eml(&one, &middle) // e - ln(exp(e-ln(x))) = ln(x)
    }

    /// `e = eml(1, 1)` — depth `1`.
    ///
    /// `eml(1, 1) = exp(1) - ln(1) = e - 0 = e`.
    pub fn euler() -> EmlTree {
        let one = EmlTree::one();
        EmlTree::eml(&one, &one)
    }

    /// `-x` (negation) — depth `+6`.
    ///
    /// Uses the identity `-x = (e - x) - e`, encoded as
    /// `eml(ln(e-x), exp(e))`:
    /// - `exp(ln(e-x)) - ln(exp(e)) = (e - x) - e = -x`.
    ///
    /// Works in the complex domain where `ln` is defined for all nonzero
    /// values, so the construction holds for all real `x`.
    pub fn neg(x: &EmlTree) -> EmlTree {
        let e_minus_x = Self::e_minus(x);
        let ln_e_minus_x = Self::ln(&e_minus_x);
        let exp_e = Self::exp(&Self::euler());
        // eml(ln(e-x), exp(e)) = exp(ln(e-x)) - ln(exp(e)) = (e-x) - e = -x
        EmlTree::eml(&ln_e_minus_x, &exp_e)
    }

    // ================================================================
    // Table 2 — Arithmetic operations
    // ================================================================

    /// `x + y = sub(x, neg(y))` = `x - (-y)`.
    ///
    /// Built by composing subtraction and negation.
    pub fn add(x: &EmlTree, y: &EmlTree) -> EmlTree {
        Self::sub(x, &Self::neg(y))
    }

    /// `x - y` — subtraction via `eml(ln(x), eml(y, 1))`.
    ///
    /// Key identity: `eml(ln(x), eml(y, 1)) = exp(ln(x)) - ln(exp(y)) = x - y`.
    ///
    /// Relies on complex evaluation where `exp(ln(z)) = z` on the
    /// principal branch, making the identity valid for all real inputs.
    pub fn sub(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let ln_x = Self::ln(x);
        let exp_y = EmlTree::eml(y, &one); // eml(y, 1) = exp(y)
                                           // eml(ln(x), exp(y)) = exp(ln(x)) - ln(exp(y)) = x - y
        EmlTree::eml(&ln_x, &exp_y)
    }

    /// `x * y = exp(ln(x) + ln(y))` — via `exp(add(ln(x), ln(y)))`.
    ///
    /// Uses the logarithmic identity `ln(x) + ln(y) = ln(xy)`, so
    /// `exp(ln(x) + ln(y)) = xy`.
    pub fn mul(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let ln_y = Self::ln(y);
        let sum = Self::add(&ln_x, &ln_y);
        Self::exp(&sum)
    }

    /// `x / y = exp(ln(x) - ln(y))`.
    ///
    /// Uses `ln(x/y) = ln(x) - ln(y)`.
    pub fn div(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let ln_y = Self::ln(y);
        let diff = Self::sub(&ln_x, &ln_y);
        Self::exp(&diff)
    }

    /// `x ^ y = exp(y * ln(x))`.
    pub fn pow(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let y_ln_x = Self::mul(y, &ln_x);
        Self::exp(&y_ln_x)
    }

    // ================================================================
    // Table 3 — Trigonometric (via complex numbers)
    // ================================================================

    /// `pi()` — returns a tree whose complex evaluation yields `iπ`.
    ///
    /// Construction: `ln(-1) = iπ` in the complex domain.
    /// This tree is used internally by `sin`/`cos` and is NOT intended
    /// for direct `eval_real` (which would return a `ComplexResult`-style
    /// error once item 5 lands).
    pub fn pi() -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        Self::ln(&neg_one) // ln(-1) = iπ
    }

    /// `sin(x) = (exp(ix) - exp(-ix)) / (2i)` — Euler formula.
    ///
    /// Constructs `i = exp(iπ/2) = exp(ln(-1)/2)`, then builds the Euler
    /// decomposition. Evaluates correctly through the complex evaluation
    /// path (item 5).
    pub fn sin(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        let half = Self::reciprocal(&Self::add(&one, &one));

        // i = exp(ln(-1) * 1/2) = exp(iπ/2)
        let ln_neg_one = Self::ln(&neg_one);
        let half_ln_neg_one = Self::mul(&half, &ln_neg_one);
        let i_val = Self::exp(&half_ln_neg_one);

        // exp(ix) and exp(-ix)
        let ix = Self::mul(&i_val, x);
        let exp_ix = Self::exp(&ix);
        let neg_ix = Self::neg(&ix);
        let exp_neg_ix = Self::exp(&neg_ix);

        // (exp(ix) - exp(-ix)) / (2i)
        let diff = Self::sub(&exp_ix, &exp_neg_ix);
        let two = Self::add(&one, &one);
        let two_i = Self::mul(&two, &i_val);
        Self::div(&diff, &two_i)
    }

    /// `cos(x) = (exp(ix) + exp(-ix)) / 2` — Euler formula.
    ///
    /// Same construction as `sin` but using the real-part identity.
    pub fn cos(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        let half = Self::reciprocal(&Self::add(&one, &one));

        let ln_neg_one = Self::ln(&neg_one);
        let half_ln_neg_one = Self::mul(&half, &ln_neg_one);
        let i_val = Self::exp(&half_ln_neg_one);

        let ix = Self::mul(&i_val, x);
        let exp_ix = Self::exp(&ix);
        let neg_ix = Self::neg(&ix);
        let exp_neg_ix = Self::exp(&neg_ix);

        // (exp(ix) + exp(-ix)) / 2
        let sum = Self::add(&exp_ix, &exp_neg_ix);
        let two = Self::add(&one, &one);
        Self::div(&sum, &two)
    }

    /// `tan(x) = sin(x) / cos(x)`.
    pub fn tan(x: &EmlTree) -> EmlTree {
        Self::div(&Self::sin(x), &Self::cos(x))
    }

    // ================================================================
    // Table 4 — Inverse trigonometric (via complex logarithms)
    // ================================================================

    /// `arcsin(x) = -i * ln(ix + sqrt(1 - x²))`.
    ///
    /// Uses the complex logarithm identity. The imaginary parts cancel
    /// for real inputs in `[-1, 1]`, yielding a real result.
    pub fn arcsin(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let ix = Self::mul(&i, x);
        let x_sq = Self::square(x);
        let one_minus_x_sq = Self::sub(&one, &x_sq);
        let sqrt_part = Self::sqrt(&one_minus_x_sq);
        // -i * ln(ix + sqrt(1 - x^2))
        Self::neg(&Self::mul(&i, &Self::ln(&Self::add(&ix, &sqrt_part))))
    }

    /// `arccos(x) = -i * ln(x + i * sqrt(1 - x²))`.
    ///
    /// Alternative form that avoids subtracting from π/2.
    pub fn arccos(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        let one_minus_x_sq = Self::sub(&one, &x_sq);
        let sqrt_part = Self::sqrt(&one_minus_x_sq);
        let i_sqrt = Self::mul(&i, &sqrt_part);
        // -i * ln(x + i*sqrt(1-x^2))
        Self::neg(&Self::mul(&i, &Self::ln(&Self::add(x, &i_sqrt))))
    }

    /// `arctan(x) = (-i/2) * ln((1 + ix) / (1 - ix))`.
    ///
    /// Uses the complex logarithm identity for arctan. The imaginary
    /// parts cancel for all real `x`, yielding a real result.
    pub fn arctan(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let two = Self::nat_unchecked(2);
        let ix = Self::mul(&i, x);
        let numerator = Self::add(&one, &ix);
        let denominator = Self::sub(&one, &ix);
        // (-i/2) * ln((1+ix)/(1-ix))
        let neg_i_half = Self::neg(&Self::mul(&i, &Self::reciprocal(&two)));
        Self::mul(&neg_i_half, &Self::ln(&Self::div(&numerator, &denominator)))
    }

    // ================================================================
    // Table 5 — Hyperbolic functions
    // ================================================================

    /// `sinh(x) = (exp(x) - exp(-x)) / 2`.
    pub fn sinh(x: &EmlTree) -> EmlTree {
        let exp_x = Self::exp(x);
        let exp_neg_x = Self::exp(&Self::neg(x));
        Self::div(&Self::sub(&exp_x, &exp_neg_x), &Self::nat_unchecked(2))
    }

    /// `cosh(x) = (exp(x) + exp(-x)) / 2`.
    pub fn cosh(x: &EmlTree) -> EmlTree {
        let exp_x = Self::exp(x);
        let exp_neg_x = Self::exp(&Self::neg(x));
        Self::div(&Self::add(&exp_x, &exp_neg_x), &Self::nat_unchecked(2))
    }

    /// `tanh(x) = sinh(x) / cosh(x)`.
    pub fn tanh(x: &EmlTree) -> EmlTree {
        Self::div(&Self::sinh(x), &Self::cosh(x))
    }

    // ================================================================
    // Table 6 — Inverse hyperbolic functions
    // ================================================================

    /// `arcsinh(x) = ln(x + sqrt(x² + 1))`.
    pub fn arcsinh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        Self::ln(&Self::add(x, &Self::sqrt(&Self::add(&x_sq, &one))))
    }

    /// `arccosh(x) = ln(x + sqrt(x² - 1))` — defined for `x >= 1`.
    pub fn arccosh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        Self::ln(&Self::add(x, &Self::sqrt(&Self::sub(&x_sq, &one))))
    }

    /// `arctanh(x) = (1/2) * ln((1 + x) / (1 - x))` — defined for `|x| < 1`.
    pub fn arctanh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let two = Self::nat_unchecked(2);
        let half = Self::reciprocal(&two);
        let numerator = Self::add(&one, x);
        let denominator = Self::sub(&one, x);
        Self::mul(&half, &Self::ln(&Self::div(&numerator, &denominator)))
    }

    // ================================================================
    // Table 7 — Powers, roots, abs
    // ================================================================

    /// `x² = exp(2 * ln(x))` — square.
    pub fn square(x: &EmlTree) -> EmlTree {
        Self::pow(x, &Self::nat_unchecked(2))
    }

    /// `sqrt(x) = x^0.5 = exp(0.5 * ln(x))`.
    pub fn sqrt(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let two = Self::add(&one, &one);
        let half = Self::reciprocal(&two);
        Self::pow(x, &half)
    }

    /// `abs(x) = sqrt(x²)`.
    pub fn abs(x: &EmlTree) -> EmlTree {
        Self::sqrt(&Self::square(x))
    }

    // ================================================================
    // Constants
    // ================================================================

    /// `-1 = neg(1)` — the constant negative one.
    pub fn neg_one() -> EmlTree {
        Self::neg(&EmlTree::one())
    }

    /// `-2 = neg(nat(2))` — the constant negative two.
    pub fn neg_two() -> EmlTree {
        Self::neg(&Self::nat_unchecked(2))
    }

    /// `i = exp(iπ/2)` — the imaginary unit.
    ///
    /// Construction: `i = exp(ln(-1) / 2)`. Since `ln(-1) = iπ`, we get
    /// `exp(iπ/2) = cos(π/2) + i*sin(π/2) = i`.
    ///
    /// This tree evaluates to a purely imaginary complex number.
    /// `eval_real` (item 5) will return a `ComplexResult`-style error.
    pub fn imag_unit() -> EmlTree {
        let two = Self::nat_unchecked(2);
        let half = Self::reciprocal(&two);
        let ln_neg_one = Self::ln(&Self::neg_one()); // iπ
        Self::exp(&Self::mul(&half, &ln_neg_one)) // exp(iπ/2) = i
    }

    /// `0 = ln(1)` — the additive identity.
    ///
    /// Built canonically as `Canonical::ln(EmlTree::one())`.
    pub fn zero() -> EmlTree {
        Self::ln(&EmlTree::one())
    }

    /// Build a natural number `n >= 1` as an EML tree: `n = 1 + 1 + ... + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`EmlError::InvalidConstant`] for `n == 0`. Use
    /// [`Canonical::zero`] explicitly for the additive identity.
    pub fn nat(n: u64) -> Result<EmlTree, EmlError> {
        if n == 0 {
            return Err(EmlError::InvalidConstant(
                "nat(0) — use Canonical::zero() instead".to_string(),
            ));
        }
        Ok(Self::nat_unchecked(n))
    }

    // ================================================================
    // Helpers
    // ================================================================

    /// `1/x = exp(-ln(x))` — reciprocal.
    pub fn reciprocal(x: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let neg_ln_x = Self::neg(&ln_x);
        Self::exp(&neg_ln_x)
    }

    /// `e - x = eml(1, eml(x, 1))` — depth `+2`.
    ///
    /// `eml(1, eml(x, 1)) = exp(1) - ln(exp(x)) = e - x`.
    /// Private helper used by [`Canonical::neg`].
    fn e_minus(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(x, &one);
        EmlTree::eml(&one, &exp_x)
    }

    /// Internal infallible `nat` — only callable with `n >= 1`.
    ///
    /// Every internal call site passes a positive literal; the public
    /// [`Canonical::nat`] is the validated entry point. Keeping the
    /// internal builder infallible avoids a cascade of `Result`-returning
    /// constructors throughout the table (e.g. `arctan`, `sinh`, `cosh`,
    /// `arctanh`, `square`, `imag_unit`, `neg_two` all call `nat(2)`).
    fn nat_unchecked(n: u64) -> EmlTree {
        let one = EmlTree::one();
        if n <= 1 {
            // Defensive — every caller passes n >= 1; this keeps the
            // helper total without panicking on the n=0 path.
            return one;
        }
        let mut result = one.clone();
        for _ in 1..n {
            result = Self::add(&result, &one);
        }
        result
    }
}

// ================================================================
// Tests — STRUCTURAL ONLY
// ================================================================
//
// Numerical correctness via `eval_real` is BLOCKED on Phase 0 item 5
// (the `eml::eval` module). Once item 5 lands, oxieml's golden roundtrip
// suite (sin(0)≈0, cos(0)≈1, exp(ln(x))≈x, etc.) should be ported here
// or into a dedicated `tests/canonical_numerical.rs`.
//
// Until then these tests verify:
// - Each constructor builds a non-empty `EmlTree` of the expected shape
// - `nat(0)` rejects with `Err(EmlError::InvalidConstant)`
// - `nat(1) == EmlTree::one()`
// - Two calls to the same constructor produce structurally-equal trees
//   (determinism — the hash-cons pool guarantees pointer identity)
// - Cached structural hashes match between independent constructions

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::tree::EmlTree;

    // ----------------------------------------------------------------
    // Table 1 — exp / ln / euler / neg
    // ----------------------------------------------------------------

    #[test]
    fn exp_depth_one() {
        let x = EmlTree::var(0);
        let e_x = Canonical::exp(&x);
        assert_eq!(e_x.depth(), 1);
        assert_eq!(e_x.size(), 3); // eml(x, 1) → 3 nodes
    }

    #[test]
    fn ln_depth_three() {
        // ln(x) = eml(1, eml(eml(1, x), 1)) — depth 3
        let x = EmlTree::var(0);
        let l_x = Canonical::ln(&x);
        assert_eq!(l_x.depth(), 3);
    }

    #[test]
    fn ln_of_one_constructs() {
        // ln(1) is canonical zero — same shape, just a different root.
        let one = EmlTree::one();
        let ln_one = Canonical::ln(&one);
        assert_eq!(ln_one.depth(), 3);
    }

    #[test]
    fn euler_is_eml_one_one() {
        let e = Canonical::euler();
        assert_eq!(e.depth(), 1);
        assert_eq!(e.size(), 3);
    }

    #[test]
    fn neg_constructs_nontrivial() {
        let x = EmlTree::var(0);
        let neg_x = Canonical::neg(&x);
        // -x decomposes via e_minus + ln + exp(e); guaranteed depth >= 6.
        assert!(neg_x.depth() >= 6);
    }

    // ----------------------------------------------------------------
    // Table 2 — arithmetic
    // ----------------------------------------------------------------

    #[test]
    fn add_constructs() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let s = Canonical::add(&x, &y);
        // add(x, y) = sub(x, neg(y)); both are non-trivial.
        assert!(s.depth() > 6);
        assert_eq!(s.num_vars(), 2);
    }

    #[test]
    fn sub_constructs() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let d = Canonical::sub(&x, &y);
        // sub uses ln(x) of depth 3 and an extra eml — root is at depth 4+.
        assert!(d.depth() >= 4);
        assert_eq!(d.num_vars(), 2);
    }

    #[test]
    fn mul_constructs() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let p = Canonical::mul(&x, &y);
        // mul = exp(add(ln(x), ln(y))); exp adds 1, add wraps further.
        assert!(p.depth() > 7);
        assert_eq!(p.num_vars(), 2);
    }

    #[test]
    fn div_constructs() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let q = Canonical::div(&x, &y);
        assert!(q.depth() > 7);
        assert_eq!(q.num_vars(), 2);
    }

    #[test]
    fn pow_constructs() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let p = Canonical::pow(&x, &y);
        // pow = exp(mul(y, ln(x))) — mul is itself ~7 deep.
        assert!(p.depth() > 8);
        assert_eq!(p.num_vars(), 2);
    }

    // ----------------------------------------------------------------
    // Table 3 — trig (deep complex-domain trees)
    // ----------------------------------------------------------------

    #[test]
    fn pi_constructs_deep() {
        let p = Canonical::pi();
        // pi = ln(neg(1)); ln adds 3 to neg's depth.
        assert!(p.depth() >= 9);
    }

    #[test]
    fn sin_constructs_deep_tree() {
        let x = EmlTree::var(0);
        let s = Canonical::sin(&x);
        // sin builds many nested operations; depth comfortably > 50.
        assert!(s.depth() > 50);
        assert_eq!(s.num_vars(), 1);
    }

    #[test]
    fn cos_constructs_deep_tree() {
        let x = EmlTree::var(0);
        let c = Canonical::cos(&x);
        assert!(c.depth() > 50);
        assert_eq!(c.num_vars(), 1);
    }

    #[test]
    fn tan_combines_sin_cos() {
        let x = EmlTree::var(0);
        let t = Canonical::tan(&x);
        // tan = div(sin, cos); deeper than either alone.
        assert!(t.depth() > 60);
        assert_eq!(t.num_vars(), 1);
    }

    // ----------------------------------------------------------------
    // Table 4 — inverse trig
    // ----------------------------------------------------------------

    #[test]
    fn arcsin_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arcsin(&x);
        assert!(a.depth() > 40);
        assert_eq!(a.num_vars(), 1);
    }

    #[test]
    fn arccos_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arccos(&x);
        assert!(a.depth() > 40);
        assert_eq!(a.num_vars(), 1);
    }

    #[test]
    fn arctan_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arctan(&x);
        assert!(a.depth() > 40);
        assert_eq!(a.num_vars(), 1);
    }

    // ----------------------------------------------------------------
    // Table 5 — hyperbolic
    // ----------------------------------------------------------------

    #[test]
    fn sinh_constructs() {
        let x = EmlTree::var(0);
        let s = Canonical::sinh(&x);
        // sinh = div(sub(exp(x), exp(neg(x))), 2)
        assert!(s.depth() > 10);
        assert_eq!(s.num_vars(), 1);
    }

    #[test]
    fn cosh_constructs() {
        let x = EmlTree::var(0);
        let c = Canonical::cosh(&x);
        assert!(c.depth() > 10);
        assert_eq!(c.num_vars(), 1);
    }

    #[test]
    fn tanh_constructs() {
        let x = EmlTree::var(0);
        let t = Canonical::tanh(&x);
        assert!(t.depth() > 12);
        assert_eq!(t.num_vars(), 1);
    }

    // ----------------------------------------------------------------
    // Table 6 — inverse hyperbolic
    // ----------------------------------------------------------------

    #[test]
    fn arcsinh_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arcsinh(&x);
        assert!(a.depth() > 10);
        assert_eq!(a.num_vars(), 1);
    }

    #[test]
    fn arccosh_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arccosh(&x);
        assert!(a.depth() > 10);
        assert_eq!(a.num_vars(), 1);
    }

    #[test]
    fn arctanh_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::arctanh(&x);
        // arctanh = mul(half, ln(div(1+x, 1-x))); deeper than 10.
        assert!(a.depth() > 10);
        assert_eq!(a.num_vars(), 1);
    }

    // ----------------------------------------------------------------
    // Table 7 — powers / roots / abs
    // ----------------------------------------------------------------

    #[test]
    fn square_constructs() {
        let x = EmlTree::var(0);
        let s = Canonical::square(&x);
        assert!(s.depth() > 8);
        assert_eq!(s.num_vars(), 1);
    }

    #[test]
    fn sqrt_constructs() {
        let x = EmlTree::var(0);
        let s = Canonical::sqrt(&x);
        assert!(s.depth() > 8);
        assert_eq!(s.num_vars(), 1);
    }

    #[test]
    fn abs_constructs() {
        let x = EmlTree::var(0);
        let a = Canonical::abs(&x);
        assert!(a.depth() > 12);
        assert_eq!(a.num_vars(), 1);
    }

    #[test]
    fn reciprocal_constructs() {
        let x = EmlTree::var(0);
        let r = Canonical::reciprocal(&x);
        // reciprocal = exp(neg(ln(x))); ln=3, neg=+6, exp=+1 → ~10.
        assert!(r.depth() >= 10);
        assert_eq!(r.num_vars(), 1);
    }

    // ----------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------

    #[test]
    fn zero_constructs() {
        let z = Canonical::zero();
        // zero = ln(1) — same shape as ln(x) with x = One leaf.
        assert_eq!(z.depth(), 3);
        assert_eq!(z.num_vars(), 0);
    }

    #[test]
    fn neg_one_constructs() {
        let n = Canonical::neg_one();
        assert!(n.depth() >= 6);
        assert_eq!(n.num_vars(), 0);
    }

    #[test]
    fn neg_two_constructs() {
        let n = Canonical::neg_two();
        assert_eq!(n.num_vars(), 0);
        assert!(n.depth() >= 6);
    }

    #[test]
    fn imag_unit_constructs_deep() {
        let i = Canonical::imag_unit();
        assert_eq!(i.num_vars(), 0);
        // imag_unit = exp(mul(half, ln(neg_one))); ln(neg_one) is ~9 deep.
        assert!(i.depth() > 10);
    }

    // ----------------------------------------------------------------
    // nat
    // ----------------------------------------------------------------

    #[test]
    fn nat_zero_returns_err() {
        let r = Canonical::nat(0);
        assert!(matches!(r, Err(EmlError::InvalidConstant(_))));
    }

    #[test]
    fn nat_one_is_one() {
        let one_tree = match Canonical::nat(1) {
            Ok(t) => t,
            Err(_) => panic!("nat(1) must succeed"),
        };
        assert_eq!(one_tree, EmlTree::one());
    }

    #[test]
    fn nat_two_constructs() {
        let two = match Canonical::nat(2) {
            Ok(t) => t,
            Err(_) => panic!("nat(2) must succeed"),
        };
        // 2 = 1 + 1 ; uses Canonical::add → at least depth 7.
        assert!(two.depth() >= 7);
    }

    #[test]
    fn nat_five_constructs() {
        let five = match Canonical::nat(5) {
            Ok(t) => t,
            Err(_) => panic!("nat(5) must succeed"),
        };
        assert_eq!(five.num_vars(), 0);
    }

    // ----------------------------------------------------------------
    // Determinism — same input → structurally equal output
    // ----------------------------------------------------------------

    #[test]
    fn determinism_exp() {
        let a = Canonical::exp(&EmlTree::var(0));
        let b = Canonical::exp(&EmlTree::var(0));
        assert_eq!(a, b);
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn determinism_ln() {
        let a = Canonical::ln(&EmlTree::var(0));
        let b = Canonical::ln(&EmlTree::var(0));
        assert_eq!(a, b);
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn determinism_sin() {
        let a = Canonical::sin(&EmlTree::var(0));
        let b = Canonical::sin(&EmlTree::var(0));
        assert_eq!(a, b);
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn determinism_pi() {
        let a = Canonical::pi();
        let b = Canonical::pi();
        assert_eq!(a, b);
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn determinism_imag_unit() {
        let a = Canonical::imag_unit();
        let b = Canonical::imag_unit();
        assert_eq!(a, b);
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    // ----------------------------------------------------------------
    // Distinctness — different inputs → different hashes
    // ----------------------------------------------------------------

    #[test]
    fn sin_and_cos_differ() {
        let x = EmlTree::var(0);
        let s = Canonical::sin(&x);
        let c = Canonical::cos(&x);
        assert_ne!(s.structural_hash(), c.structural_hash());
        assert_ne!(s, c);
    }

    #[test]
    fn euler_and_zero_differ() {
        let e = Canonical::euler();
        let z = Canonical::zero();
        assert_ne!(e.structural_hash(), z.structural_hash());
    }

    #[test]
    fn neg_one_and_neg_two_differ() {
        let n1 = Canonical::neg_one();
        let n2 = Canonical::neg_two();
        assert_ne!(n1.structural_hash(), n2.structural_hash());
    }
}
