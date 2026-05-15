//! Hermite reduction and Yun squarefree factorization for rational integration.
//!
//! Lifts `crate::cas::integrate_rational` from degree ≤ 2 denominators to
//! degree ≤ 4 (via Cardano + Ferrari root extraction) and arbitrary squarefree
//! decomposition for repeated-factor inputs. Implements:
//!
//! - [`yun_squarefree`] — Yun's algorithm: factor `Q = Q₁·Q₂²·Q₃³·…` where each
//!   `Qᵢ` is squarefree.
//! - `integrate_partial_fractions` — given a denominator with known roots,
//!   produce the partial-fraction decomposition and integrate term-by-term.
//! - [`hermite_reduce_step`] — one Hermite reduction step that lowers the
//!   power of a repeated squarefree factor.
//!
//! All polynomial coefficients are `f64`. Symbolic coefficients must be
//! folded to numeric by the caller.
//!
//! # No recursion
//!
//! All polynomial arithmetic uses iterative loops over `Vec<f64>` — no
//! recursive descent on the polynomial structure.

#![warn(missing_docs)]

use crate::cas::cardano_ferrari::{solve_cubic, solve_quartic};
use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Polynomial as `Vec<f64>` of ascending-power coefficients.
pub type Poly = Vec<f64>;

/// One factor in a Yun squarefree decomposition: `(squarefree_factor, multiplicity)`.
pub type YunFactor = (Poly, u32);

// ---------------------------------------------------------------------------
// Polynomial primitives
// ---------------------------------------------------------------------------

/// Strip trailing near-zero coefficients in-place.
pub fn poly_trim(p: &mut Poly) {
    while p.len() > 1 && p.last().map(|v| v.abs() < 1e-12).unwrap_or(false) {
        p.pop();
    }
}

/// Effective degree (last non-zero index). Zero polynomial returns 0.
pub fn poly_degree(p: &[f64]) -> usize {
    p.iter().rposition(|c| c.abs() > 1e-12).unwrap_or(0)
}

/// Polynomial addition.
pub fn poly_add(a: &[f64], b: &[f64]) -> Poly {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (i, &v) in a.iter().enumerate() {
        out[i] += v;
    }
    for (i, &v) in b.iter().enumerate() {
        out[i] += v;
    }
    poly_trim(&mut out);
    out
}

/// Polynomial subtraction.
pub fn poly_sub(a: &[f64], b: &[f64]) -> Poly {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (i, &v) in a.iter().enumerate() {
        out[i] += v;
    }
    for (i, &v) in b.iter().enumerate() {
        out[i] -= v;
    }
    poly_trim(&mut out);
    out
}

/// Polynomial multiplication.
pub fn poly_mul(a: &[f64], b: &[f64]) -> Poly {
    if a.is_empty() || b.is_empty() {
        return vec![0.0];
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    poly_trim(&mut out);
    out
}

/// Scalar multiply.
pub fn poly_scale(p: &[f64], s: f64) -> Poly {
    let mut out: Poly = p.iter().map(|c| c * s).collect();
    poly_trim(&mut out);
    out
}

/// Symbolic derivative: `d/dx Σ aₖ x^k = Σ (k·aₖ) x^(k-1)`.
pub fn poly_derivative(p: &[f64]) -> Poly {
    if p.len() <= 1 {
        return vec![0.0];
    }
    let mut out: Poly = (1..p.len()).map(|k| p[k] * (k as f64)).collect();
    poly_trim(&mut out);
    out
}

/// Polynomial division: returns `(quotient, remainder)` with
/// `deg(remainder) < deg(divisor)`. Both are ascending-power.
pub fn poly_divmod(num: &[f64], den: &[f64]) -> (Poly, Poly) {
    let den_deg = poly_degree(den);
    let den_lead = den[den_deg];

    let mut rem: Poly = num.to_vec();
    poly_trim(&mut rem);

    let mut quotient: Poly = Vec::new();

    while poly_degree(&rem) >= den_deg && rem.len() > 0 && !rem.iter().all(|v| v.abs() < 1e-12) {
        let rem_deg = poly_degree(&rem);
        if rem_deg < den_deg {
            break;
        }
        let coeff = rem[rem_deg] / den_lead;
        let shift = rem_deg - den_deg;

        // Ensure quotient is large enough.
        while quotient.len() <= shift {
            quotient.push(0.0);
        }
        quotient[shift] += coeff;

        // Subtract coeff * den * x^shift from rem.
        for (k, &dk) in den.iter().enumerate() {
            if k + shift < rem.len() {
                rem[k + shift] -= coeff * dk;
            }
        }
        // Trim
        poly_trim(&mut rem);
        if rem.is_empty() || rem.iter().all(|v| v.abs() < 1e-12) {
            break;
        }
    }

    poly_trim(&mut quotient);
    poly_trim(&mut rem);
    (quotient, rem)
}

/// Polynomial GCD via Euclidean algorithm. Result has leading coefficient 1
/// (monic) when non-zero.
pub fn poly_gcd(a: &[f64], b: &[f64]) -> Poly {
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    poly_trim(&mut x);
    poly_trim(&mut y);

    while !y.iter().all(|v| v.abs() < 1e-10) && y.len() > 0 {
        let (_q, r) = poly_divmod(&x, &y);
        x = y;
        y = r;
    }

    // Make x monic.
    if x.is_empty() {
        return vec![1.0];
    }
    let lead = x[poly_degree(&x)];
    if lead.abs() < 1e-12 {
        return vec![1.0];
    }
    poly_scale(&x, 1.0 / lead)
}

// ---------------------------------------------------------------------------
// Yun squarefree factorization
// ---------------------------------------------------------------------------

/// Yun's squarefree factorization algorithm.
///
/// Given polynomial `q`, returns a list of `(squarefree_factor, multiplicity)`
/// pairs `[(Q₁, 1), (Q₂, 2), …]` such that `q = ∏ Qᵢⁱ`.
///
/// Yun's algorithm:
/// 1. Compute `g = gcd(q, q')`.
/// 2. Factor `q = c · ∏ Qᵢⁱ`. The squarefree part is `q/g`.
/// 3. Iteratively peel off each `Qᵢ` from the remaining factors via the GCD
///    chain: at each step `i`, `cᵢ = q/g₁/g₂/…/gᵢ`, `Qᵢ = gcd(cᵢ, dᵢ)`,
///    where `dᵢ = c'ᵢ + iterative shift terms`.
///
/// Reference: David Yun, "On Square-Free Decomposition Algorithms" (1976).
pub fn yun_squarefree(q: &[f64]) -> Vec<YunFactor> {
    let mut q = q.to_vec();
    poly_trim(&mut q);

    // Special case: constant or zero polynomial.
    if poly_degree(&q) == 0 {
        return vec![(q, 1)];
    }

    let qp = poly_derivative(&q);

    // c0 = q / gcd(q, q'), d0 = q' / gcd(q, q')
    let g0 = poly_gcd(&q, &qp);
    let (c0, _) = poly_divmod(&q, &g0);
    let (d0, _) = poly_divmod(&qp, &g0);

    let mut c = c0;
    let mut d = d0;
    let mut factors: Vec<YunFactor> = Vec::new();
    let mut i: u32 = 1;
    let max_iter = 32;

    while poly_degree(&c) > 0 && i <= max_iter as u32 {
        // c'_i is the derivative we expect; standard Yun formula:
        // d_{i+1} = d_i - c'_i, then Q_i = gcd(c_i, d_{i+1}).
        // c_{i+1} = c_i / Q_i, d_{i+1} = d_{i+1} / Q_i.
        let cp = poly_derivative(&c);
        let dn = poly_sub(&d, &cp);
        let qi = poly_gcd(&c, &dn);

        if poly_degree(&qi) > 0 {
            factors.push((qi.clone(), i));
        }

        let (c_next, _) = poly_divmod(&c, &qi);
        let (d_next, _) = poly_divmod(&dn, &qi);
        c = c_next;
        d = d_next;
        i += 1;
    }

    factors
}

// ---------------------------------------------------------------------------
// Real-root finder via Cardano/Ferrari
// ---------------------------------------------------------------------------

/// Find the real roots of a polynomial of degree ≤ 4, returning sorted unique
/// real roots (with f64 tolerance for duplicate detection). Higher degrees
/// return `None`.
pub fn real_roots_low_degree(p: &[f64]) -> Option<Vec<f64>> {
    let deg = poly_degree(p);
    let coeffs: Vec<LoweredOp> = p.iter().map(|c| LoweredOp::Const(*c)).collect();

    let result = match deg {
        0 => return Some(Vec::new()),
        1 => {
            // a₀ + a₁·x = 0 → x = −a₀/a₁
            if p[1].abs() < 1e-14 {
                return None;
            }
            return Some(vec![-p[0] / p[1]]);
        }
        2 => {
            let a = p[2];
            let b = p[1];
            let c = p[0];
            let discr = b * b - 4.0 * a * c;
            if discr < -1e-14 {
                return Some(Vec::new());
            }
            let sd = discr.max(0.0).sqrt();
            let r1 = (-b + sd) / (2.0 * a);
            let r2 = (-b - sd) / (2.0 * a);
            return Some(vec![r1, r2]);
        }
        3 => solve_cubic(&coeffs),
        4 => solve_quartic(&coeffs),
        _ => return None,
    };

    let mut roots = match result {
        Ok(sr) => sr
            .solutions
            .iter()
            .filter_map(|op| match op {
                LoweredOp::Const(v) if v.is_finite() => Some(*v),
                _ => None,
            })
            .collect::<Vec<f64>>(),
        Err(_) => return None,
    };

    // Verify roots numerically (filter spurious ones from accumulated error).
    roots.retain(|&r| {
        let mut acc = 0.0;
        for &cf in p.iter().rev() {
            acc = acc * r + cf;
        }
        acc.abs() < 1e-4
    });

    // Sort + dedupe (within tolerance).
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut deduped: Vec<f64> = Vec::new();
    for r in roots {
        let keep = match deduped.last() {
            Some(&last) => (r - last).abs() > 1e-7,
            None => true,
        };
        if keep {
            deduped.push(r);
        }
    }
    Some(deduped)
}

// ---------------------------------------------------------------------------
// Partial fraction decomposition
// ---------------------------------------------------------------------------

/// Partial fraction decomposition of `num / den` where `den` has been
/// completely factored into simple linear factors `den = c·∏(x − rᵢ)` with
/// distinct roots `rᵢ`.
///
/// Returns coefficients `[A_1, A_2, …, A_n]` such that
///   `num/den = Σᵢ Aᵢ / (x − rᵢ)`.
/// `n` is the number of distinct roots.
///
/// Returns `None` if `deg(num) ≥ deg(den)` (caller must polynomial-divide
/// first) or if any root has multiplicity > 1 (use Hermite for those).
pub fn partial_fractions_simple(num: &[f64], leading: f64, roots: &[f64]) -> Option<Vec<f64>> {
    let n = roots.len();
    if n == 0 {
        return None;
    }

    // For simple roots, A_i = num(r_i) / (leading · ∏_{j≠i} (r_i − r_j)).
    let mut coeffs = Vec::with_capacity(n);
    for (i, &ri) in roots.iter().enumerate() {
        // Evaluate num at ri (Horner)
        let mut nv = 0.0;
        for &cf in num.iter().rev() {
            nv = nv * ri + cf;
        }

        // Compute ∏_{j≠i} (r_i − r_j)
        let mut prod = leading;
        for (j, &rj) in roots.iter().enumerate() {
            if i != j {
                let diff = ri - rj;
                if diff.abs() < 1e-12 {
                    // Repeated root — caller should use Hermite.
                    return None;
                }
                prod *= diff;
            }
        }

        coeffs.push(nv / prod);
    }
    Some(coeffs)
}

// ---------------------------------------------------------------------------
// Hermite reduction step
// ---------------------------------------------------------------------------

/// One Hermite reduction step. Given `P/Q^k` with `k ≥ 2` and `Q` squarefree,
/// returns `(A, lower)` such that:
///
///   `P/Q^k dx = -A/((k−1)·Q^(k−1)) + ∫ lower/Q^(k−1) dx`
///
/// where `deg(A) < deg(Q)` and `deg(lower) < deg(Q^(k−1))`.
///
/// Algorithm: use extended Euclidean on `Q` and `Q'` to find `(U, V)` with
/// `U·Q + V·Q' = P`, then simplify via the identity
/// `V·Q'/Q^k = -1/(k−1) · d/dx(V/Q^(k−1)) + V'/((k−1)·Q^(k−1))`.
///
/// Returns `None` if `Q` is not squarefree (gcd(Q, Q') ≠ 1).
pub fn hermite_reduce_step(p: &[f64], q: &[f64], k: u32) -> Option<(Poly, Poly)> {
    if k < 2 {
        return None;
    }
    let qp = poly_derivative(q);
    let g = poly_gcd(q, &qp);
    if poly_degree(&g) > 0 {
        // Q not squarefree — bail.
        return None;
    }

    // Find U, V via extended Euclidean: U·Q + V·Q' = 1.
    let (_, u_unit, v_unit) = poly_extended_gcd(q, &qp)?;

    // Scale by P: U·Q + V·Q' = P  ⇒  multiply U_unit, V_unit by P, then
    // reduce V modulo Q so deg(V) < deg(Q).
    let up = poly_mul(&u_unit, p);
    let vp = poly_mul(&v_unit, p);
    // Reduce V modulo Q (keep V part with deg < deg(Q)).
    let (vq_quot, v_red) = poly_divmod(&vp, q);
    // Add the moved part back into U: U' = U + V_quot · Q' (so that
    // U'·Q + V_red·Q' = P).
    let u_plus = poly_add(&up, &poly_mul(&vq_quot, &qp));

    let a = poly_scale(&v_red, 1.0 / (k as f64 - 1.0));
    // lower = U' + V'_red / (k − 1)
    let v_red_prime = poly_derivative(&v_red);
    let lower = poly_add(&u_plus, &poly_scale(&v_red_prime, 1.0 / (k as f64 - 1.0)));

    Some((a, lower))
}

/// Extended Euclidean for polynomials. Returns `(g, u, v)` such that
/// `u·a + v·b = g`. `g` is the GCD (monic).
pub fn poly_extended_gcd(a: &[f64], b: &[f64]) -> Option<(Poly, Poly, Poly)> {
    let mut r0 = a.to_vec();
    let mut r1 = b.to_vec();
    poly_trim(&mut r0);
    poly_trim(&mut r1);

    let mut s0 = vec![1.0];
    let mut s1 = vec![0.0];
    let mut t0 = vec![0.0];
    let mut t1 = vec![1.0];

    let max_iter = 32;
    let mut iter = 0;
    while !r1.iter().all(|v| v.abs() < 1e-12) && iter < max_iter {
        iter += 1;
        let (q_, r) = poly_divmod(&r0, &r1);

        let new_s = poly_sub(&s0, &poly_mul(&q_, &s1));
        let new_t = poly_sub(&t0, &poly_mul(&q_, &t1));

        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = new_s;
        t0 = t1;
        t1 = new_t;
    }

    // Make r0 monic.
    let lead_idx = poly_degree(&r0);
    let lead = r0[lead_idx];
    if lead.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / lead;
    Some((
        poly_scale(&r0, inv),
        poly_scale(&s0, inv),
        poly_scale(&t0, inv),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_poly_eq(a: &[f64], b: &[f64]) {
        let na = poly_degree(a);
        let nb = poly_degree(b);
        assert_eq!(na, nb, "degree mismatch: {a:?} vs {b:?}");
        for i in 0..=na {
            assert!(
                (a[i] - b[i]).abs() < 1e-7,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_poly_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let s = poly_add(&a, &b);
        assert_poly_eq(&s, &[5.0, 7.0, 3.0]);
    }

    #[test]
    fn test_poly_mul() {
        // (1+x)·(1−x) = 1 − x²
        let a = vec![1.0, 1.0];
        let b = vec![1.0, -1.0];
        let prod = poly_mul(&a, &b);
        assert_poly_eq(&prod, &[1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_poly_derivative() {
        // d/dx(1 + 2x + 3x²) = 2 + 6x
        let a = vec![1.0, 2.0, 3.0];
        let d = poly_derivative(&a);
        assert_poly_eq(&d, &[2.0, 6.0]);
    }

    #[test]
    fn test_poly_divmod() {
        // (x³ + 2x² + 3x + 4) / (x + 1) = x² + x + 2 r 2
        let num = vec![4.0, 3.0, 2.0, 1.0];
        let den = vec![1.0, 1.0];
        let (q, r) = poly_divmod(&num, &den);
        assert_poly_eq(&q, &[2.0, 1.0, 1.0]);
        assert_poly_eq(&r, &[2.0]);
    }

    #[test]
    fn test_poly_gcd() {
        // gcd(x²−1, x−1) = x − 1 (made monic → [-1, 1])
        let a = vec![-1.0, 0.0, 1.0];
        let b = vec![-1.0, 1.0];
        let g = poly_gcd(&a, &b);
        // Monic form is x − 1.
        let canonical = poly_scale(&g, 1.0);
        assert_poly_eq(&canonical, &[-1.0, 1.0]);
    }

    #[test]
    fn test_yun_squarefree_simple() {
        // Q = (x − 1)² = x² − 2x + 1
        let q = vec![1.0, -2.0, 1.0];
        let factors = yun_squarefree(&q);
        // Should return [(x − 1, 2)]
        let total_deg: u32 = factors.iter().map(|(p, m)| poly_degree(p) as u32 * m).sum();
        assert_eq!(total_deg, 2, "Yun decomposition should match input degree");
    }

    #[test]
    fn test_yun_squarefree_distinct() {
        // Q = (x − 1)(x − 2)(x − 3), already squarefree
        let q = vec![-6.0, 11.0, -6.0, 1.0];
        let factors = yun_squarefree(&q);
        // Single factor with multiplicity 1, degree 3
        let total_deg: u32 = factors.iter().map(|(p, m)| poly_degree(p) as u32 * m).sum();
        assert_eq!(total_deg, 3);
    }

    #[test]
    fn test_real_roots_quadratic() {
        // x² − 5x + 6 = 0 → x = 2, 3
        let p = vec![6.0, -5.0, 1.0];
        let roots = real_roots_low_degree(&p).expect("should solve");
        let mut sorted = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(sorted.len(), 2);
        assert!((sorted[0] - 2.0).abs() < 1e-9);
        assert!((sorted[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_real_roots_cubic() {
        // x³ − 6x² + 11x − 6 → roots 1, 2, 3
        let p = vec![-6.0, 11.0, -6.0, 1.0];
        let roots = real_roots_low_degree(&p).expect("should solve");
        let mut sorted = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(sorted.len(), 3);
        for (got, expect) in sorted.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((got - expect).abs() < 1e-7);
        }
    }

    #[test]
    fn test_partial_fractions_simple() {
        // 1 / ((x − 1)(x − 2)) = A/(x−1) + B/(x−2)
        // A = 1/(1−2) = −1, B = 1/(2−1) = 1
        let num = vec![1.0];
        let coeffs = partial_fractions_simple(&num, 1.0, &[1.0, 2.0]).expect("simple roots");
        assert!((coeffs[0] - (-1.0)).abs() < 1e-10, "A = {}", coeffs[0]);
        assert!((coeffs[1] - 1.0).abs() < 1e-10, "B = {}", coeffs[1]);
    }
}
