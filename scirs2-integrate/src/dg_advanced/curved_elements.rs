//! Curved element mappings for isoparametric DG
//!
//! Provides isoparametric maps from reference triangle/quad elements to physical
//! elements, Jacobian computation, arc boundary parameterization, Gordon-Hall
//! blended transfinite interpolation, and high-order quadrature on curved elements.
//!
//! ## Reference element
//! The reference triangle is T̂ = {(ξ,η) : ξ≥0, η≥0, ξ+η≤1}.
//! Physical coordinates are obtained via an isoparametric map
//! x(ξ,η) = Σ_k x_k φ_k(ξ,η) where φ_k are Lagrange basis functions.

use super::types::CurvedElement;
use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// Lagrange basis functions on the reference triangle
// ─────────────────────────────────────────────────────────────────────────────

/// Lagrange basis nodes for the reference triangle at polynomial order p.
///
/// Returns equally spaced nodes on T̂ in the standard "warp-blend" pattern.
/// For order 1 (3 nodes): corners (0,0), (1,0), (0,1).
/// For order 2 (6 nodes): corners + edge midpoints.
/// For order 3 (10 nodes): corners + third-points + interior.
pub fn lagrange_nodes_triangle(order: usize) -> IntegrateResult<Vec<[f64; 2]>> {
    if order == 0 {
        return Ok(vec![[1.0 / 3.0, 1.0 / 3.0]]);
    }
    let p = order as f64;
    let mut nodes = Vec::new();
    for j in 0..=order {
        for i in 0..=(order - j) {
            let xi = i as f64 / p;
            let eta = j as f64 / p;
            nodes.push([xi, eta]);
        }
    }
    Ok(nodes)
}

/// Evaluate all Lagrange basis functions at (xi, eta) for a given order.
/// Returns a vector of length (p+1)(p+2)/2 of basis values.
pub fn lagrange_basis_triangle(xi: f64, eta: f64, order: usize) -> IntegrateResult<Vec<f64>> {
    let nodes = lagrange_nodes_triangle(order)?;
    let n = nodes.len();
    // Build Vandermonde-like evaluation via barycentric coordinates for order 1/2/3
    // For general order we use the product-form on the reference triangle
    // using the "collapsed coordinates" Proriol–Koornwinder–Dubiner (PKD) approach
    // For this implementation we use direct Lagrange interpolation for small orders
    // by computing basis functions as products of 1D Lagrange on the simplex.
    //
    // We implement the standard "warp-blend" formula: φ_k(ξ,η) = L_i(ξ/(1-η)) * L_j(η)
    // using the property of the collapsed reference triangle.
    // For simplicity (and correctness at low order), we use direct multi-variate Lagrange:
    //   φ_k(ξ,η) = Π_{l≠k} [(ξ-ξ_l, η-η_l)·(ξ_k-ξ_l, η-η_l)^{-1}]
    // but we need a stable formula. We use the direct formula with
    // two barycentric-like coordinates.

    if n == 1 {
        return Ok(vec![1.0]);
    }

    // Use direct formula: φ_k(p) = Π_{l≠k} dist(p, p_l) / dist(p_k, p_l)
    // This is NOT the standard Lagrange (which is for 1D polynomials),
    // but for the reference triangle the standard approach is as follows.
    //
    // For orders 1, 2, 3 we can use the barycentric formulation:
    //   λ_1 = 1 - ξ - η,  λ_2 = ξ,  λ_3 = η
    // and express basis in terms of these.
    let lam1 = 1.0 - xi - eta; // barycentric coord associated with (0,0)
    let lam2 = xi; // barycentric coord associated with (1,0)
    let lam3 = eta; // barycentric coord associated with (0,1)

    match order {
        1 => {
            // 3 nodes: (0,0), (1,0), (0,1)
            // φ_0 = λ_1, φ_1 = λ_2, φ_2 = λ_3
            Ok(vec![lam1, lam2, lam3])
        }
        2 => {
            // 6 nodes: (0,0),(1,0),(0,1),(0.5,0),(0.5,0.5),(0,0.5)
            // Serendipity-type quadratic basis:
            let phi0 = lam1 * (2.0 * lam1 - 1.0);
            let phi1 = lam2 * (2.0 * lam2 - 1.0);
            let phi2 = lam3 * (2.0 * lam3 - 1.0);
            let phi3 = 4.0 * lam1 * lam2;
            let phi4 = 4.0 * lam2 * lam3;
            let phi5 = 4.0 * lam1 * lam3;
            Ok(vec![phi0, phi1, phi2, phi3, phi4, phi5])
        }
        3 => {
            // 10 nodes: standard cubic basis on triangle
            // corner nodes
            let phi0 = lam1 * (3.0 * lam1 - 1.0) * (3.0 * lam1 - 2.0) / 2.0;
            let phi1 = lam2 * (3.0 * lam2 - 1.0) * (3.0 * lam2 - 2.0) / 2.0;
            let phi2 = lam3 * (3.0 * lam3 - 1.0) * (3.0 * lam3 - 2.0) / 2.0;
            // edge 1 (λ_1,λ_2): two midpoints
            let phi3 = 9.0 / 2.0 * lam1 * lam2 * (3.0 * lam1 - 1.0);
            let phi4 = 9.0 / 2.0 * lam1 * lam2 * (3.0 * lam2 - 1.0);
            // edge 2 (λ_2,λ_3): two midpoints
            let phi5 = 9.0 / 2.0 * lam2 * lam3 * (3.0 * lam2 - 1.0);
            let phi6 = 9.0 / 2.0 * lam2 * lam3 * (3.0 * lam3 - 1.0);
            // edge 3 (λ_3,λ_1): two midpoints
            let phi7 = 9.0 / 2.0 * lam3 * lam1 * (3.0 * lam3 - 1.0);
            let phi8 = 9.0 / 2.0 * lam3 * lam1 * (3.0 * lam1 - 1.0);
            // interior node
            let phi9 = 27.0 * lam1 * lam2 * lam3;
            Ok(vec![
                phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9,
            ])
        }
        _ => {
            // General order: use the product-form simplex Lagrange basis.
            // For multi-index (i,j) with k = third index = order-i-j:
            //   φ_{(i,j,k)}(λ₁,λ₂,λ₃) = ℓ_i^p(λ₁) · ℓ_j^p(λ₂) · ℓ_k^p(λ₃)
            // where ℓ_n^p(λ) = (1/n!) · ∏_{l=0}^{n-1} (p·λ − l), ℓ_0 = 1.
            //
            // Node enumeration: outer j in 0..=order, inner i in 0..=(order-j),
            // matching `lagrange_nodes_triangle`.
            lagrange_basis_triangle_general(lam1, lam2, lam3, order)
        }
    }
}

/// Evaluate the 1D simplex Lagrange factor ℓ_n^p(λ).
///
/// ℓ_n^p(λ) = (1/n!) · ∏_{l=0}^{n-1} (p·λ − l), with ℓ_0 = 1.
fn simplex_lagrange_1d(n: usize, p: f64, lam: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut val = 1.0_f64;
    for l in 0..n {
        val *= (p * lam - l as f64) / (l as f64 + 1.0);
    }
    val
}

/// Derivative of ℓ_n^p(λ) w.r.t. λ using the product rule.
///
/// d/dλ [ ℓ_n^p(λ) ] = Σ_{m=0}^{n-1} p/( m+1 ) · ∏_{l≠m} (p·λ − l) / (l+1)
/// which simplifies to (p/ℓ_n^p) · Σ_{m=0}^{n-1} ℓ_n^p / (p·λ − m).
/// We compute it directly as a sum of products to avoid division-by-zero.
fn simplex_lagrange_1d_deriv(n: usize, p: f64, lam: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    // Compute all factors (p·λ - l) / (l+1) first
    let factors: Vec<f64> = (0..n)
        .map(|l| (p * lam - l as f64) / (l as f64 + 1.0))
        .collect();

    // Derivative via logarithmic differentiation (avoiding zero issues):
    // d/dλ ℓ_n = Σ_{m=0}^{n-1} p/(m+1) · ∏_{l≠m} factors[l]
    let mut deriv = 0.0_f64;
    for m in 0..n {
        let mut term = p / (m as f64 + 1.0);
        for l in 0..n {
            if l != m {
                term *= factors[l];
            }
        }
        deriv += term;
    }
    deriv
}

/// General Lagrange basis on the reference triangle for arbitrary order p.
///
/// Returns n_basis = (p+1)(p+2)/2 values ordered as:
/// outer j in 0..=p, inner i in 0..=(p-j), k = p-i-j.
fn lagrange_basis_triangle_general(
    lam1: f64,
    lam2: f64,
    lam3: f64,
    order: usize,
) -> IntegrateResult<Vec<f64>> {
    let p = order as f64;
    let n_basis = (order + 1) * (order + 2) / 2;
    let mut phi = Vec::with_capacity(n_basis);

    for j in 0..=order {
        for i in 0..=(order - j) {
            let k = order - i - j;
            let val = simplex_lagrange_1d(i, p, lam2)
                * simplex_lagrange_1d(j, p, lam3)
                * simplex_lagrange_1d(k, p, lam1);
            phi.push(val);
        }
    }

    Ok(phi)
}

/// Derivatives of the Lagrange basis w.r.t. (ξ, η) on the reference triangle.
///
/// Returns (dphi_dxi, dphi_deta) each of length n_basis.
pub fn lagrange_basis_triangle_deriv(
    xi: f64,
    eta: f64,
    order: usize,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let lam1 = 1.0 - xi - eta;
    let lam2 = xi;
    let lam3 = eta;

    match order {
        0 => Ok((vec![0.0], vec![0.0])),
        1 => {
            // φ_0 = λ_1 = 1-ξ-η,  ∂/∂ξ = -1, ∂/∂η = -1
            // φ_1 = λ_2 = ξ,       ∂/∂ξ =  1, ∂/∂η =  0
            // φ_2 = λ_3 = η,       ∂/∂ξ =  0, ∂/∂η =  1
            Ok((vec![-1.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0]))
        }
        2 => {
            // Using chain rule on the quadratic basis
            // dλ_1/dξ = -1, dλ_1/dη = -1
            // dλ_2/dξ =  1, dλ_2/dη =  0
            // dλ_3/dξ =  0, dλ_3/dη =  1
            let d0_dxi = -(4.0 * lam1 - 1.0);
            let d0_deta = -(4.0 * lam1 - 1.0);
            let d1_dxi = 4.0 * lam2 - 1.0;
            let d1_deta = 0.0;
            let d2_dxi = 0.0;
            let d2_deta = 4.0 * lam3 - 1.0;
            let d3_dxi = 4.0 * (-lam2 + lam1 * 1.0);
            let d3_deta = -4.0 * lam2;
            let d4_dxi = 4.0 * lam3;
            let d4_deta = 4.0 * lam2;
            let d5_dxi = -4.0 * lam3;
            let d5_deta = 4.0 * (-lam3 + lam1);
            Ok((
                vec![d0_dxi, d1_dxi, d2_dxi, d3_dxi, d4_dxi, d5_dxi],
                vec![d0_deta, d1_deta, d2_deta, d3_deta, d4_deta, d5_deta],
            ))
        }
        3 => {
            // Cubic derivatives via chain rule; dλ/dξ and dλ/dη as above
            let dl1_dxi = -1.0_f64;
            let dl1_deta = -1.0_f64;
            let dl2_dxi = 1.0_f64;
            let dl2_deta = 0.0_f64;
            let dl3_dxi = 0.0_f64;
            let dl3_deta = 1.0_f64;

            let corner_d = |l: f64, dl_dxi: f64, dl_deta: f64| -> (f64, f64) {
                // d/dξ [ l*(3l-1)*(3l-2)/2 ] = (3l-1)*(3l-2)/2 * dl + l*(3)*(3l-2)/2 * dl + l*(3l-1)*(3)/2 * dl
                // simplify: d/dξ = dl * [ (3l-1)(3l-2)/2 + 3l(3l-2)/2 + 3l(3l-1)/2 ]
                //                = dl * (27l^2 - 18l + 2)/2
                let factor = (27.0 * l * l - 18.0 * l + 2.0) / 2.0;
                (dl_dxi * factor, dl_deta * factor)
            };

            let (d0_dxi, d0_deta) = corner_d(lam1, dl1_dxi, dl1_deta);
            let (d1_dxi, d1_deta) = corner_d(lam2, dl2_dxi, dl2_deta);
            let (d2_dxi, d2_deta) = corner_d(lam3, dl3_dxi, dl3_deta);

            // edge midpoint φ_3 = 9/2 * λ_1 * λ_2 * (3λ_1 - 1)
            // ∂/∂ξ = 9/2 * [ dλ1*λ2*(3λ1-1) + λ1*dλ2*(3λ1-1) + λ1*λ2*3*dλ1 ]
            let d3_dxi = 4.5
                * (dl1_dxi * lam2 * (3.0 * lam1 - 1.0)
                    + lam1 * dl2_dxi * (3.0 * lam1 - 1.0)
                    + lam1 * lam2 * 3.0 * dl1_dxi);
            let d3_deta = 4.5
                * (dl1_deta * lam2 * (3.0 * lam1 - 1.0)
                    + lam1 * dl2_deta * (3.0 * lam1 - 1.0)
                    + lam1 * lam2 * 3.0 * dl1_deta);

            // φ_4 = 9/2 * λ_1 * λ_2 * (3λ_2 - 1)
            let d4_dxi = 4.5
                * (dl1_dxi * lam2 * (3.0 * lam2 - 1.0)
                    + lam1 * dl2_dxi * (3.0 * lam2 - 1.0)
                    + lam1 * lam2 * 3.0 * dl2_dxi);
            let d4_deta = 4.5
                * (dl1_deta * lam2 * (3.0 * lam2 - 1.0)
                    + lam1 * dl2_deta * (3.0 * lam2 - 1.0)
                    + lam1 * lam2 * 3.0 * dl2_deta);

            // φ_5 = 9/2 * λ_2 * λ_3 * (3λ_2 - 1)
            let d5_dxi = 4.5
                * (dl2_dxi * lam3 * (3.0 * lam2 - 1.0)
                    + lam2 * dl3_dxi * (3.0 * lam2 - 1.0)
                    + lam2 * lam3 * 3.0 * dl2_dxi);
            let d5_deta = 4.5
                * (dl2_deta * lam3 * (3.0 * lam2 - 1.0)
                    + lam2 * dl3_deta * (3.0 * lam2 - 1.0)
                    + lam2 * lam3 * 3.0 * dl2_deta);

            // φ_6 = 9/2 * λ_2 * λ_3 * (3λ_3 - 1)
            let d6_dxi = 4.5
                * (dl2_dxi * lam3 * (3.0 * lam3 - 1.0)
                    + lam2 * dl3_dxi * (3.0 * lam3 - 1.0)
                    + lam2 * lam3 * 3.0 * dl3_dxi);
            let d6_deta = 4.5
                * (dl2_deta * lam3 * (3.0 * lam3 - 1.0)
                    + lam2 * dl3_deta * (3.0 * lam3 - 1.0)
                    + lam2 * lam3 * 3.0 * dl3_deta);

            // φ_7 = 9/2 * λ_3 * λ_1 * (3λ_3 - 1)
            let d7_dxi = 4.5
                * (dl3_dxi * lam1 * (3.0 * lam3 - 1.0)
                    + lam3 * dl1_dxi * (3.0 * lam3 - 1.0)
                    + lam3 * lam1 * 3.0 * dl3_dxi);
            let d7_deta = 4.5
                * (dl3_deta * lam1 * (3.0 * lam3 - 1.0)
                    + lam3 * dl1_deta * (3.0 * lam3 - 1.0)
                    + lam3 * lam1 * 3.0 * dl3_deta);

            // φ_8 = 9/2 * λ_3 * λ_1 * (3λ_1 - 1)
            let d8_dxi = 4.5
                * (dl3_dxi * lam1 * (3.0 * lam1 - 1.0)
                    + lam3 * dl1_dxi * (3.0 * lam1 - 1.0)
                    + lam3 * lam1 * 3.0 * dl1_dxi);
            let d8_deta = 4.5
                * (dl3_deta * lam1 * (3.0 * lam1 - 1.0)
                    + lam3 * dl1_deta * (3.0 * lam1 - 1.0)
                    + lam3 * lam1 * 3.0 * dl1_deta);

            // φ_9 = 27 * λ_1 * λ_2 * λ_3
            let d9_dxi =
                27.0 * (dl1_dxi * lam2 * lam3 + lam1 * dl2_dxi * lam3 + lam1 * lam2 * dl3_dxi);
            let d9_deta =
                27.0 * (dl1_deta * lam2 * lam3 + lam1 * dl2_deta * lam3 + lam1 * lam2 * dl3_deta);

            Ok((
                vec![
                    d0_dxi, d1_dxi, d2_dxi, d3_dxi, d4_dxi, d5_dxi, d6_dxi, d7_dxi, d8_dxi, d9_dxi,
                ],
                vec![
                    d0_deta, d1_deta, d2_deta, d3_deta, d4_deta, d5_deta, d6_deta, d7_deta,
                    d8_deta, d9_deta,
                ],
            ))
        }
        _ => {
            // General order: use product-rule on the simplex Lagrange factors.
            // φ_{(i,j,k)} = ℓ_i^p(λ₂) · ℓ_j^p(λ₃) · ℓ_k^p(λ₁), k = p-i-j.
            //
            // ∂φ/∂ξ = ∂φ/∂λ₁·(-1) + ∂φ/∂λ₂·(1) + ∂φ/∂λ₃·(0)
            // ∂φ/∂η = ∂φ/∂λ₁·(-1) + ∂φ/∂λ₂·(0) + ∂φ/∂λ₃·(1)
            lagrange_basis_triangle_deriv_general(lam1, lam2, lam3, order)
        }
    }
}

/// General Lagrange basis derivatives for arbitrary order p.
///
/// Returns (dphi_dxi, dphi_deta) each of length (p+1)(p+2)/2, enumerated
/// in the same (outer j, inner i) order as `lagrange_basis_triangle_general`.
fn lagrange_basis_triangle_deriv_general(
    lam1: f64,
    lam2: f64,
    lam3: f64,
    order: usize,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let p = order as f64;
    let n_basis = (order + 1) * (order + 2) / 2;
    let mut dphi_dxi = Vec::with_capacity(n_basis);
    let mut dphi_deta = Vec::with_capacity(n_basis);

    // Barycentric chain rule constants:
    //   dλ₁/dξ = -1,  dλ₂/dξ = 1,   dλ₃/dξ = 0
    //   dλ₁/dη = -1,  dλ₂/dη = 0,   dλ₃/dη = 1

    for j in 0..=order {
        for i in 0..=(order - j) {
            let k = order - i - j;

            let li = simplex_lagrange_1d(i, p, lam2); // ℓ_i^p(λ₂)
            let lj = simplex_lagrange_1d(j, p, lam3); // ℓ_j^p(λ₃)
            let lk = simplex_lagrange_1d(k, p, lam1); // ℓ_k^p(λ₁)

            let dli = simplex_lagrange_1d_deriv(i, p, lam2); // d/dλ₂
            let dlj = simplex_lagrange_1d_deriv(j, p, lam3); // d/dλ₃
            let dlk = simplex_lagrange_1d_deriv(k, p, lam1); // d/dλ₁

            // ∂φ/∂λ₁ = lk' · li · lj
            // ∂φ/∂λ₂ = li' · lj · lk
            // ∂φ/∂λ₃ = lj' · li · lk
            let dphi_dlam1 = dlk * li * lj;
            let dphi_dlam2 = dli * lj * lk;
            let dphi_dlam3 = dlj * li * lk;

            // Chain rule to (ξ, η)
            let d_dxi = -dphi_dlam1 + dphi_dlam2; // dphi_dlam3 * 0 omitted
            let d_deta = -dphi_dlam1 + dphi_dlam3; // dphi_dlam2 * 0 omitted

            dphi_dxi.push(d_dxi);
            dphi_deta.push(d_deta);
        }
    }

    Ok((dphi_dxi, dphi_deta))
}

// ─────────────────────────────────────────────────────────────────────────────
// Isoparametric mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map a reference-triangle point (ξ,η) to physical coordinates via an
/// isoparametric map of given `order`.
///
/// `nodes` must contain (order+1)(order+2)/2 physical node coordinates.
///
/// # Returns
/// Physical coordinates [x, y].
pub fn isoparametric_map(
    ref_pt: [f64; 2],
    nodes: &[[f64; 2]],
    order: usize,
) -> IntegrateResult<[f64; 2]> {
    let phi = lagrange_basis_triangle(ref_pt[0], ref_pt[1], order)?;
    if phi.len() != nodes.len() {
        return Err(IntegrateError::DimensionMismatch(format!(
            "isoparametric_map: {} basis functions but {} nodes",
            phi.len(),
            nodes.len()
        )));
    }
    let mut x = 0.0;
    let mut y = 0.0;
    for (k, &p) in phi.iter().enumerate() {
        x += p * nodes[k][0];
        y += p * nodes[k][1];
    }
    Ok([x, y])
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobian computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Jacobian matrix of the isoparametric map at reference point (ξ,η).
///
/// J = [[∂x/∂ξ, ∂x/∂η],
///      [∂y/∂ξ, ∂y/∂η]]
pub fn jacobian(
    ref_pt: [f64; 2],
    nodes: &[[f64; 2]],
    order: usize,
) -> IntegrateResult<[[f64; 2]; 2]> {
    let (dphi_dxi, dphi_deta) = lagrange_basis_triangle_deriv(ref_pt[0], ref_pt[1], order)?;
    if dphi_dxi.len() != nodes.len() {
        return Err(IntegrateError::DimensionMismatch(format!(
            "jacobian: {} derivatives but {} nodes",
            dphi_dxi.len(),
            nodes.len()
        )));
    }
    let mut dx_dxi = 0.0;
    let mut dx_deta = 0.0;
    let mut dy_dxi = 0.0;
    let mut dy_deta = 0.0;
    for k in 0..nodes.len() {
        dx_dxi += dphi_dxi[k] * nodes[k][0];
        dx_deta += dphi_deta[k] * nodes[k][0];
        dy_dxi += dphi_dxi[k] * nodes[k][1];
        dy_deta += dphi_deta[k] * nodes[k][1];
    }
    Ok([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
}

/// Compute the determinant of the 2×2 Jacobian matrix.
pub fn det_jacobian(j: &[[f64; 2]; 2]) -> f64 {
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

/// Compute the inverse of the 2×2 Jacobian matrix.
///
/// Returns an error if the Jacobian is singular (|det J| < ε).
pub fn inv_jacobian(j: &[[f64; 2]; 2]) -> IntegrateResult<[[f64; 2]; 2]> {
    let det = det_jacobian(j);
    if det.abs() < 1e-14 {
        return Err(IntegrateError::ValueError(
            "Jacobian is singular (det ≈ 0)".into(),
        ));
    }
    let inv_det = 1.0 / det;
    Ok([
        [j[1][1] * inv_det, -j[0][1] * inv_det],
        [-j[1][0] * inv_det, j[0][0] * inv_det],
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Curved boundary elements
// ─────────────────────────────────────────────────────────────────────────────

/// Parameterize a circular arc boundary edge.
///
/// Given parameter t ∈ [0, 1], returns the physical point on the arc of
/// radius `r` centred at `center`, sweeping from angle `theta_start` to
/// `theta_end` linearly in t.
pub fn arc_boundary_map(
    t: f64,
    center: [f64; 2],
    r: f64,
    theta_start: f64,
    theta_end: f64,
) -> [f64; 2] {
    let theta = theta_start + t * (theta_end - theta_start);
    [center[0] + r * theta.cos(), center[1] + r * theta.sin()]
}

/// Gordon-Hall transfinite interpolation for a quadrilateral element.
///
/// Maps reference coordinates (ξ, η) ∈ [-1,1]² to physical coordinates
/// by blending the four boundary edge maps.
///
/// `edges` encodes the four corners in the order:
///   [bottom-left, bottom-right, top-right, top-left]
/// i.e., the bilinear blending uses:
///   x(ξ,η) = (1-ξ)(1-η)/4 * x00 + (1+ξ)(1-η)/4 * x10
///           + (1+ξ)(1+η)/4 * x11 + (1-ξ)(1+η)/4 * x01
pub fn blended_transfinite_interpolation(xi: f64, eta: f64, edges: &[[f64; 2]; 4]) -> [f64; 2] {
    // Map reference square [-1,1]^2 to [0,1]^2
    let s = (xi + 1.0) * 0.5; // s ∈ [0,1]
    let t = (eta + 1.0) * 0.5; // t ∈ [0,1]

    // Bilinear interpolation: corners in order (0,0), (1,0), (1,1), (0,1)
    // edges[0] = (s=0,t=0), edges[1] = (s=1,t=0), edges[2] = (s=1,t=1), edges[3] = (s=0,t=1)
    let w00 = (1.0 - s) * (1.0 - t);
    let w10 = s * (1.0 - t);
    let w11 = s * t;
    let w01 = (1.0 - s) * t;

    [
        w00 * edges[0][0] + w10 * edges[1][0] + w11 * edges[2][0] + w01 * edges[3][0],
        w00 * edges[0][1] + w10 * edges[1][1] + w11 * edges[2][1] + w01 * edges[3][1],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauss-Legendre quadrature on the reference triangle
// ─────────────────────────────────────────────────────────────────────────────

/// Gauss-Legendre quadrature points and weights on the reference triangle T̂.
///
/// Uses the Dunavant rules for n_points = 1, 3, 4, 6, 7.
/// Returns (points, weights) where points are in barycentric (ξ,η) coordinates
/// and weights sum to 1/2 (area of T̂).
fn gauss_triangle_dunavant(n: usize) -> IntegrateResult<(Vec<[f64; 2]>, Vec<f64>)> {
    match n {
        1 => {
            // 1-point rule, exact for degree 1
            Ok((vec![[1.0 / 3.0, 1.0 / 3.0]], vec![0.5]))
        }
        3 => {
            // 3-point rule (Dunavant degree 2)
            let a = 1.0 / 6.0;
            let b = 2.0 / 3.0;
            let w = 1.0 / 6.0;
            Ok((vec![[a, a], [b, a], [a, b]], vec![w, w, w]))
        }
        4 => {
            // 4-point rule (degree 3)
            let a1 = 1.0 / 3.0;
            let a2 = 1.0 / 5.0;
            let b2 = 3.0 / 5.0;
            let w1 = -27.0 / 96.0;
            let w2 = 25.0 / 96.0;
            Ok((
                vec![[a1, a1], [a2, a2], [b2, a2], [a2, b2]],
                vec![w1, w2, w2, w2],
            ))
        }
        6 => {
            // 6-point rule (degree 4)
            let a1 = 0.445948490915965;
            let b1 = 0.108103018168070;
            let a2 = 0.091576213509771;
            let b2 = 0.816847572980459;
            let w1 = 0.111690794839005;
            let w2 = 0.054975871827661;
            Ok((
                vec![[a1, a1], [b1, a1], [a1, b1], [a2, a2], [b2, a2], [a2, b2]],
                vec![w1, w1, w1, w2, w2, w2],
            ))
        }
        _ => {
            // Fall back to a tensor-product Gauss-Legendre rule on the triangle
            // via the Duffy transformation: (r,s) ∈ [0,1]^2 → (ξ=r, η=(1-r)*s)
            // Jacobian factor: (1-r)
            let n_1d = n.max(2);
            let (gl_nodes, gl_weights) = gauss_legendre_1d(n_1d);
            // Transform from [-1,1] to [0,1]: x_01 = (x+1)/2, w_01 = w/2
            let mut pts = Vec::new();
            let mut wts = Vec::new();
            for (i, &r_ref) in gl_nodes.iter().enumerate() {
                let r = (r_ref + 1.0) * 0.5;
                let wr = gl_weights[i] * 0.5;
                for (j, &s_ref) in gl_nodes.iter().enumerate() {
                    let s = (s_ref + 1.0) * 0.5;
                    let ws = gl_weights[j] * 0.5;
                    let xi_pt = r;
                    let eta_pt = (1.0 - r) * s;
                    let jac = (1.0 - r).max(0.0);
                    pts.push([xi_pt, eta_pt]);
                    wts.push(wr * ws * jac);
                }
            }
            Ok((pts, wts))
        }
    }
}

/// Compute Gauss-Legendre nodes and weights on [-1, 1] using the Golub-Welsch
/// algorithm (tridiagonal eigenvalue problem) or a hard-coded table for small n.
pub fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.5773502691896257, 0.5773502691896257],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![
                0.5555555555555556,
                0.888_888_888_888_889,
                0.5555555555555556,
            ],
        ),
        4 => (
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        ),
        6 => (
            vec![
                -0.932_469_514_203_152,
                -0.6612093864662645,
                -0.2386191860831969,
                0.2386191860831969,
                0.6612093864662645,
                0.932_469_514_203_152,
            ],
            vec![
                0.1713244923791704,
                0.3607615730481386,
                0.467_913_934_572_691,
                0.467_913_934_572_691,
                0.3607615730481386,
                0.1713244923791704,
            ],
        ),
        7 => (
            vec![
                -0.9491079123427585,
                -0.7415311855993945,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993945,
                0.9491079123427585,
            ],
            vec![
                0.1294849661688697,
                0.2797053914892767,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892767,
                0.1294849661688697,
            ],
        ),
        _ => {
            // For n > 7: use Newton-Raphson to find zeros of P_n(x)
            // Initial guesses: x_k = cos(π * (k + 0.75) / (n + 0.5)) (Stroud & Secrest)
            gauss_legendre_newton(n)
        }
    }
}

/// Compute GL nodes and weights for arbitrary n using Newton-Raphson on P_n.
///
/// Uses the Stroud-Secrest initial guess and the three-term recurrence for P_n.
/// Weights are w_i = 2 / ((1 - x_i²) [P'_n(x_i)]²).
fn gauss_legendre_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    let nf = n as f64;

    // Only need to compute n/2 nodes due to symmetry
    let n_half = n.div_ceil(2);

    for k in 0..n_half {
        // Initial guess (Stroud-Secrest approximation)
        let theta = std::f64::consts::PI * (k as f64 + 0.75) / (nf + 0.5);
        let mut x = -theta.cos(); // node in (-1, 0] for k < n/2

        // Newton-Raphson iteration
        for _ in 0..100 {
            // Evaluate P_n(x) and P'_n(x) via three-term recurrence
            let mut p0 = 1.0_f64;
            let mut p1 = x;
            for j in 2..=n {
                let jf = j as f64;
                let p2 = ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf;
                p0 = p1;
                p1 = p2;
            }
            // P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
            let pn = p1;
            let pn1 = p0;
            let dpn = if (x * x - 1.0).abs() > 1e-12 {
                nf * (x * pn - pn1) / (x * x - 1.0)
            } else {
                // Limit at x = ±1
                let s = if x > 0.0 {
                    1.0
                } else {
                    (-1_f64).powi(n as i32 + 1)
                };
                s * nf * (nf + 1.0) * 0.5
            };

            if dpn.abs() < 1e-300 {
                break;
            }
            let dx = -pn / dpn;
            x += dx;
            if dx.abs() < 1e-15 * x.abs().max(1.0) {
                break;
            }
        }

        // Evaluate P_n(x) and P'_n(x) for weight computation
        let mut p0 = 1.0_f64;
        let mut p1 = x;
        for j in 2..=n {
            let jf = j as f64;
            let p2 = ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf;
            p0 = p1;
            p1 = p2;
        }
        let pn = p1;
        let pn1 = p0;
        let dpn = if (x * x - 1.0).abs() > 1e-12 {
            nf * (x * pn - pn1) / (x * x - 1.0)
        } else {
            let s = if x > 0.0 {
                1.0
            } else {
                (-1_f64).powi(n as i32 + 1)
            };
            s * nf * (nf + 1.0) * 0.5
        };

        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);

        // Use symmetry to fill both ends
        let mirror = n - 1 - k;
        nodes[k] = -x;
        nodes[mirror] = x;
        weights[k] = w;
        weights[mirror] = w;
    }

    (nodes, weights)
}

// ─────────────────────────────────────────────────────────────────────────────
// Curved quadrature
// ─────────────────────────────────────────────────────────────────────────────

/// Compute quadrature points and weights on a curved element.
///
/// Maps reference triangle Dunavant quadrature points to physical coordinates
/// via the element's isoparametric map, incorporating the Jacobian determinant.
///
/// Returns (physical_points, weights) where weights include |det J|.
pub fn curved_quad_points_weights(
    element: &CurvedElement,
    n: usize,
) -> IntegrateResult<(Vec<[f64; 2]>, Vec<f64>)> {
    // Determine mapping order from number of vertices: n_nodes = (p+1)(p+2)/2
    let order = match element.vertices.len() {
        1 => 0,
        3 => 1,
        6 => 2,
        10 => 3,
        15 => 4,
        21 => 5,
        n => {
            // Try to infer order for any valid triangular node count
            let mut inferred = None;
            for p in 6..=20 {
                if (p + 1) * (p + 2) / 2 == n {
                    inferred = Some(p);
                    break;
                }
            }
            match inferred {
                Some(p) => p,
                None => {
                    return Err(IntegrateError::ValueError(format!(
                        "curved_quad_points_weights: unexpected vertex count {}",
                        n
                    )))
                }
            }
        }
    };

    let (ref_pts, ref_wts) = gauss_triangle_dunavant(n)?;

    let mut phys_pts = Vec::with_capacity(ref_pts.len());
    let mut phys_wts = Vec::with_capacity(ref_pts.len());

    for (i, &rp) in ref_pts.iter().enumerate() {
        let phys = isoparametric_map(rp, &element.vertices, order)?;
        let j = jacobian(rp, &element.vertices, order)?;
        let det = det_jacobian(&j).abs();
        phys_pts.push(phys);
        phys_wts.push(ref_wts[i] * det);
    }

    Ok((phys_pts, phys_wts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_isoparametric_map_linear() {
        // Linear map: reference triangle corners should map to physical corners
        let v0 = [0.0, 0.0];
        let v1 = [2.0, 0.0];
        let v2 = [0.0, 3.0];
        let nodes = vec![v0, v1, v2];

        // Corner (0,0) → v0
        let p0 = isoparametric_map([0.0, 0.0], &nodes, 1).unwrap();
        assert!((p0[0] - v0[0]).abs() < 1e-12 && (p0[1] - v0[1]).abs() < 1e-12);

        // Corner (1,0) → v1
        let p1 = isoparametric_map([1.0, 0.0], &nodes, 1).unwrap();
        assert!((p1[0] - v1[0]).abs() < 1e-12 && (p1[1] - v1[1]).abs() < 1e-12);

        // Corner (0,1) → v2
        let p2 = isoparametric_map([0.0, 1.0], &nodes, 1).unwrap();
        assert!((p2[0] - v2[0]).abs() < 1e-12 && (p2[1] - v2[1]).abs() < 1e-12);
    }

    #[test]
    fn test_jacobian_constant() {
        // Linear element has constant Jacobian
        let v0 = [0.0, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0];
        let nodes = vec![v0, v1, v2];

        let j0 = jacobian([0.0, 0.0], &nodes, 1).unwrap();
        let j1 = jacobian([0.5, 0.0], &nodes, 1).unwrap();
        let j2 = jacobian([0.25, 0.25], &nodes, 1).unwrap();

        for (i, row) in j0.iter().enumerate() {
            for (k, &val) in row.iter().enumerate() {
                assert!((val - j1[i][k]).abs() < 1e-12);
                assert!((val - j2[i][k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_det_jacobian_positive() {
        let v0 = [0.0, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0];
        let nodes = vec![v0, v1, v2];
        let j = jacobian([1.0 / 3.0, 1.0 / 3.0], &nodes, 1).unwrap();
        let det = det_jacobian(&j);
        assert!(
            det > 0.0,
            "Jacobian determinant should be positive for CCW orientation"
        );
    }

    #[test]
    fn test_arc_boundary_parameterization() {
        let center = [0.0, 0.0];
        let r = 2.0;
        // Points on the arc should lie on the circle
        for i in 0..10 {
            let t = i as f64 / 9.0;
            let pt = arc_boundary_map(t, center, r, 0.0, PI / 2.0);
            let dist = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
            assert!((dist - r).abs() < 1e-12, "Point should lie on circle");
        }
    }

    #[test]
    fn test_blended_transfinite_corners() {
        let edges: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // At (ξ=-1, η=-1) → edges[0]
        let p00 = blended_transfinite_interpolation(-1.0, -1.0, &edges);
        assert!((p00[0] - edges[0][0]).abs() < 1e-12);
        assert!((p00[1] - edges[0][1]).abs() < 1e-12);

        // At (ξ=1, η=-1) → edges[1]
        let p10 = blended_transfinite_interpolation(1.0, -1.0, &edges);
        assert!((p10[0] - edges[1][0]).abs() < 1e-12);
        assert!((p10[1] - edges[1][1]).abs() < 1e-12);

        // At (ξ=1, η=1) → edges[2]
        let p11 = blended_transfinite_interpolation(1.0, 1.0, &edges);
        assert!((p11[0] - edges[2][0]).abs() < 1e-12);
        assert!((p11[1] - edges[2][1]).abs() < 1e-12);

        // At (ξ=-1, η=1) → edges[3]
        let p01 = blended_transfinite_interpolation(-1.0, 1.0, &edges);
        assert!((p01[0] - edges[3][0]).abs() < 1e-12);
        assert!((p01[1] - edges[3][1]).abs() < 1e-12);
    }

    #[test]
    fn test_curved_quad_points_count() {
        let element = CurvedElement::linear_triangle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
        let (pts, wts) = curved_quad_points_weights(&element, 3).unwrap();
        assert_eq!(pts.len(), 3, "3-point rule should yield 3 points");
        assert_eq!(wts.len(), 3);
    }

    // ─── General high-order Lagrange basis tests ───────────────────────────

    /// Verify node counts for orders 4 and 5.
    #[test]
    fn test_lagrange_basis_node_count_high_order() {
        for order in 4..=5 {
            let nodes = lagrange_nodes_triangle(order).unwrap();
            let expected = (order + 1) * (order + 2) / 2;
            assert_eq!(nodes.len(), expected, "order {order}: wrong node count");

            let phi = lagrange_basis_triangle(0.25, 0.25, order).unwrap();
            assert_eq!(phi.len(), expected, "order {order}: wrong basis count");

            let (dx, dy) = lagrange_basis_triangle_deriv(0.25, 0.25, order).unwrap();
            assert_eq!(dx.len(), expected, "order {order}: wrong deriv x count");
            assert_eq!(dy.len(), expected, "order {order}: wrong deriv y count");
        }
    }

    /// Lagrange property: φ_k(nodes[l]) = δ_{kl} for orders 4 and 5.
    #[test]
    fn test_lagrange_property_order4() {
        let order = 4;
        let nodes = lagrange_nodes_triangle(order).unwrap();
        let n = nodes.len();
        for l in 0..n {
            let [xi, eta] = nodes[l];
            let phi = lagrange_basis_triangle(xi, eta, order).unwrap();
            for k in 0..n {
                let expected = if k == l { 1.0 } else { 0.0 };
                assert!(
                    (phi[k] - expected).abs() < 1e-10,
                    "order 4: φ_{k}(node[{l}]) = {}, expected {expected}",
                    phi[k]
                );
            }
        }
    }

    #[test]
    fn test_lagrange_property_order5() {
        let order = 5;
        let nodes = lagrange_nodes_triangle(order).unwrap();
        let n = nodes.len();
        for l in 0..n {
            let [xi, eta] = nodes[l];
            let phi = lagrange_basis_triangle(xi, eta, order).unwrap();
            for k in 0..n {
                let expected = if k == l { 1.0 } else { 0.0 };
                assert!(
                    (phi[k] - expected).abs() < 1e-10,
                    "order 5: φ_{k}(node[{l}]) = {}, expected {expected}",
                    phi[k]
                );
            }
        }
    }

    /// Partition of unity: Σ φ_k(ξ,η) = 1 at interior points.
    #[test]
    fn test_partition_of_unity_high_order() {
        let test_pts = [[0.1, 0.2], [1.0 / 3.0, 1.0 / 3.0], [0.5, 0.1], [0.2, 0.5]];
        for order in 4..=5 {
            for &[xi, eta] in &test_pts {
                let phi = lagrange_basis_triangle(xi, eta, order).unwrap();
                let sum: f64 = phi.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "order {order} at ({xi},{eta}): partition of unity = {sum}"
                );
            }
        }
    }

    /// Sum of derivatives = 0: Σ ∂φ_k/∂ξ = 0 and Σ ∂φ_k/∂η = 0.
    #[test]
    fn test_derivatives_sum_zero_high_order() {
        let test_pts = [[0.1, 0.2], [1.0 / 3.0, 1.0 / 3.0], [0.4, 0.3]];
        for order in 4..=5 {
            for &[xi, eta] in &test_pts {
                let (dxi, deta) = lagrange_basis_triangle_deriv(xi, eta, order).unwrap();
                let sum_xi: f64 = dxi.iter().sum();
                let sum_eta: f64 = deta.iter().sum();
                assert!(
                    sum_xi.abs() < 1e-11,
                    "order {order} at ({xi},{eta}): sum dφ/dξ = {sum_xi}"
                );
                assert!(
                    sum_eta.abs() < 1e-11,
                    "order {order} at ({xi},{eta}): sum dφ/dη = {sum_eta}"
                );
            }
        }
    }

    /// Finite-difference check of derivatives at an interior point for order 4.
    #[test]
    fn test_derivative_finite_difference_order4() {
        let order = 4;
        let xi = 0.3;
        let eta = 0.25;
        let h = 1e-6;

        let phi_p = lagrange_basis_triangle(xi + h, eta, order).unwrap();
        let phi_m = lagrange_basis_triangle(xi - h, eta, order).unwrap();
        let (dxi_exact, _) = lagrange_basis_triangle_deriv(xi, eta, order).unwrap();

        for k in 0..dxi_exact.len() {
            let fd = (phi_p[k] - phi_m[k]) / (2.0 * h);
            assert!(
                (fd - dxi_exact[k]).abs() < 1e-7,
                "order 4 FD dφ_{k}/dξ: fd={fd}, exact={}",
                dxi_exact[k]
            );
        }

        let phi_p_eta = lagrange_basis_triangle(xi, eta + h, order).unwrap();
        let phi_m_eta = lagrange_basis_triangle(xi, eta - h, order).unwrap();
        let (_, deta_exact) = lagrange_basis_triangle_deriv(xi, eta, order).unwrap();

        for k in 0..deta_exact.len() {
            let fd = (phi_p_eta[k] - phi_m_eta[k]) / (2.0 * h);
            assert!(
                (fd - deta_exact[k]).abs() < 1e-7,
                "order 4 FD dφ_{k}/dη: fd={fd}, exact={}",
                deta_exact[k]
            );
        }
    }

    /// Isoparametric map corners are exact for order 4 on a flat triangle.
    #[test]
    fn test_isoparametric_map_order4_corners() {
        let order = 4;
        // Flat (linear) triangle: all nodes are on the linear map of the reference triangle
        // The physical corners
        let x0 = [0.0_f64, 0.0];
        let x1 = [3.0, 0.0];
        let x2 = [0.0, 2.0];

        // Build n_nodes = 15 nodes from the isoparametric mapping of a linear element:
        // Each reference node (ξ_l, η_l) maps to x0*(1-ξ-η) + x1*ξ + x2*η
        let ref_nodes = lagrange_nodes_triangle(order).unwrap();
        let nodes: Vec<[f64; 2]> = ref_nodes
            .iter()
            .map(|&[xi, eta]| {
                let l1 = 1.0 - xi - eta;
                [
                    l1 * x0[0] + xi * x1[0] + eta * x2[0],
                    l1 * x0[1] + xi * x1[1] + eta * x2[1],
                ]
            })
            .collect();

        // Corner (0,0) → x0
        let p = isoparametric_map([0.0, 0.0], &nodes, order).unwrap();
        assert!((p[0] - x0[0]).abs() < 1e-11 && (p[1] - x0[1]).abs() < 1e-11);

        // Corner (1,0) → x1
        let p = isoparametric_map([1.0, 0.0], &nodes, order).unwrap();
        assert!((p[0] - x1[0]).abs() < 1e-11 && (p[1] - x1[1]).abs() < 1e-11);

        // Corner (0,1) → x2
        let p = isoparametric_map([0.0, 1.0], &nodes, order).unwrap();
        assert!((p[0] - x2[0]).abs() < 1e-11 && (p[1] - x2[1]).abs() < 1e-11);
    }

    /// curved_quad_points_weights works for order-4 element (15 nodes).
    #[test]
    fn test_curved_quad_order4_flat_area() {
        let order = 4;
        let x0 = [0.0_f64, 0.0];
        let x1 = [1.0, 0.0];
        let x2 = [0.0, 1.0];

        let ref_nodes = lagrange_nodes_triangle(order).unwrap();
        let vertices: Vec<[f64; 2]> = ref_nodes
            .iter()
            .map(|&[xi, eta]| {
                let l1 = 1.0 - xi - eta;
                [
                    l1 * x0[0] + xi * x1[0] + eta * x2[0],
                    l1 * x0[1] + xi * x1[1] + eta * x2[1],
                ]
            })
            .collect();

        use super::super::types::GeometricMap;
        let element = CurvedElement::new(vertices, GeometricMap::Isoparametric);
        let (_, wts) = curved_quad_points_weights(&element, 6).unwrap();
        let area: f64 = wts.iter().sum();
        // Area of unit right triangle = 0.5
        assert!(
            (area - 0.5).abs() < 1e-10,
            "Order-4 element area = {area}, expected 0.5"
        );
    }
}
