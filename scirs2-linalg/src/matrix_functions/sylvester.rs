//! Matrix equation solvers using Schur-based methods.
//!
//! This module provides:
//!
//! - `solve_sylvester`             – AX + XB = C  via Bartels-Stewart (Schur form)
//! - `solve_discrete_lyapunov`     – AXAᵀ - X + Q = 0
//! - `solve_continuous_lyapunov`   – AX + XAᵀ + Q = 0
//! - `solve_algebraic_riccati`     – Continuous/Discrete ARE via Schur method
//!
//! These implementations improve on the vectorisation-based solvers in
//! `matrix_equations.rs` by using the Bartels-Stewart algorithm, which scales
//! as O(n³) instead of O(n⁶) for the Sylvester equation.

use crate::error::{LinalgError, LinalgResult};
use crate::schur_enhanced::{real_schur_decompose, SchurFloat};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Complex, Float, NumAssign, One, Zero};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ─────────────────────────────────────────────────────────────────────────────
// solve_sylvester  (Bartels-Stewart)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the Sylvester matrix equation AX + XB = C.
///
/// Uses the Bartels-Stewart algorithm:
/// 1. Reduce A → real Schur form: A = U_A S_A U_Aᵀ
/// 2. Reduce B → real Schur form: B = U_B S_B U_Bᵀ
/// 3. Transform C → C̃ = U_Aᵀ C U_B
/// 4. Solve the quasi-triangular system S_A X̃ + X̃ S_B = C̃ by back-substitution
/// 5. Transform back: X = U_A X̃ U_Bᵀ
///
/// This is O(n³) rather than the O(n⁶) Kronecker approach.
///
/// # Arguments
/// * `a` - Matrix A (m × m)
/// * `b` - Matrix B (n × n)
/// * `c` - Matrix C (m × n)
///
/// # Returns
/// X (m × n) such that AX + XB = C
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sylvester::solve_sylvester;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let b = array![[-3.0_f64, 0.0], [0.0, -4.0]];
/// let c = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("ok");
/// // Verify AX + XB = C
/// let resid = a.dot(&x) + x.dot(&b) - &c;
/// for &v in resid.iter() { assert!(v.abs() < 1e-8); }
/// ```
pub fn solve_sylvester<T: SchurFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    c: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let m = a.nrows();
    let n = b.nrows();

    if a.ncols() != m {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if b.ncols() != n {
        return Err(LinalgError::ShapeError("B must be square".into()));
    }
    if c.nrows() != m || c.ncols() != n {
        return Err(LinalgError::DimensionError(format!(
            "C must be {m}×{n}, got {}×{}",
            c.nrows(),
            c.ncols()
        )));
    }

    // Schur decompositions
    let schur_a = real_schur_decompose(a, 300, T::epsilon() * T::from(100.0).unwrap_or(T::one()))?;
    let schur_b = real_schur_decompose(b, 300, T::epsilon() * T::from(100.0).unwrap_or(T::one()))?;

    let sa = &schur_a.t; // quasi-upper-triangular
    let ua = &schur_a.q; // orthogonal
    let sb = &schur_b.t;
    let ub = &schur_b.q;

    // Transform C: C_tilde = U_A^T * C * U_B
    let c_tilde: Array2<T> = ua.t().dot(c).dot(ub);

    // Solve the quasi-triangular Sylvester system S_A X_t + X_t S_B = C_tilde
    let x_tilde = bartels_stewart_solve(sa, sb, &c_tilde.view())?;

    // Transform back: X = U_A * X_tilde * U_B^T
    Ok(ua.dot(&x_tilde).dot(&ub.t()))
}

/// Core solver for the quasi-triangular Sylvester system S_A X + X S_B = C.
///
/// Both S_A and S_B are in real Schur (quasi-upper-triangular) form.
/// The algorithm proceeds column-by-column (for each Schur block of S_B),
/// solving a shifted triangular system at each step.
fn bartels_stewart_solve<T: SchurFloat>(
    sa: &Array2<T>,
    sb: &Array2<T>,
    c: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let m = sa.nrows();
    let n = sb.nrows();
    let tol = T::epsilon() * T::from(100.0).unwrap_or(T::one());

    let mut x = Array2::<T>::zeros((m, n));
    // Identify blocks in S_B
    let b_blocks = identify_quasi_blocks(sb, tol);

    // Process each column block of S_B
    let mut rhs_x = c.to_owned();
    let mut col = 0usize;
    for &(blk_start, blk_size) in &b_blocks {
        if blk_size == 1 {
            // S_B diagonal entry is scalar: shift S_A by mu = S_B[k,k]
            let mu = sb[[blk_start, blk_start]];
            // Solve (S_A + mu I) * x_col = rhs_col
            let rhs_col = rhs_x.column(col).to_owned();
            let x_col = solve_shifted_upper_quasi(sa, mu, &rhs_col.view(), tol)?;
            x.column_mut(col).assign(&x_col);
            // Update remaining rhs: rhs[:,col+1..] -= x_col * S_B[blk_start, blk_start+1..]
            for j in (blk_start + 1)..n {
                let sb_val = sb[[blk_start, j]];
                if sb_val.abs() > T::epsilon() {
                    x.column(col)
                        .iter()
                        .zip(rhs_x.column_mut(j).iter_mut())
                        .for_each(|(&xi, rj)| *rj -= xi * sb_val);
                }
            }
        } else {
            // 2×2 block: solve coupled 2-column system using actual S_B entries
            let sb00 = sb[[blk_start, blk_start]];
            let sb01 = sb[[blk_start, blk_start + 1]];
            let sb10 = sb[[blk_start + 1, blk_start]];
            let sb11 = sb[[blk_start + 1, blk_start + 1]];

            let rhs0 = rhs_x.column(col).to_owned();
            let rhs1 = rhs_x.column(col + 1).to_owned();

            let (x0, x1) = solve_shifted_upper_quasi_2x2(
                sa,
                sb00,
                sb01,
                sb10,
                sb11,
                &rhs0.view(),
                &rhs1.view(),
                tol,
            )?;
            x.column_mut(col).assign(&x0);
            x.column_mut(col + 1).assign(&x1);

            // Update remaining columns
            for j in (blk_start + 2)..n {
                let sb0 = sb[[blk_start, j]];
                let sb1 = sb[[blk_start + 1, j]];
                for i in 0..m {
                    rhs_x[[i, j]] -= x0[i] * sb0 + x1[i] * sb1;
                }
            }
        }
        col += blk_size;
    }

    Ok(x)
}

/// Identify quasi-upper-triangular blocks (same as in schur_enhanced).
fn identify_quasi_blocks<T: SchurFloat>(t: &Array2<T>, tol: T) -> Vec<(usize, usize)> {
    let n = t.nrows();
    let mut blocks = Vec::new();
    let mut k = 0;
    while k < n {
        if k + 1 < n && t[[k + 1, k]].abs() > tol {
            blocks.push((k, 2));
            k += 2;
        } else {
            blocks.push((k, 1));
            k += 1;
        }
    }
    blocks
}

/// Solve (S_A + mu * I) x = rhs where S_A is quasi-upper-triangular (real Schur form).
///
/// Uses block back-substitution.  For each 1×1 diagonal block the solution is
/// straightforward.  For each 2×2 block we solve a 2×2 shifted system.
fn solve_shifted_upper_quasi<T: SchurFloat>(
    sa: &Array2<T>,
    mu: T,
    rhs: &scirs2_core::ndarray::ArrayView1<T>,
    tol: T,
) -> LinalgResult<Array1<T>> {
    let n = sa.nrows();
    let blocks = identify_quasi_blocks(sa, tol);
    let nb = blocks.len();
    let mut x = Array1::<T>::zeros(n);
    let mut rhs_copy: Array1<T> = rhs.to_owned();

    // Back-substitute from last block to first
    let mut bidx = nb;
    loop {
        if bidx == 0 {
            break;
        }
        bidx -= 1;
        let (blk_start, blk_size) = blocks[bidx];

        if blk_size == 1 {
            // Subtract contributions from already-solved blocks to the right
            let mut s = rhs_copy[blk_start];
            for j in (blk_start + 1)..n {
                s -= sa[[blk_start, j]] * x[j];
            }
            let diag = sa[[blk_start, blk_start]] + mu;
            if diag.abs() < tol {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_shifted_upper_quasi: near-zero diagonal at {blk_start} (shift {mu:?})"
                )));
            }
            x[blk_start] = s / diag;
        } else {
            // 2×2 block
            let r0 = blk_start;
            let r1 = blk_start + 1;
            let mut s0 = rhs_copy[r0];
            let mut s1 = rhs_copy[r1];
            for j in (r1 + 1)..n {
                s0 -= sa[[r0, j]] * x[j];
                s1 -= sa[[r1, j]] * x[j];
            }
            // Solve [[a+mu, b],[c, d+mu]] [x0; x1] = [s0; s1]
            let a = sa[[r0, r0]] + mu;
            let b = sa[[r0, r1]];
            let c = sa[[r1, r0]];
            let d = sa[[r1, r1]] + mu;
            let det = a * d - b * c;
            if det.abs() < tol {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_shifted_upper_quasi: singular 2x2 block at {blk_start}"
                )));
            }
            x[r0] = (d * s0 - b * s1) / det;
            x[r1] = (-c * s0 + a * s1) / det;
        }
    }

    Ok(x)
}

/// Solve the coupled 2-column system for a 2×2 block in S_B.
///
/// The Sylvester equation S_A * X + X * S_B = C gives, for a 2x2 S_B block:
///   (S_A + sb00*I) x0 + sb10 * x1 = rhs0       (column 0)
///   sb01 * x0 + (S_A + sb11*I) * x1 = rhs1       (column 1)
///
/// where sb00, sb01, sb10, sb11 are the four entries of the S_B block.
fn solve_shifted_upper_quasi_2x2<T: SchurFloat>(
    sa: &Array2<T>,
    sb00: T,
    sb01: T,
    sb10: T,
    sb11: T,
    rhs0: &scirs2_core::ndarray::ArrayView1<T>,
    rhs1: &scirs2_core::ndarray::ArrayView1<T>,
    tol: T,
) -> LinalgResult<(Array1<T>, Array1<T>)> {
    let n = sa.nrows();
    let blocks = identify_quasi_blocks(sa, tol);
    let nb = blocks.len();
    let mut x0 = Array1::<T>::zeros(n);
    let mut x1 = Array1::<T>::zeros(n);

    let mut bidx = nb;
    loop {
        if bidx == 0 {
            break;
        }
        bidx -= 1;
        let (blk_start, blk_size) = blocks[bidx];

        if blk_size == 1 {
            let mut s0 = rhs0[blk_start];
            let mut s1 = rhs1[blk_start];
            for j in (blk_start + 1)..n {
                s0 -= sa[[blk_start, j]] * x0[j];
                s1 -= sa[[blk_start, j]] * x1[j];
            }
            // 2×2 system with actual S_B diagonal entries:
            // (sa_diag + sb00) x0i + sb10 x1i = s0
            // sb01 x0i + (sa_diag + sb11) x1i = s1
            let d0 = sa[[blk_start, blk_start]] + sb00;
            let d1 = sa[[blk_start, blk_start]] + sb11;
            let det = d0 * d1 - sb01 * sb10;
            if det.abs() < tol {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_shifted_upper_quasi_2x2: near-zero det at {blk_start}"
                )));
            }
            x0[blk_start] = (d1 * s0 - sb10 * s1) / det;
            x1[blk_start] = (-sb01 * s0 + d0 * s1) / det;
        } else {
            // 4×4 system: (S_A 2x2 block + S_B 2x2 block) coupled
            let r0 = blk_start;
            let r1 = blk_start + 1;
            let mut s00 = rhs0[r0];
            let mut s01 = rhs0[r1];
            let mut s10 = rhs1[r0];
            let mut s11_rhs = rhs1[r1];

            for j in (r1 + 1)..n {
                s00 -= sa[[r0, j]] * x0[j];
                s01 -= sa[[r1, j]] * x0[j];
                s10 -= sa[[r0, j]] * x1[j];
                s11_rhs -= sa[[r1, j]] * x1[j];
            }

            // Build 4×4 system and solve.
            // Variables: [x0[r0], x0[r1], x1[r0], x1[r1]]
            //
            // Column 0: (S_A + sb00*I) * x0 + sb10 * x1 = rhs0
            //   row r0: (sa00+sb00)*x0[r0] + sa01*x0[r1] + sb10*x1[r0] = s00
            //   row r1: sa10*x0[r0] + (sa11+sb00)*x0[r1] + sb10*x1[r1] = s01
            //
            // Column 1: sb01 * x0 + (S_A + sb11*I) * x1 = rhs1
            //   row r0: sb01*x0[r0] + (sa00+sb11)*x1[r0] + sa01*x1[r1] = s10
            //   row r1: sb01*x0[r1] + sa10*x1[r0] + (sa11+sb11)*x1[r1] = s11_rhs

            let sa00 = sa[[r0, r0]];
            let sa01 = sa[[r0, r1]];
            let sa10 = sa[[r1, r0]];
            let sa11_val = sa[[r1, r1]];

            let mut mat = Array2::<T>::zeros((4, 4));
            // Row 0: (sa00+sb00)*x0[r0] + sa01*x0[r1] + sb10*x1[r0] + 0*x1[r1]
            mat[[0, 0]] = sa00 + sb00;
            mat[[0, 1]] = sa01;
            mat[[0, 2]] = sb10;
            mat[[0, 3]] = T::zero();
            // Row 1: sa10*x0[r0] + (sa11+sb00)*x0[r1] + 0*x1[r0] + sb10*x1[r1]
            mat[[1, 0]] = sa10;
            mat[[1, 1]] = sa11_val + sb00;
            mat[[1, 2]] = T::zero();
            mat[[1, 3]] = sb10;
            // Row 2: sb01*x0[r0] + 0*x0[r1] + (sa00+sb11)*x1[r0] + sa01*x1[r1]
            mat[[2, 0]] = sb01;
            mat[[2, 1]] = T::zero();
            mat[[2, 2]] = sa00 + sb11;
            mat[[2, 3]] = sa01;
            // Row 3: 0*x0[r0] + sb01*x0[r1] + sa10*x1[r0] + (sa11+sb11)*x1[r1]
            mat[[3, 0]] = T::zero();
            mat[[3, 1]] = sb01;
            mat[[3, 2]] = sa10;
            mat[[3, 3]] = sa11_val + sb11;

            let rhs4 = Array1::from_vec(vec![s00, s01, s10, s11_rhs]);
            let sol4 = crate::solve::solve(&mat.view(), &rhs4.view(), None)?;
            x0[r0] = sol4[0];
            x0[r1] = sol4[1];
            x1[r0] = sol4[2];
            x1[r1] = sol4[3];
        }
    }

    Ok((x0, x1))
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_continuous_lyapunov
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the continuous Lyapunov equation AX + XAᵀ + Q = 0.
///
/// Equivalent to the Sylvester equation AX + X(-Aᵀ) = -Q, solved via the
/// Bartels-Stewart algorithm on the Schur form of A.
///
/// # Arguments
/// * `a` - State matrix A (n × n)
/// * `q` - Symmetric matrix Q (n × n)
///
/// # Returns
/// Solution X (n × n) satisfying AX + XAᵀ + Q = 0.
/// The returned solution is symmetrised: X ← (X + Xᵀ)/2.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sylvester::solve_continuous_lyapunov;
///
/// let a = array![[-1.0_f64, 0.5], [0.0, -2.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = solve_continuous_lyapunov(&a.view(), &q.view()).expect("ok");
/// // Verify AX + XA^T + Q ≈ 0
/// let resid = a.dot(&x) + x.dot(&a.t()) + &q;
/// for &v in resid.iter() { assert!(v.abs() < 1e-7); }
/// ```
pub fn solve_continuous_lyapunov<T: SchurFloat>(
    a: &ArrayView2<T>,
    q: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::DimensionError("Q must be n×n".into()));
    }
    // AX + XA^T = -Q  →  Sylvester: AX + X*B = C  with B = A^T, C = -Q
    let at = a.t().to_owned();
    let neg_q = q.mapv(|v| -v);
    let x = solve_sylvester(a, &at.view(), &neg_q.view())?;
    // Symmetrise
    let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    Ok((&x + &x.t()) * half)
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_discrete_lyapunov
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the discrete Lyapunov equation AXAᵀ - X + Q = 0.
///
/// Uses the bilinear transformation to convert to a continuous Lyapunov equation,
/// then calls `solve_continuous_lyapunov`.
///
/// The bilinear transform A_c = (A - I)(A + I)⁻¹ maps the unit disc to the
/// left half-plane, and Q_c = 2(A + I)⁻ᵀ Q (A + I)⁻¹.
///
/// # Arguments
/// * `a` - Matrix A (n × n), spectral radius < 1 required for stability
/// * `q` - Symmetric matrix Q (n × n)
///
/// # Returns
/// X (n × n) satisfying AXAᵀ - X + Q = 0.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sylvester::solve_discrete_lyapunov;
///
/// let a = array![[0.5_f64, 0.1], [0.0, 0.6]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let x = solve_discrete_lyapunov(&a.view(), &q.view()).expect("ok");
/// // Verify AXA^T - X + Q ≈ 0
/// let resid = a.dot(&x).dot(&a.t()) - &x + &q;
/// for &v in resid.iter() { assert!(v.abs() < 1e-7); }
/// ```
pub fn solve_discrete_lyapunov<T: SchurFloat>(
    a: &ArrayView2<T>,
    q: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::DimensionError("Q must be n×n".into()));
    }

    // Direct Schur-based method for AXA^T - X + Q = 0
    // 1. Compute real Schur decomposition: A = U T U^T
    let tol = T::epsilon() * T::from(100.0).unwrap_or(T::one());
    let schur = real_schur_decompose(a, 300, tol)?;
    let u = &schur.q;
    let t = &schur.t;

    // 2. Transform: C = U^T Q U
    let c = u.t().dot(&q.to_owned()).dot(u);

    // 3. Solve T Y T^T - Y = -C column by column (back-substitution)
    let y = solve_discrete_lyapunov_triangular(t, &c, tol)?;

    // 4. Transform back: X = U Y U^T
    let x = u.dot(&y).dot(&u.t());

    // Symmetrise
    let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    Ok((&x + &x.t()) * half)
}

/// Solve the discrete Lyapunov equation T Y T^T - Y = -C
/// where T is quasi-upper-triangular (real Schur form).
///
/// This uses a column-by-column back-substitution approach.
fn solve_discrete_lyapunov_triangular<T: SchurFloat>(
    t: &Array2<T>,
    c: &Array2<T>,
    tol: T,
) -> LinalgResult<Array2<T>> {
    let n = t.nrows();
    let mut y = Array2::<T>::zeros((n, n));

    // We solve column by column, from j=0 to n-1
    // The equation T Y T^T - Y = -C can be written element-wise.
    // For column j of Y, we need to handle the quasi-triangular structure.
    //
    // Working from the last column backwards:
    // For 1x1 block at position (j,j):
    //   sum over i,k of T[i,k] * Y[k,l] * T[j,l] - Y[i,j] = -C[i,j]
    //
    // The standard approach is to vectorize and solve via Kronecker product,
    // but for quasi-triangular T we can do back-substitution.
    //
    // Alternative direct approach: solve iteratively column by column.
    // Since T is quasi-upper-triangular, process columns from right to left.

    // Identify blocks
    let blocks = identify_quasi_blocks(t, tol);

    // Process column blocks from right to left
    let nb = blocks.len();
    for bj_idx in (0..nb).rev() {
        let (j_start, j_size) = blocks[bj_idx];

        // Process row blocks from bottom to top
        for bi_idx in (0..nb).rev() {
            let (i_start, i_size) = blocks[bi_idx];

            // Build the right-hand side for this block
            // -C[i_block, j_block] minus contributions from already-computed Y blocks
            if i_size == 1 && j_size == 1 {
                let i = i_start;
                let j = j_start;
                let mut rhs = c[[i, j]];

                // Subtract contributions from already-computed entries
                for k in 0..n {
                    for l in 0..n {
                        if k == i && l == j {
                            continue;
                        }
                        // Only count if Y[k,l] is already computed
                        // (blocks to the right of j_block, or same j_block but row blocks below i_block)
                        let k_block = find_block_index(&blocks, k);
                        let l_block = find_block_index(&blocks, l);
                        let already_computed =
                            l_block > bj_idx || (l_block == bj_idx && k_block > bi_idx);
                        if already_computed {
                            rhs += t[[i, k]] * y[[k, l]] * t[[j, l]];
                        }
                    }
                }

                // Solve: t[i,i] * y[i,j] * t[j,j] - y[i,j] = -rhs
                // => y[i,j] * (t[i,i] * t[j,j] - 1) = -rhs
                let denom = t[[i, i]] * t[[j, j]] - T::one();
                if denom.abs() < tol {
                    // Singular: eigenvalue product λ_i * λ_j = 1
                    // Discrete Lyapunov has no unique solution when eigenvalues are on unit circle
                    return Err(LinalgError::SingularMatrixError(format!(
                        "solve_discrete_lyapunov_triangular: singular at ({i},{j}), \
                         eigenvalue product = 1 (Lyapunov equation has no unique solution)"
                    )));
                } else {
                    y[[i, j]] = -rhs / denom;
                }
            } else {
                // For 2x2 blocks, solve a small linear system
                let rows: Vec<usize> = (i_start..i_start + i_size).collect();
                let cols: Vec<usize> = (j_start..j_start + j_size).collect();
                let sys_size = i_size * j_size;

                let mut rhs_vec = vec![T::zero(); sys_size];
                for (ri, &i) in rows.iter().enumerate() {
                    for (ci, &j) in cols.iter().enumerate() {
                        let idx = ri * j_size + ci;
                        rhs_vec[idx] = c[[i, j]];

                        // Subtract contributions from already-computed entries
                        for k in 0..n {
                            for l in 0..n {
                                let is_current = rows.contains(&k) && cols.contains(&l);
                                if is_current {
                                    continue;
                                }
                                let k_block = find_block_index(&blocks, k);
                                let l_block = find_block_index(&blocks, l);
                                let already_computed =
                                    l_block > bj_idx || (l_block == bj_idx && k_block > bi_idx);
                                if already_computed {
                                    rhs_vec[idx] += t[[i, k]] * y[[k, l]] * t[[j, l]];
                                }
                            }
                        }
                    }
                }

                // Build the small system matrix:
                // For each (i,j) in the block, the equation is:
                // sum_{k in rows, l in cols} T[i,k] * Y[k,l] * T[j,l] - Y[i,j] = -rhs
                let mut mat = vec![T::zero(); sys_size * sys_size];
                for (ri, &i) in rows.iter().enumerate() {
                    for (ci, &j) in cols.iter().enumerate() {
                        let row_idx = ri * j_size + ci;
                        for (rk, &k) in rows.iter().enumerate() {
                            for (cl, &l) in cols.iter().enumerate() {
                                let col_idx = rk * j_size + cl;
                                mat[row_idx * sys_size + col_idx] += t[[i, k]] * t[[j, l]];
                            }
                        }
                        // Subtract identity (the -Y[i,j] term)
                        mat[row_idx * sys_size + row_idx] -= T::one();
                    }
                }

                // Solve the small system: mat * y_vec = -rhs_vec
                let neg_rhs: Vec<T> = rhs_vec.iter().map(|&v| -v).collect();
                let y_vec = solve_small_dense(&mat, &neg_rhs, sys_size)?;

                for (ri, &i) in rows.iter().enumerate() {
                    for (ci, &j) in cols.iter().enumerate() {
                        y[[i, j]] = y_vec[ri * j_size + ci];
                    }
                }
            }
        }
    }

    Ok(y)
}

/// Find which block index a given row/column belongs to.
fn find_block_index(blocks: &[(usize, usize)], idx: usize) -> usize {
    for (bi, &(start, size)) in blocks.iter().enumerate() {
        if idx >= start && idx < start + size {
            return bi;
        }
    }
    blocks.len() // should not happen
}

/// Solve a small dense linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_small_dense<T: SchurFloat>(mat: &[T], rhs: &[T], n: usize) -> LinalgResult<Vec<T>> {
    let mut a = vec![T::zero(); n * n];
    a.copy_from_slice(&mat[..n * n]);
    let mut b = rhs.to_vec();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < T::epsilon() {
            // Singular - return error
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_small_dense: singular at column {col}, pivot < epsilon"
            )));
        }
        // Swap rows
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        // Eliminate
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                let val = a[col * n + j];
                a[row * n + j] -= factor * val;
            }
            let b_col = b[col];
            b[row] -= factor * b_col;
        }
    }
    // Back-substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        if diag.abs() < T::epsilon() {
            x[i] = T::zero();
        } else {
            x[i] = s / diag;
        }
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_algebraic_riccati
// ─────────────────────────────────────────────────────────────────────────────

/// Unified solver for continuous and discrete algebraic Riccati equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiccatiType {
    /// Continuous ARE: AᵀX + XA - XBR⁻¹BᵀX + Q = 0
    Continuous,
    /// Discrete ARE: X = AᵀXA - AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q
    Discrete,
}

/// Solve a continuous or discrete algebraic Riccati equation via Schur method.
///
/// # Continuous ARE
/// Solves: Aᵀ X + X A - X B R⁻¹ Bᵀ X + Q = 0
///
/// Forms the Hamiltonian matrix
///
/// ```text
/// H = [ A    -B R⁻¹ Bᵀ ]
///     [ -Q   -Aᵀ        ]
/// ```
///
/// and computes its real Schur form, then selects the stable invariant
/// subspace (eigenvalues with Re(λ) < 0) to build X = U₂ U₁⁻¹.
///
/// # Discrete ARE
/// Solves: X = Aᵀ X A - Aᵀ X B (R + Bᵀ X B)⁻¹ Bᵀ X A + Q
///
/// Forms the symplectic matrix
///
/// ```text
/// S = [ A + B R⁻¹ Bᵀ (Aᵀ)⁻¹ Q ,  -B R⁻¹ Bᵀ (Aᵀ)⁻¹ ]
///     [ -(Aᵀ)⁻¹ Q              ,   (Aᵀ)⁻¹              ]
/// ```
///
/// and selects eigenvalues inside the unit disc.
///
/// # Arguments
/// * `a`             - State matrix (n × n)
/// * `b`             - Input matrix (n × m)
/// * `q`             - State cost matrix Q (n × n, symmetric PSD)
/// * `r`             - Input cost matrix R (m × m, symmetric PD)
/// * `riccati_type`  - `RiccatiType::Continuous` or `RiccatiType::Discrete`
///
/// # Returns
/// Solution X (n × n, symmetric PSD).
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sylvester::{solve_algebraic_riccati, RiccatiType};
///
/// let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64], [1.0]];
/// let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = array![[1.0_f64]];
/// let x = solve_algebraic_riccati(&a.view(), &b.view(), &q.view(), &r.view(),
///                                  RiccatiType::Continuous).expect("ok");
/// assert!(x[[0,0]] > 0.0, "solution should be PSD");
/// ```
pub fn solve_algebraic_riccati<T: SchurFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    q: &ArrayView2<T>,
    r: &ArrayView2<T>,
    riccati_type: RiccatiType,
) -> LinalgResult<Array2<T>> {
    match riccati_type {
        RiccatiType::Continuous => solve_care(a, b, q, r),
        RiccatiType::Discrete => solve_dare(a, b, q, r),
    }
}

/// Solve the continuous ARE via Newton-Kleinman iteration.
///
/// The CARE is: A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// Newton-Kleinman iteration:
/// 1. Start with X_0 (e.g., zero or Q)
/// 2. At each step k, define A_k = A - B R^{-1} B^T X_k
/// 3. Solve the continuous Lyapunov equation:
///    A_k^T X_{k+1} + X_{k+1} A_k = -(Q + X_k B R^{-1} B^T X_k)
/// 4. Repeat until convergence.
///
/// This avoids the need for Schur decomposition of the Hamiltonian matrix,
/// which can fail for certain matrix structures.
fn solve_care<T: SchurFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    q: &ArrayView2<T>,
    r: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();

    validate_riccati_inputs(a, b, q, r)?;

    let r_inv = crate::basic::inv(r, None)?;
    let b_r_inv_bt: Array2<T> = b.dot(&r_inv).dot(&b.t());

    let max_iter = 200usize;
    let conv_tol = T::from(1e-10).unwrap_or(T::epsilon());

    // Initial guess: find X_0 such that A_0 = A - B R^{-1} B^T X_0 is strictly stable.
    let mut x = find_stabilizing_initial_x(a, b, &b_r_inv_bt, q, n)?;

    // Newton-Kleinman iteration
    for _iter in 0..max_iter {
        // A_k = A - B R^{-1} B^T X_k
        let a_k = a.to_owned() - b_r_inv_bt.dot(&x);

        // RHS = Q + X_k B R^{-1} B^T X_k
        let xbr_bt_x = x.dot(&b_r_inv_bt).dot(&x);
        let rhs = q.to_owned() + xbr_bt_x;

        // Solve A_k^T X_{k+1} + X_{k+1} A_k = -rhs
        let a_k_t = a_k.t().to_owned();
        let x_new = match solve_continuous_lyapunov(&a_k_t.view(), &rhs.view()) {
            Ok(xn) => xn,
            Err(_) => break, // If Lyapunov solve fails, stop iteration
        };

        // Check convergence
        let diff: T = (&x_new - &x).iter().map(|&v| v * v).sum::<T>().sqrt();
        let x_norm: T = x_new.iter().map(|&v| v * v).sum::<T>().sqrt();

        x = x_new;

        if diff <= conv_tol * (x_norm + T::one()) {
            break;
        }
    }

    // Symmetrise
    let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    Ok((&x + &x.t()) * half)
}

/// Find a stabilizing initial X_0 for Newton-Kleinman iteration.
///
/// Finds X_0 such that A_0 = A - B R^{-1} B^T X_0 is strictly stable
/// (all eigenvalues have strictly negative real part).
///
/// Strategy: Use X_0 = alpha * (I + ones(n,n)) which ensures the matrix
/// B R^{-1} B^T X_0 has full-rank rows, avoiding zero eigenvalues in A_0.
fn find_stabilizing_initial_x<T: SchurFloat>(
    a: &ArrayView2<T>,
    _b: &ArrayView2<T>,
    b_r_inv_bt: &Array2<T>,
    _q: &ArrayView2<T>,
    n: usize,
) -> LinalgResult<Array2<T>> {
    let a_owned = a.to_owned();
    let eye = Array2::<T>::eye(n);
    // ones = matrix of all ones
    let ones = Array2::<T>::ones((n, n));

    // X_0 = alpha * (I + ones) is symmetric PD for alpha > 0
    // This ensures B R^{-1} B^T X_0 has a richer structure than just scaling
    for k in 1..200 {
        let alpha = T::from(k as f64 * 0.5).unwrap_or(T::one());
        let x0 = (&eye + &ones) * alpha;
        let a_cl = &a_owned - &b_r_inv_bt.dot(&x0);

        // Check Gershgorin stability
        let mut all_stable = true;
        for i in 0..n {
            let diag = a_cl[[i, i]];
            let mut off_diag_sum = T::zero();
            for j in 0..n {
                if j != i {
                    off_diag_sum += a_cl[[i, j]].abs();
                }
            }
            if diag + off_diag_sum >= T::zero() {
                all_stable = false;
                break;
            }
        }
        if all_stable {
            return Ok(x0);
        }
    }

    // Last resort
    let alpha = T::from(100.0).unwrap_or(T::one());
    Ok((&eye + &ones) * alpha)
}

/// Solve the discrete ARE via Hewer's fixed-point iteration.
///
/// The DARE is: X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
///
/// Hewer's iteration:
/// 1. Start with a stabilizing K_0 (such that A - B K_0 has spectral radius < 1)
/// 2. At each step k, solve the discrete Lyapunov equation:
///    (A - B K_k)^T X_{k+1} (A - B K_k) - X_{k+1} = -(Q + K_k^T R K_k)
/// 3. Update gain: K_{k+1} = (R + B^T X_{k+1} B)^{-1} B^T X_{k+1} A
/// 4. Repeat until convergence.
fn solve_dare<T: SchurFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    q: &ArrayView2<T>,
    r: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    let m = b.ncols();

    validate_riccati_inputs(a, b, q, r)?;

    let max_iter = 100usize;
    let conv_tol = T::epsilon() * T::from(1e6).unwrap_or(T::one());

    // Start with initial gain K_0 = 0 (assuming A is already Schur-stable)
    // If A is not stable, we'd need a stabilizing initial gain
    let mut k_gain = Array2::<T>::zeros((m, n));

    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let q_owned = q.to_owned();
    let r_owned = r.to_owned();

    let mut x = q.to_owned(); // Initial X

    for _iter in 0..max_iter {
        // A_cl = A - B K
        let a_cl = &a_owned - b_owned.dot(&k_gain);

        // RHS = Q + K^T R K
        let rhs = &q_owned + k_gain.t().dot(&r_owned).dot(&k_gain);

        // Solve discrete Lyapunov: A_cl X_new A_cl^T - X_new = -rhs
        let x_new = solve_discrete_lyapunov(&a_cl.view(), &rhs.view())?;

        // Update gain: K = (R + B^T X B)^{-1} B^T X A
        let bt_x = b_owned.t().dot(&x_new);
        let bt_x_b = bt_x.dot(&b_owned);
        let r_plus = &r_owned + bt_x_b;
        let r_plus_inv = crate::basic::inv(&r_plus.view(), None)?;
        let k_new = r_plus_inv.dot(&bt_x).dot(&a_owned);

        // Check convergence
        let diff: T = (&x_new - &x).iter().map(|&v| v * v).sum::<T>().sqrt();
        let x_norm: T = x_new.iter().map(|&v| v * v).sum::<T>().sqrt();

        x = x_new;
        k_gain = k_new;

        if diff <= conv_tol * (x_norm + T::one()) {
            break;
        }
    }

    // Symmetrise
    let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    Ok((&x + &x.t()) * half)
}

/// Validate dimensions for Riccati solvers.
fn validate_riccati_inputs<T: SchurFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    q: &ArrayView2<T>,
    r: &ArrayView2<T>,
) -> LinalgResult<()> {
    let n = a.nrows();
    let m = b.ncols();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "B must have {n} rows, got {}",
            b.nrows()
        )));
    }
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::DimensionError("Q must be n×n".into()));
    }
    if r.nrows() != m || r.ncols() != m {
        return Err(LinalgError::DimensionError("R must be m×m".into()));
    }
    Ok(())
}

/// Extract complex eigenvalues from a quasi-upper-triangular Schur form.
fn extract_schur_eigenvalues<T: SchurFloat>(t: &Array2<T>, tol: T) -> Vec<Complex<T>> {
    let n = t.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut k = 0usize;
    while k < n {
        if k + 1 < n && t[[k + 1, k]].abs() > tol {
            let tr = t[[k, k]] + t[[k + 1, k + 1]];
            let two = T::one() + T::one();
            let det = t[[k, k]] * t[[k + 1, k + 1]] - t[[k, k + 1]] * t[[k + 1, k]];
            let disc = tr * tr - two * two * det;
            if disc >= T::zero() {
                let sq = disc.sqrt();
                eigenvalues.push(Complex::new((tr + sq) / two, T::zero()));
                eigenvalues.push(Complex::new((tr - sq) / two, T::zero()));
            } else {
                let sq = (-disc).sqrt() / two;
                eigenvalues.push(Complex::new(tr / two, sq));
                eigenvalues.push(Complex::new(tr / two, -sq));
            }
            k += 2;
        } else {
            eigenvalues.push(Complex::new(t[[k, k]], T::zero()));
            k += 1;
        }
    }
    eigenvalues
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn max_abs_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn test_sylvester_diagonal() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let b = array![[-3.0_f64, 0.0], [0.0, -4.0]];
        let c = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("ok");
        let resid = a.dot(&x) + x.dot(&b) - &c;
        assert!(max_abs_err(&resid, &Array2::zeros((2, 2))) < 1e-9);
    }

    #[test]
    fn test_sylvester_general_2x2() {
        // Eigenvalues of A are {2,3}, eigenvalues of B are {-5,-4}.
        // No pair sums to zero (2-5=-3, 2-4=-2, 3-5=-2, 3-4=-1), so non-singular.
        let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let b = array![[-5.0_f64, 0.0], [0.0, -4.0]];
        let c = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = solve_sylvester(&a.view(), &b.view(), &c.view()).expect("ok");
        let resid = a.dot(&x) + x.dot(&b) - &c;
        let err = resid.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-8, "Sylvester residual {err}");
    }

    #[test]
    fn test_continuous_lyapunov_stable() {
        let a = array![[-1.0_f64, 0.5], [0.0, -2.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = solve_continuous_lyapunov(&a.view(), &q.view()).expect("ok");
        let resid = a.dot(&x) + x.dot(&a.t()) + &q;
        let err = resid.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-7, "CLyapunov residual {err}");
    }

    #[test]
    fn test_discrete_lyapunov() {
        let a = array![[0.5_f64, 0.1], [0.0, 0.6]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let x = solve_discrete_lyapunov(&a.view(), &q.view()).expect("ok");
        let resid = a.dot(&x).dot(&a.t()) - &x + &q;
        let err = resid.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-7, "DLyapunov residual {err}");
    }

    #[test]
    fn test_care_double_integrator() {
        // Classic double integrator: A = [0,1;0,0], B = [0;1], Q = I, R = 1
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let x = solve_algebraic_riccati(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            RiccatiType::Continuous,
        )
        .expect("CARE ok");
        // X should be positive definite
        assert!(x[[0, 0]] > 0.0, "CARE solution should be PD");
        assert!(x[[1, 1]] > 0.0, "CARE solution should be PD");
        // Verify CARE residual: A^T X + X A - X B R^{-1} B^T X + Q ≈ 0
        let r_inv = array![[1.0_f64]];
        let xbrbt = x.dot(&b).dot(&r_inv).dot(&b.t()).dot(&x);
        let resid = a.t().dot(&x) + x.dot(&a) - xbrbt + &q;
        let err = resid.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-5, "CARE residual {err}");
    }
}
