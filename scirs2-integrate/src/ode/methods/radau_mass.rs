//! Radau method with mass matrix support
//!
//! This module implements the Radau IIA method (3-stage, order 5) for solving ODEs
//! with support for mass matrices of the form M(t,y)·y' = f(t,y).
//!
//! The canonical K-parameterization is used: K_i are the stage derivatives
//! (i.e., K_i = y'(t + c_i * h)), so:
//!
//!   Y_i = y_n + h * Σ_j a_ij * K_j   (stage values)
//!   M(t + c_i*h, Y_i) * K_i = f(t + c_i*h, Y_i)  (DAE constraint per stage)
//!   y_{n+1} = y_n + h * Σ_j b_j * K_j  (solution update, b = last row of A for Radau IIA)
//!
//! Newton system (simplified Hairer-Wanner form):
//!   F(K) = [(I_3 ⊗ M) - h*(A ⊗ J)] * ΔK = -R(K)
//! where R_i(K) = M * K_i - f(Y_i), M and J evaluated once at (t_n, y_n).

use crate::error::IntegrateResult;
use crate::ode::types::{MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEResult};
use crate::ode::utils::common::calculate_error_weights;
use crate::ode::utils::dense_output::DenseSolution;
use crate::ode::utils::interpolation::ContinuousOutputMethod;
use crate::ode::utils::mass_matrix;
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ── Radau IIA 3-stage Butcher tableau (Hairer-Wanner, Table 5.5) ─────────────
// c = [(4-√6)/10,  (4+√6)/10,  1]
// A = [a11  a12  a13]     (see below)
//     [a21  a22  a23]
//     [a31  a32  a33]
// b = [a31, a32, a33]  (b = last row of A for Radau IIA)
//
// a11 = (88 − 7√6)/360          ≈  0.1968154772236605
// a12 = (296 − 169√6)/1800      ≈ −0.0655354258501984
// a13 = (−2 + 3√6)/225          ≈  0.0237709743482202
// a21 = (296 + 169√6)/1800      ≈  0.3944243147390873
// a22 = (88 + 7√6)/360          ≈  0.2920734116652285
// a23 = (−2 − 3√6)/225          ≈ −0.0415487521259979
// a31 = (16 − √6)/36            ≈  0.3764030627004673
// a32 = (16 + √6)/36            ≈  0.5124858261884216
// a33 = 1/9                     ≈  0.1111111111111111

/// Radau IIA c-coefficients
const C1_F64: f64 = 0.155_051_025_721_682_2; // (4-√6)/10
const C2_F64: f64 = 0.644_948_974_278_317_8; // (4+√6)/10

/// Butcher A-matrix entries
const A11_F64: f64 = 0.196_815_477_223_660_4;
const A12_F64: f64 = -0.065_535_425_850_198_4; // CORRECTED (was -0.067833…)
const A13_F64: f64 = 0.023_770_974_348_220_15; // CORRECTED (was -0.020795…)
const A21_F64: f64 = 0.394_424_314_739_087_3;
const A22_F64: f64 = 0.292_073_411_665_228_4; // CORRECTED (was 0.292100…)
const A23_F64: f64 = -0.041_548_752_125_997_9; // CORRECTED (was +0.041663…)
const A31_F64: f64 = 0.376_403_062_700_467_3;
const A32_F64: f64 = 0.512_485_826_188_421_6;
const A33_F64: f64 = 1.0 / 9.0;

/// Max Newton iterations per step (Hairer-Wanner §IV.8 recommends ≤ 7 for Radau)
const MAX_NEWTON_ITER: usize = 7;
/// Newton convergence tolerance (relative)
const NEWTON_KAPPA: f64 = 0.1; // contraction factor threshold

/// Finite-difference perturbation scale
const FD_EPS: f64 = 1e-7;

// ── Error estimator constants (Hairer-Wanner §IV.8 embedded estimate) ────────
// E = [(-13-7√6)/3, (-13+7√6)/3, -1/3]  (scipy radau.py, line 15)
// ZE = (E[0]*Z1 + E[1]*Z2 + E[2]*Z3) / h where Z_i = Y_i - y_n = h * Σ_j a_ij K_j
// So ZE = Σ_i E[i] * Σ_j A[i,j] * K_j = ec_k1*K1 + ec_k2*K2 + ec_k3*K3
// ec_ki = E[0]*A[0,i] + E[1]*A[1,i] + E[2]*A[2,i]  (column-wise contraction of E^T A)
const EC_K1_F64: f64 = -1.558_078_204_724_922_6; // E[0]*A11 + E[1]*A21 + E[2]*A31
const EC_K2_F64: f64 = 0.891_411_538_058_255_2; // E[0]*A12 + E[1]*A22 + E[2]*A32
const EC_K3_F64: f64 = -1.0 / 3.0; // E[0]*A13 + E[1]*A23 + E[2]*A33

/// MU_REAL = 3 + 3^(2/3) - 3^(1/3) ≈ 3.6378 — real eigenvalue of Radau A^{-1}
/// The error system is (MU_REAL/h * M - J) * err = K₀ + ZE
/// where K₀ = M⁻¹·f(t_n, y_n) is the initial stage derivative.
const MU_REAL_F64: f64 = 3.637_834_252_744_496;

// ─────────────────────────────────────────────────────────────────────────────

/// Solve an ODE with mass matrix using the Radau IIA method (K-parameterization).
///
/// Solves M(t,y)·y' = f(t,y) using the 3-stage, order-5 Radau IIA implicit
/// Runge-Kutta method with a coupled Newton iteration for the stage derivatives K.
#[allow(dead_code)]
pub fn radau_method_with_mass<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;
    let n = y0.len();
    let sn = 3 * n; // dimension of coupled 3N system

    // Validate mass matrix
    mass_matrix::check_mass_compatibility(&mass_matrix, t_start, y0.view())?;

    // Butcher coefficients (cast to generic F once)
    let c1 = F::from_f64(C1_F64).expect("from_f64");
    let c2 = F::from_f64(C2_F64).expect("from_f64");
    let c3 = F::one();

    let a11 = F::from_f64(A11_F64).expect("from_f64");
    let a12 = F::from_f64(A12_F64).expect("from_f64");
    let a13 = F::from_f64(A13_F64).expect("from_f64");
    let a21 = F::from_f64(A21_F64).expect("from_f64");
    let a22 = F::from_f64(A22_F64).expect("from_f64");
    let a23 = F::from_f64(A23_F64).expect("from_f64");
    let a31 = F::from_f64(A31_F64).expect("from_f64");
    let a32 = F::from_f64(A32_F64).expect("from_f64");
    let a33 = F::from_f64(A33_F64).expect("from_f64");

    // b = last row of A (Radau IIA property)
    let b1 = a31;
    let b2 = a32;
    let b3 = a33;

    // Step size initialisation
    let span = t_end - t_start;
    let h0 = opts
        .h0
        .unwrap_or_else(|| span / F::from_f64(100.0).expect("from_f64"));
    let min_step = opts
        .min_step
        .unwrap_or_else(|| span * F::from_f64(1e-10).expect("from_f64"));
    let max_step = opts.max_step.unwrap_or(span);

    // Integration variables
    let mut t = t_start;
    let mut y = y0.clone();
    let mut h = h0;

    // Stored trajectory
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];
    let mut dy_values: Vec<Array1<F>> = Vec::new();

    if opts.dense_output {
        let f_y0 = f(t, y.view());
        let dy0 = mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_y0.view())?;
        dy_values.push(dy0);
    }

    // Statistics
    let mut func_evals: usize = 1;
    let mut step_count: usize = 0;
    let mut accepted_steps: usize = 0;
    let mut rejected_steps: usize = 0;
    let mut n_lu: usize = 0;
    let mut n_jac: usize = 0;

    let rtol = opts.rtol;
    let atol = opts.atol;

    // Cached Jacobian and mass matrix (reused across accepted steps; cleared on rejection).
    // This implements simplified Newton per Hairer-Wanner §IV.8: J is recomputed only when
    // Newton convergence degrades (rejection path) or at the first step.
    let mut cached_jac: Option<Array2<F>> = None;
    let mut cached_m: Option<Option<Array2<F>>> = None; // outer Option = "cached"; inner = actual matrix or identity sentinel

    // ── Main integration loop ──────────────────────────────────────────────
    while t < t_end && step_count < opts.max_steps {
        if t + h > t_end {
            h = t_end - t;
        }
        h = h.min(max_step).max(min_step);

        let t1 = t + c1 * h;
        let t2 = t + c2 * h;
        let t3 = t + c3 * h;

        step_count += 1;

        // ── Evaluate M and J at (t_n, y_n) — reuse across accepted steps ───
        if cached_jac.is_none() {
            let f0 = f(t, y.view());
            func_evals += 1;
            let jac = finite_diff_jac(&f, t, &y, &f0, F::from_f64(FD_EPS).expect("from_f64"));
            cached_jac = Some(jac);
            n_jac += 1;

            // Evaluate mass matrix at (t_n, y_n)
            let m_opt = mass_matrix.evaluate(t, y.view());
            cached_m = Some(m_opt);
        }

        let jac = cached_jac.as_ref().expect("jacobian must be set");
        let m_eval = cached_m.as_ref().expect("mass matrix must be set");

        // ── Build coupled 3N×3N Newton matrix once per step ─────────────
        // J_Newton = (I_3 ⊗ M) - h*(A ⊗ J)
        // Laid out as block [3×3] of [n×n] blocks.
        // Block (i,j) = δ_ij * M  − h * a_ij * J
        let newton_mat = build_coupled_matrix(
            n,
            sn,
            h,
            &a11,
            &a12,
            &a13,
            &a21,
            &a22,
            &a23,
            &a31,
            &a32,
            &a33,
            m_eval,
            jac,
            &mass_matrix,
            t,
            &y,
        );
        n_lu += 1;

        // ── Initial guess K0 from M^{-1}*f(t_n, y_n) ────────────────────
        let k_init = compute_initial_k(&mass_matrix, t, &y, m_eval, &f, &mut func_evals)?;

        // Stage derivatives: K = [K1; K2; K3] as 3N vector
        let mut k = Array1::<F>::zeros(sn);
        for i in 0..n {
            k[i] = k_init[i];
            k[n + i] = k_init[i];
            k[2 * n + i] = k_init[i];
        }

        let error_weights = calculate_error_weights(&y, atol, rtol);

        // ── Newton iteration ──────────────────────────────────────────────
        let mut newton_converged = false;
        let mut prev_res_norm = F::from_f64(f64::MAX).expect("from_f64");

        for newton_iter in 0..MAX_NEWTON_ITER {
            // Compute stage values Y_i = y + h * Σ_j a_ij * K_j
            let k1 = k.slice(scirs2_core::ndarray::s![0..n]).to_owned();
            let k2 = k.slice(scirs2_core::ndarray::s![n..2 * n]).to_owned();
            let k3 = k.slice(scirs2_core::ndarray::s![2 * n..3 * n]).to_owned();

            let y1 = &y + &((&k1 * a11 + &k2 * a12 + &k3 * a13) * h);
            let y2 = &y + &((&k1 * a21 + &k2 * a22 + &k3 * a23) * h);
            let y3 = &y + &((&k1 * a31 + &k2 * a32 + &k3 * a33) * h);

            // Evaluate f at each stage
            let f1 = f(t1, y1.view());
            let f2 = f(t2, y2.view());
            let f3 = f(t3, y3.view());
            func_evals += 3;

            // Residual R_i = M*K_i - f(Y_i), assembled into 3N vector
            let r = compute_residual(
                n,
                &k1,
                &k2,
                &k3,
                &f1,
                &f2,
                &f3,
                &mass_matrix,
                t1,
                t2,
                t3,
                &y1,
                &y2,
                &y3,
            )?;

            // Residual norm (RMS over all components)
            let res_norm = rms_norm_weighted(&r, &error_weights, 3);

            // Newton convergence check
            if newton_iter > 0 {
                let theta = res_norm / prev_res_norm;
                // Diverging: give up early
                if theta > F::one() {
                    break;
                }
                // κ-rate convergence: estimate remaining error
                let kappa = F::from_f64(NEWTON_KAPPA).expect("from_f64");
                let predicted_final = theta / (F::one() - theta) * res_norm;
                if predicted_final < kappa || res_norm < F::from_f64(1e-10).expect("from_f64") {
                    newton_converged = true;
                    break;
                }
            }
            prev_res_norm = res_norm;

            if res_norm < F::from_f64(1e-10).expect("from_f64") {
                newton_converged = true;
                break;
            }

            // Solve J_Newton * ΔK = -R
            let neg_r = r.mapv(|x| -x);
            let dk = solve_coupled_system(&newton_mat, &neg_r, sn)?;

            // Update K
            k = k + dk;
        }

        // If Newton didn't converge in max iterations, try one last residual check
        if !newton_converged {
            // Recompute final residual to see if we're close enough
            let k1 = k.slice(scirs2_core::ndarray::s![0..n]).to_owned();
            let k2 = k.slice(scirs2_core::ndarray::s![n..2 * n]).to_owned();
            let k3 = k.slice(scirs2_core::ndarray::s![2 * n..3 * n]).to_owned();
            let y1 = &y + &((&k1 * a11 + &k2 * a12 + &k3 * a13) * h);
            let y2 = &y + &((&k1 * a21 + &k2 * a22 + &k3 * a23) * h);
            let y3 = &y + &((&k1 * a31 + &k2 * a32 + &k3 * a33) * h);
            let f1 = f(t1, y1.view());
            let f2 = f(t2, y2.view());
            let f3 = f(t3, y3.view());
            func_evals += 3;
            let r = compute_residual(
                n,
                &k1,
                &k2,
                &k3,
                &f1,
                &f2,
                &f3,
                &mass_matrix,
                t1,
                t2,
                t3,
                &y1,
                &y2,
                &y3,
            )?;
            let res_norm = rms_norm_weighted(&r, &error_weights, 3);
            if res_norm < F::from_f64(1e-6).expect("from_f64") {
                newton_converged = true;
            }
        }

        if !newton_converged {
            // Reduce step size and force Jacobian recomputation
            h *= F::from_f64(0.5).expect("from_f64");
            rejected_steps += 1;
            cached_jac = None;
            cached_m = None;

            if h < min_step {
                return Err(crate::error::IntegrateError::ComputationError(
                    "Radau Newton iteration failed to converge even at minimum step size"
                        .to_string(),
                ));
            }
            continue;
        }

        // ── Solution update ───────────────────────────────────────────────
        let k1 = k.slice(scirs2_core::ndarray::s![0..n]).to_owned();
        let k2 = k.slice(scirs2_core::ndarray::s![n..2 * n]).to_owned();
        let k3 = k.slice(scirs2_core::ndarray::s![2 * n..3 * n]).to_owned();

        let y_new = &y + &((&k1 * b1 + &k2 * b2 + &k3 * b3) * h);

        // ── Error estimate (Hairer-Wanner §IV.8 embedded estimator) ──────────
        // ZE = ec_k1*K1 + ec_k2*K2 + ec_k3*K3  (column contraction of E^T * A applied to K)
        // rhs_err = K_0 + ZE  where K_0 = M^{-1}*f(t_n, y_n) is the initial stage derivative.
        //
        // The scipy formula uses f_0 + ZE, which is correct for M=I where K_0=f_0.
        // For M≠I, K_0 = M^{-1}*f_0. At leading order, ZE ≈ -K_0, so K_0 + ZE ≈ 0.
        // Using K_0 here gives O(h^4) or better cancellation (correct O(h^5) estimate).
        //
        // After solving: (MU_REAL/h * M - J) * err = K_0 + ZE
        // the step controller uses exponent 1/4 (scipy convention).
        let ec_k1 = F::from_f64(EC_K1_F64).expect("from_f64");
        let ec_k2 = F::from_f64(EC_K2_F64).expect("from_f64");
        let ec_k3 = F::from_f64(EC_K3_F64).expect("from_f64");
        let ze = &k1 * ec_k1 + &k2 * ec_k2 + &k3 * ec_k3;
        let rhs_err = &k_init + &ze;

        // Build error matrix: (MU_REAL/h * M - J)
        let mu_h = F::from_f64(MU_REAL_F64).expect("from_f64") / h;
        let err_mat = build_error_matrix(n, mu_h, m_eval, jac);

        let error_vec = {
            use crate::ode::utils::linear_solvers::solve_linear_system;
            solve_linear_system(&err_mat.view(), &rhs_err.view())
                .unwrap_or_else(|_| rhs_err.clone())
        };

        // Update error weights using max(|y|, |y_new|) per scipy convention
        let scale: Array1<F> = y
            .iter()
            .zip(y_new.iter())
            .map(|(&yi, &yi_new)| {
                let mx = if yi.abs() > yi_new.abs() {
                    yi.abs()
                } else {
                    yi_new.abs()
                };
                atol + mx * rtol
            })
            .collect();

        let error_norm = error_vec
            .iter()
            .zip(scale.iter())
            .map(|(&e, &w)| (e / w).powi(2))
            .sum::<F>()
            .sqrt()
            / F::from_f64((n as f64).sqrt()).expect("from_f64");

        if error_norm <= F::one() {
            // Accept step
            t += h;
            y = y_new;

            t_values.push(t);
            y_values.push(y.clone());

            if opts.dense_output {
                let f_y = f(t, y.view());
                func_evals += 1;
                let dy = mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_y.view())?;
                dy_values.push(dy);
            }

            accepted_steps += 1;

            // Jacobian and mass matrix cache is intentionally kept across accepted steps.
            // The Newton matrix is rebuilt with the new h each step (so the LU changes),
            // but reusing J saves the O(N²) finite-difference Jacobian evaluation.
            // The cache is only invalidated on rejection (Newton failed → stale J).

            // Step size control: exponent 1/4 (embedded order-4 estimate, scipy convention).
            // fac_max=5 per Hairer-Wanner, safety=0.9 simplified.
            let fac_min = F::from_f64(0.2).expect("from_f64");
            let fac_max = F::from_f64(5.0).expect("from_f64");
            let safety = F::from_f64(0.9).expect("from_f64");
            let factor = if error_norm < F::from_f64(1e-14).expect("from_f64") {
                fac_max
            } else {
                let exponent = F::from_f64(0.25).expect("from_f64");
                let raw = safety * (F::one() / error_norm).powf(exponent);
                raw.max(fac_min).min(fac_max)
            };
            h *= factor;
        } else {
            // Reject step
            let exponent = F::from_f64(0.25).expect("from_f64");
            let safety = F::from_f64(0.9).expect("from_f64");
            let factor = (safety * (F::one() / error_norm).powf(exponent))
                .max(F::from_f64(0.2).expect("from_f64"))
                .min(F::from_f64(1.0).expect("from_f64"));
            h *= factor;
            rejected_steps += 1;
            // Invalidate Jacobian cache on rejection — Newton failure indicates
            // the current J is stale or the problem has changed enough to need a fresh one.
            cached_jac = None;
            cached_m = None;
        }
    }

    let success = t >= t_end;
    let message = if success {
        Some(format!("Integration successful, reached t = {t:?}"))
    } else {
        Some(format!("Integration incomplete, stopped at t = {t:?}"))
    };

    let _dense_output = if opts.dense_output {
        Some(DenseSolution::new(
            t_values.clone(),
            y_values.clone(),
            Some(dy_values),
            Some(ContinuousOutputMethod::CubicHermite),
            None,
        ))
    } else {
        None
    };

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu,
        n_jac,
        method: ODEMethod::Radau,
    })
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Finite-difference Jacobian ∂f/∂y at (t, y) using forward differences.
fn finite_diff_jac<F, Func>(f: &Func, t: F, y: &Array1<F>, f0: &Array1<F>, eps: F) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let mut jac = Array2::<F>::zeros((n, n));
    for j in 0..n {
        let scale = F::one() + y[j].abs();
        let h_j = eps * scale;
        let mut yp = y.clone();
        yp[j] += h_j;
        let fp = f(t, yp.view());
        for i in 0..n {
            jac[[i, j]] = (fp[i] - f0[i]) / h_j;
        }
    }
    jac
}

/// Build the coupled 3N×3N Newton matrix:
///   Block (i,j) = δ_ij * M  − h * a_ij * J
#[allow(clippy::too_many_arguments)]
fn build_coupled_matrix<F>(
    n: usize,
    sn: usize,
    h: F,
    a11: &F,
    a12: &F,
    a13: &F,
    a21: &F,
    a22: &F,
    a23: &F,
    a31: &F,
    a32: &F,
    a33: &F,
    m_opt: &Option<Array2<F>>,
    jac: &Array2<F>,
    _mass: &MassMatrix<F>,
    _t: F,
    _y: &Array1<F>,
) -> Array2<F>
where
    F: IntegrateFloat,
{
    let a_mat = [[*a11, *a12, *a13], [*a21, *a22, *a23], [*a31, *a32, *a33]];
    let mut mat = Array2::<F>::zeros((sn, sn));

    for bi in 0..3 {
        for bj in 0..3 {
            let a_ij = a_mat[bi][bj];
            let row_off = bi * n;
            let col_off = bj * n;
            for i in 0..n {
                for j in 0..n {
                    // δ_{block-diag} * M_{ij}
                    let m_ij = if bi == bj {
                        match m_opt {
                            Some(ref m) => m[[i, j]],
                            None => {
                                if i == j {
                                    F::one()
                                } else {
                                    F::zero()
                                }
                            }
                        }
                    } else {
                        F::zero()
                    };
                    // −h * a_ij * J_{ij}
                    let j_term = h * a_ij * jac[[i, j]];
                    mat[[row_off + i, col_off + j]] = m_ij - j_term;
                }
            }
        }
    }
    mat
}

/// Build the N×N error-estimation matrix: mu_h * M - J
///
/// Used for the Hairer-Wanner embedded error estimator:
///   (MU_REAL/h * M - J) * err = rhs_err
fn build_error_matrix<F>(n: usize, mu_h: F, m_opt: &Option<Array2<F>>, jac: &Array2<F>) -> Array2<F>
where
    F: IntegrateFloat,
{
    let mut mat = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let m_ij = match m_opt {
                Some(ref m) => m[[i, j]],
                None => {
                    if i == j {
                        F::one()
                    } else {
                        F::zero()
                    }
                }
            };
            mat[[i, j]] = mu_h * m_ij - jac[[i, j]];
        }
    }
    mat
}

/// Compute the initial guess for K: M^{-1} * f(t_n, y_n) broadcast to all stages.
fn compute_initial_k<F, Func>(
    mass: &MassMatrix<F>,
    t: F,
    y: &Array1<F>,
    m_opt: &Option<Array2<F>>,
    f: &Func,
    func_evals: &mut usize,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let f0 = f(t, y.view());
    *func_evals += 1;

    let k0 = match m_opt {
        None => f0, // identity mass — K = f directly
        Some(_) => mass_matrix::solve_mass_system(mass, t, y.view(), f0.view())?,
    };
    Ok(k0)
}

/// Compute residual vector R of length 3N:
///   R_i = M * K_i - f(Y_i)   for each stage i ∈ {1,2,3}
///
/// M is evaluated at (t_i, Y_i) for state-dependent mass, or at t_i for time-dependent.
#[allow(clippy::too_many_arguments)]
fn compute_residual<F>(
    n: usize,
    k1: &Array1<F>,
    k2: &Array1<F>,
    k3: &Array1<F>,
    f1: &Array1<F>,
    f2: &Array1<F>,
    f3: &Array1<F>,
    mass: &MassMatrix<F>,
    t1: F,
    t2: F,
    t3: F,
    y1: &Array1<F>,
    y2: &Array1<F>,
    y3: &Array1<F>,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    let mut r = Array1::<F>::zeros(3 * n);

    // Stage 1
    let mk1 = apply_m_vec(mass, t1, y1, k1)?;
    for i in 0..n {
        r[i] = mk1[i] - f1[i];
    }

    // Stage 2
    let mk2 = apply_m_vec(mass, t2, y2, k2)?;
    for i in 0..n {
        r[n + i] = mk2[i] - f2[i];
    }

    // Stage 3
    let mk3 = apply_m_vec(mass, t3, y3, k3)?;
    for i in 0..n {
        r[2 * n + i] = mk3[i] - f3[i];
    }

    Ok(r)
}

/// Apply mass matrix to a vector: result = M(t, y) * v
fn apply_m_vec<F>(
    mass: &MassMatrix<F>,
    t: F,
    y: &Array1<F>,
    v: &Array1<F>,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    mass_matrix::apply_mass(mass, t, y.view(), v.view())
}

/// RMS-weighted norm: sqrt(Σ (r_i / w_{i mod n})^2 / (s*n))
/// where s = number of stages.
fn rms_norm_weighted<F: IntegrateFloat>(r: &Array1<F>, weights: &Array1<F>, stages: usize) -> F {
    let n = weights.len();
    let total = r.len();
    let sum = r
        .iter()
        .enumerate()
        .map(|(idx, &ri)| {
            let w = weights[idx % n];
            (ri / w).powi(2)
        })
        .sum::<F>();
    let denom = F::from_usize(total.max(1)).expect("from_usize")
        / F::from_usize(stages).expect("from_usize");
    (sum / denom).sqrt()
}

/// Solve the 3N×3N linear system using Gaussian elimination with partial pivoting.
fn solve_coupled_system<F: IntegrateFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
    sn: usize,
) -> IntegrateResult<Array1<F>> {
    use crate::ode::utils::linear_solvers::solve_linear_system;
    let av = a.view();
    let bv = b.view();
    // solve_linear_system expects shape (sn, sn) matrix
    debug_assert_eq!(a.shape(), &[sn, sn]);
    solve_linear_system(&av, &bv)
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Verify Butcher tableau satisfies the order-5 consistency conditions:
    /// Σ b_i = 1, Σ b_i * c_i = 1/2, etc.
    #[test]
    fn test_butcher_tableau_consistency() {
        let b1 = A31_F64;
        let b2 = A32_F64;
        let b3 = A33_F64;
        let c1 = C1_F64;
        let c2 = C2_F64;
        let c3 = 1.0_f64;

        // Σ b_i = 1 (order 1)
        assert_relative_eq!(b1 + b2 + b3, 1.0, epsilon = 1e-12);

        // Σ b_i * c_i = 1/2 (order 2)
        assert_relative_eq!(b1 * c1 + b2 * c2 + b3 * c3, 0.5, epsilon = 1e-12);

        // Σ b_i * c_i^2 = 1/3 (order 3)
        assert_relative_eq!(
            b1 * c1.powi(2) + b2 * c2.powi(2) + b3 * c3.powi(2),
            1.0 / 3.0,
            epsilon = 1e-12
        );

        // Row-sum condition: Σ_j a_ij = c_i
        let row_sum_1 = A11_F64 + A12_F64 + A13_F64;
        let row_sum_2 = A21_F64 + A22_F64 + A23_F64;
        let row_sum_3 = A31_F64 + A32_F64 + A33_F64;
        assert_relative_eq!(row_sum_1, c1, epsilon = 1e-13);
        assert_relative_eq!(row_sum_2, c2, epsilon = 1e-13);
        assert_relative_eq!(row_sum_3, c3, epsilon = 1e-13);
    }
}
