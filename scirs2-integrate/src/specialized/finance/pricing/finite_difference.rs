//! Finite difference methods for option pricing

use crate::error::IntegrateResult;
use crate::specialized::finance::models::VolatilityModel;
use crate::specialized::finance::solvers::StochasticPDESolver;
use crate::specialized::finance::types::{FinancialOption, OptionStyle, OptionType};
use scirs2_core::ndarray::Array2;

/// Main finite difference pricing function
pub fn price_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
) -> IntegrateResult<f64> {
    match &solver.volatility_model {
        VolatilityModel::Constant(sigma) => black_scholes_finite_difference(solver, option, *sigma),
        VolatilityModel::Heston {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => heston_finite_difference(solver, option, *v0, *theta, *kappa, *sigma, *rho),
        // Add other models as needed
        _ => Err(crate::error::IntegrateError::ValueError(
            "Finite difference not implemented for this volatility model".to_string(),
        )),
    }
}

/// Black-Scholes finite difference solver (Crank-Nicolson scheme for stability)
pub fn black_scholes_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    sigma: f64,
) -> IntegrateResult<f64> {
    let n_s = solver.n_asset;
    let n_t = solver.n_time;

    let s_max = option.spot * 3.0;
    let ds = s_max / (n_s - 1) as f64;
    let dt = option.maturity / (n_t - 1) as f64;

    // Initialize grid
    let mut v = vec![0.0; n_s];

    // Terminal condition
    for i in 0..n_s {
        let s = i as f64 * ds;
        v[i] = solver.payoff(option, s);
    }

    // Backward time stepping with Crank-Nicolson
    for _ in (0..n_t - 1).rev() {
        let mut v_new = vec![0.0; n_s];

        // Boundary conditions
        v_new[0] = match option.option_type {
            OptionType::Call => 0.0,
            OptionType::Put => option.strike * (-option.risk_free_rate * dt).exp(),
        };
        v_new[n_s - 1] = match option.option_type {
            OptionType::Call => s_max - option.strike * (-option.risk_free_rate * dt).exp(),
            OptionType::Put => 0.0,
        };

        // Build tridiagonal system for Crank-Nicolson
        // Implicit part: A * V^{n+1} = B * V^n + boundary terms
        let mut lower = vec![0.0; n_s - 1]; // Sub-diagonal
        let mut diag = vec![0.0; n_s]; // Diagonal
        let mut upper = vec![0.0; n_s - 1]; // Super-diagonal
        let mut rhs = vec![0.0; n_s]; // Right-hand side

        // Boundary nodes
        diag[0] = 1.0;
        rhs[0] = v_new[0];
        diag[n_s - 1] = 1.0;
        rhs[n_s - 1] = v_new[n_s - 1];

        // Interior nodes
        for i in 1..n_s - 1 {
            let j = i as f64;
            let alpha = 0.25 * dt * (sigma * sigma * j * j - option.risk_free_rate * j);
            let beta = -0.5 * dt * (sigma * sigma * j * j + option.risk_free_rate);
            let gamma = 0.25 * dt * (sigma * sigma * j * j + option.risk_free_rate * j);

            // Implicit side coefficients (left-hand side)
            if i > 0 {
                lower[i - 1] = -alpha;
            }
            diag[i] = 1.0 - beta;
            if i < n_s - 1 {
                upper[i] = -gamma;
            }

            // Explicit side (right-hand side)
            rhs[i] = alpha * v[i - 1] + (1.0 + beta) * v[i] + gamma * v[i + 1];
        }

        // Solve tridiagonal system using Thomas algorithm
        v_new = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

        // Apply early exercise condition for American options
        if option.option_style == OptionStyle::American {
            for i in 1..n_s - 1 {
                let s = i as f64 * ds;
                v_new[i] = v_new[i].max(solver.payoff(option, s));
            }
        }

        v = v_new;
    }

    // Interpolate to get option value at initial spot price
    let spot_idx = ((option.spot / ds) as usize).min(n_s - 2);
    let weight = (option.spot - spot_idx as f64 * ds) / ds;

    Ok(v[spot_idx] * (1.0 - weight) + v[spot_idx + 1] * weight)
}

/// Thomas algorithm for solving tridiagonal systems
/// Solves: lower[i-1] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1] = rhs[i]
fn solve_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> IntegrateResult<Vec<f64>> {
    let n = diag.len();
    if lower.len() != n - 1 || upper.len() != n - 1 || rhs.len() != n {
        return Err(crate::error::IntegrateError::ValueError(
            "Inconsistent tridiagonal system dimensions".to_string(),
        ));
    }

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward elimination
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n - 1 {
        let m = 1.0 / (diag[i] - lower[i - 1] * c_prime[i - 1]);
        c_prime[i] = upper[i] * m;
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) * m;
    }

    // Last row
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2])
        / (diag[n - 1] - lower[n - 2] * c_prime[n - 2]);

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Heston model finite difference solver using the Craig-Sneyd (In't Hout-Welfert) ADI scheme.
///
/// The Heston PDE is:
///   dV/dt = (1/2)vS²(d²V/dS²) + ρσvS(d²V/dSdv) + (1/2)σ²v(d²V/dv²)
///           + rS(dV/dS) + κ(θ-v)(dV/dv) - rV
///
/// We split into three operators:
///   L₀ = mixed derivative (explicit throughout)
///   L₁ = S-direction (diffusion + drift - half discount)
///   L₂ = v-direction (diffusion + drift - half discount)
///
/// Craig-Sneyd scheme (θ_cs = 0.5, two-pass stabilization):
///   Pass 1 (predictor):
///     Y₀ = Uₙ + dt*(L₀ + L₁ + L₂)*Uₙ
///     (I - θ·dt·L₁)·Y₁ = Y₀ - θ·dt·L₁·Uₙ
///     (I - θ·dt·L₂)·Y₂ = Y₁ - θ·dt·L₂·Uₙ
///   Pass 2 (corrector, stabilizes mixed term):
///     Ỹ₀ = Y₀ + (dt/2)*L₀*(Y₂ - Uₙ)
///     (I - θ·dt·L₁)·Ỹ₁ = Ỹ₀ - θ·dt·L₁·Uₙ
///     (I - θ·dt·L₂)·Uₙ₊₁ = Ỹ₁ - θ·dt·L₂·Uₙ
pub fn heston_finite_difference(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
) -> IntegrateResult<f64> {
    let n_s = solver.n_asset.max(50);
    let n_v = solver.n_vol.unwrap_or(30).max(20);
    let n_t = solver.n_time.max(200);

    // Craig-Sneyd parameter (θ = 1/2 gives Crank-Nicolson accuracy)
    let theta_cs = 0.5_f64;

    // Grid parameters — use log-uniform spacing for S for better accuracy
    // S ∈ [0, S_max], v ∈ [0, V_max]
    let s_max = option.spot * 4.0;
    let v_max = (theta * 5.0).max(0.8);
    let ds = s_max / (n_s - 1) as f64;
    let dv = v_max / (n_v - 1) as f64;
    let dt = option.maturity / n_t as f64;

    // Initialize 2D value grid (n_s × n_v), index [i_s, i_v]
    let mut u = Array2::zeros((n_s, n_v));

    // Terminal condition: payoff at maturity
    for i in 0..n_s {
        let s = i as f64 * ds;
        for j in 0..n_v {
            u[[i, j]] = solver.payoff(option, s);
        }
    }

    // Pre-compute grid node coordinates
    let s_nodes: Vec<f64> = (0..n_s).map(|i| i as f64 * ds).collect();
    let v_nodes: Vec<f64> = (0..n_v).map(|j| (j as f64 * dv).max(1e-8)).collect();

    // Helper: apply boundary conditions to a grid
    let apply_boundaries =
        |grid: &mut Array2<f64>, dt_frac: f64, option: &FinancialOption, s_max: f64| {
            let disc = (-option.risk_free_rate * dt_frac).exp();
            for j in 0..n_v {
                // S = 0 boundary
                grid[[0, j]] = match option.option_type {
                    OptionType::Call => 0.0,
                    OptionType::Put => option.strike * disc,
                };
                // S = S_max boundary
                grid[[n_s - 1, j]] = match option.option_type {
                    OptionType::Call => s_max - option.strike * disc,
                    OptionType::Put => 0.0,
                };
            }
            for i in 0..n_s {
                // v = 0: Neumann (zero diffusion, copy from j=1)
                grid[[i, 0]] = grid[[i, 1]];
                // v = v_max: linear extrapolation
                if n_v >= 3 {
                    grid[[i, n_v - 1]] = 2.0 * grid[[i, n_v - 2]] - grid[[i, n_v - 3]];
                }
            }
        };

    // Helper: compute L₀*grid (mixed derivative, fully explicit)
    // L₀[i,j] = ρ σ v_j S_i * (d²U/dSdv) ≈ ρσv_j S_i/(4 ds dv) * cross_term
    let compute_l0 = |grid: &Array2<f64>| -> Array2<f64> {
        let mut result = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                let s = s_nodes[i];
                let v = v_nodes[j];
                // Central difference for d²U/dSdv
                let cross = (grid[[i + 1, j + 1]] - grid[[i + 1, j - 1]] - grid[[i - 1, j + 1]]
                    + grid[[i - 1, j - 1]])
                    / (4.0 * ds * dv);
                result[[i, j]] = rho * sigma * v * s * cross;
            }
        }
        result
    };

    // Helper: build L₁ tridiagonal coefficients (S-direction, j fixed)
    // L₁[i] = (1/2)v S² d²U/dS² + rS dU/dS - (r/2)U
    // Using central differences: a[i]*U[i-1] + b[i]*U[i] + c[i]*U[i+1]
    let l1_coeffs = |variance: f64, r: f64| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0; n_s];
        let mut b = vec![0.0; n_s];
        let mut c = vec![0.0; n_s];
        for i in 1..n_s - 1 {
            let s = s_nodes[i];
            // d²U/dS² coefficient: (1/2)*v*S²/ds²
            let diff_s = 0.5 * variance * s * s / (ds * ds);
            // dU/dS coefficient: r*S/(2*ds)
            let adv_s = r * s / (2.0 * ds);
            a[i] = diff_s - adv_s;
            b[i] = -2.0 * diff_s - 0.5 * r;
            c[i] = diff_s + adv_s;
        }
        (a, b, c)
    };

    // Helper: build L₂ tridiagonal coefficients (v-direction, i fixed)
    // L₂[j] = (1/2)σ²v d²U/dv² + κ(θ-v) dU/dv - (r/2)U
    let l2_coeffs = |r: f64| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0; n_v];
        let mut b = vec![0.0; n_v];
        let mut c = vec![0.0; n_v];
        for j in 1..n_v - 1 {
            let v = v_nodes[j];
            // d²U/dv² coefficient: (1/2)*σ²*v/dv²
            let diff_v = 0.5 * sigma * sigma * v / (dv * dv);
            // dU/dv coefficient: κ(θ-v)/(2*dv)
            let drift_v = kappa * (theta - v) / (2.0 * dv);
            a[j] = diff_v - drift_v;
            b[j] = -2.0 * diff_v - 0.5 * r;
            c[j] = diff_v + drift_v;
        }
        (a, b, c)
    };

    // Helper: apply (I - θ_cs*dt*L₁)*result = rhs, solving n_v independent tridiagonals
    // rhs is (n_s, n_v); we solve each column j independently
    let implicit_s_step = |rhs: &Array2<f64>,
                           u_n: &Array2<f64>,
                           r: f64,
                           theta_factor: f64|
     -> IntegrateResult<Array2<f64>> {
        let mut result = rhs.clone();
        // Apply boundary conditions on result first (boundaries are fixed)
        for j in 0..n_v {
            result[[0, j]] = u_n[[0, j]];
            result[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for j in 1..n_v - 1 {
            let variance = v_nodes[j];
            let (a, b, c) = l1_coeffs(variance, r);
            // Build (I - theta_factor * L₁) system
            let mut lower_tri = vec![0.0; n_s - 1];
            let mut diag_tri = vec![0.0; n_s];
            let mut upper_tri = vec![0.0; n_s - 1];
            let mut rhs_col = vec![0.0; n_s];

            // Boundary rows stay as-is
            diag_tri[0] = 1.0;
            rhs_col[0] = result[[0, j]];
            diag_tri[n_s - 1] = 1.0;
            rhs_col[n_s - 1] = result[[n_s - 1, j]];

            for i in 1..n_s - 1 {
                lower_tri[i - 1] = -theta_factor * a[i];
                diag_tri[i] = 1.0 - theta_factor * b[i];
                upper_tri[i] = -theta_factor * c[i];
                rhs_col[i] = rhs[[i, j]];
            }

            let sol = solve_tridiagonal(&lower_tri, &diag_tri, &upper_tri, &rhs_col)?;
            for i in 0..n_s {
                result[[i, j]] = sol[i];
            }
        }
        Ok(result)
    };

    // Helper: apply (I - θ_cs*dt*L₂)*result = rhs, solving n_s independent tridiagonals
    let implicit_v_step = |rhs: &Array2<f64>,
                           u_n: &Array2<f64>,
                           r: f64,
                           theta_factor: f64|
     -> IntegrateResult<Array2<f64>> {
        let mut result = rhs.clone();
        let (a, b, c) = l2_coeffs(r);
        for i in 1..n_s - 1 {
            // Build (I - theta_factor * L₂) system
            let mut lower_tri = vec![0.0; n_v - 1];
            let mut diag_tri = vec![0.0; n_v];
            let mut upper_tri = vec![0.0; n_v - 1];
            let mut rhs_col = vec![0.0; n_v];

            // Boundary rows
            diag_tri[0] = 1.0;
            rhs_col[0] = u_n[[i, 1]]; // v=0: Neumann copy
            diag_tri[n_v - 1] = 1.0;
            if n_v >= 3 {
                rhs_col[n_v - 1] = 2.0 * u_n[[i, n_v - 2]] - u_n[[i, n_v - 3]];
            }

            for j in 1..n_v - 1 {
                lower_tri[j - 1] = -theta_factor * a[j];
                diag_tri[j] = 1.0 - theta_factor * b[j];
                upper_tri[j] = -theta_factor * c[j];
                rhs_col[j] = rhs[[i, j]];
            }

            let sol = solve_tridiagonal(&lower_tri, &diag_tri, &upper_tri, &rhs_col)?;
            for j in 0..n_v {
                result[[i, j]] = sol[j];
            }
        }
        // Fix S boundaries after V solve
        for j in 0..n_v {
            result[[0, j]] = u_n[[0, j]];
            result[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        Ok(result)
    };

    // Backward time stepping
    for _t_idx in 0..n_t {
        let u_n = u.clone();
        let r = option.risk_free_rate;

        // --- Compute explicit operators applied to U_n ---
        let l0_un = compute_l0(&u_n);

        // L₁*U_n (for each j, apply L₁ along S)
        let mut l1_un = Array2::zeros((n_s, n_v));
        for j in 1..n_v - 1 {
            let variance = v_nodes[j];
            let (a, b, c) = l1_coeffs(variance, r);
            for i in 1..n_s - 1 {
                l1_un[[i, j]] =
                    a[i] * u_n[[i - 1, j]] + b[i] * u_n[[i, j]] + c[i] * u_n[[i + 1, j]];
            }
        }

        // L₂*U_n (for each i, apply L₂ along v)
        let mut l2_un = Array2::zeros((n_s, n_v));
        let (a2, b2, c2) = l2_coeffs(r);
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                l2_un[[i, j]] =
                    a2[j] * u_n[[i, j - 1]] + b2[j] * u_n[[i, j]] + c2[j] * u_n[[i, j + 1]];
            }
        }

        // ======================================================
        // Craig-Sneyd predictor pass
        // ======================================================
        // Y₀ = U_n + dt*(L₀ + L₁ + L₂)*U_n
        let mut y0 = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                y0[[i, j]] = u_n[[i, j]] + dt * (l0_un[[i, j]] + l1_un[[i, j]] + l2_un[[i, j]]);
            }
        }
        // Boundaries stay at U_n values for intermediate steps
        for j in 0..n_v {
            y0[[0, j]] = u_n[[0, j]];
            y0[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            y0[[i, 0]] = u_n[[i, 0]];
            y0[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }

        // RHS₁ = Y₀ - θ*dt*L₁*U_n  →  (I - θ*dt*L₁)*Y₁ = RHS₁
        let mut rhs1 = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                rhs1[[i, j]] = y0[[i, j]] - theta_cs * dt * l1_un[[i, j]];
            }
        }
        // Boundary values (used as fixed in implicit step)
        for j in 0..n_v {
            rhs1[[0, j]] = u_n[[0, j]];
            rhs1[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            rhs1[[i, 0]] = u_n[[i, 0]];
            rhs1[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }
        let y1 = implicit_s_step(&rhs1, &u_n, r, theta_cs * dt)?;

        // RHS₂ = Y₁ - θ*dt*L₂*U_n  →  (I - θ*dt*L₂)*Y₂ = RHS₂
        let mut rhs2 = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                rhs2[[i, j]] = y1[[i, j]] - theta_cs * dt * l2_un[[i, j]];
            }
        }
        for j in 0..n_v {
            rhs2[[0, j]] = u_n[[0, j]];
            rhs2[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            rhs2[[i, 0]] = u_n[[i, 0]];
            rhs2[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }
        let y2 = implicit_v_step(&rhs2, &u_n, r, theta_cs * dt)?;

        // ======================================================
        // Craig-Sneyd corrector pass (stabilizes the mixed term)
        // ======================================================
        // Ỹ₀ = Y₀ + (dt/2)*L₀*(Y₂ - U_n)
        let l0_y2_minus_un = {
            let mut diff = Array2::zeros((n_s, n_v));
            for i in 0..n_s {
                for j in 0..n_v {
                    diff[[i, j]] = y2[[i, j]] - u_n[[i, j]];
                }
            }
            compute_l0(&diff)
        };

        let mut y0_tilde = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                y0_tilde[[i, j]] = y0[[i, j]] + 0.5 * dt * l0_y2_minus_un[[i, j]];
            }
        }
        for j in 0..n_v {
            y0_tilde[[0, j]] = u_n[[0, j]];
            y0_tilde[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            y0_tilde[[i, 0]] = u_n[[i, 0]];
            y0_tilde[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }

        // RHS̃₁ = Ỹ₀ - θ*dt*L₁*U_n  →  (I - θ*dt*L₁)*Ỹ₁ = RHS̃₁
        let mut rhs1_tilde = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                rhs1_tilde[[i, j]] = y0_tilde[[i, j]] - theta_cs * dt * l1_un[[i, j]];
            }
        }
        for j in 0..n_v {
            rhs1_tilde[[0, j]] = u_n[[0, j]];
            rhs1_tilde[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            rhs1_tilde[[i, 0]] = u_n[[i, 0]];
            rhs1_tilde[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }
        let y1_tilde = implicit_s_step(&rhs1_tilde, &u_n, r, theta_cs * dt)?;

        // RHS̃₂ = Ỹ₁ - θ*dt*L₂*U_n  →  (I - θ*dt*L₂)*U_{n+1} = RHS̃₂
        let mut rhs2_tilde = Array2::zeros((n_s, n_v));
        for i in 1..n_s - 1 {
            for j in 1..n_v - 1 {
                rhs2_tilde[[i, j]] = y1_tilde[[i, j]] - theta_cs * dt * l2_un[[i, j]];
            }
        }
        for j in 0..n_v {
            rhs2_tilde[[0, j]] = u_n[[0, j]];
            rhs2_tilde[[n_s - 1, j]] = u_n[[n_s - 1, j]];
        }
        for i in 0..n_s {
            rhs2_tilde[[i, 0]] = u_n[[i, 0]];
            rhs2_tilde[[i, n_v - 1]] = u_n[[i, n_v - 1]];
        }
        let mut u_new = implicit_v_step(&rhs2_tilde, &u_n, r, theta_cs * dt)?;

        // Apply financial boundary conditions for this time step
        apply_boundaries(&mut u_new, dt, option, s_max);

        // Apply early exercise condition for American options
        if option.option_style == crate::specialized::finance::types::OptionStyle::American {
            for i in 1..n_s - 1 {
                let s = s_nodes[i];
                let intrinsic = solver.payoff(option, s);
                for j in 1..n_v - 1 {
                    u_new[[i, j]] = u_new[[i, j]].max(intrinsic);
                }
            }
        }

        u = u_new;
    }

    // Interpolate to get option value at (S0, V0) using bilinear interpolation
    let i_s = ((option.spot / ds) as usize).min(n_s - 2);
    let i_v = ((v0 / dv) as usize).min(n_v - 2);

    let w_s = (option.spot - i_s as f64 * ds) / ds;
    let w_v = (v0 - i_v as f64 * dv) / dv;

    let v00 = u[[i_s, i_v]];
    let v10 = u[[i_s + 1, i_v]];
    let v01 = u[[i_s, i_v + 1]];
    let v11 = u[[i_s + 1, i_v + 1]];

    let value = (1.0 - w_s) * (1.0 - w_v) * v00
        + w_s * (1.0 - w_v) * v10
        + (1.0 - w_s) * w_v * v01
        + w_s * w_v * v11;

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::models::VolatilityModel;
    use crate::specialized::finance::types::{FinanceMethod, OptionStyle, OptionType};

    #[test]
    fn test_black_scholes_fd_european_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::FiniteDifference,
        );

        let price = price_finite_difference(&solver, &option).expect("Operation failed");

        // Black-Scholes reference: ~10.45
        assert!(price > 9.0 && price < 12.0, "Price: {}", price);
    }

    #[test]
    fn test_black_scholes_fd_european_put() {
        let option = FinancialOption {
            option_type: OptionType::Put,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::FiniteDifference,
        );

        let price = price_finite_difference(&solver, &option).expect("Operation failed");

        // Black-Scholes reference: ~5.57
        assert!(price > 4.5 && price < 7.0, "Price: {}", price);
    }

    #[test]
    fn test_heston_fd_european_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let solver = StochasticPDESolver::new(
            50,
            30,
            VolatilityModel::Heston {
                v0: 0.04,
                theta: 0.04,
                kappa: 2.0,
                sigma: 0.3,
                rho: -0.7,
            },
            FinanceMethod::FiniteDifference,
        );

        let price = price_finite_difference(&solver, &option).expect("Operation failed");

        // Should be reasonable (close to Black-Scholes for these params)
        assert!(price > 8.0 && price < 13.0, "Price: {}", price);
    }
}
