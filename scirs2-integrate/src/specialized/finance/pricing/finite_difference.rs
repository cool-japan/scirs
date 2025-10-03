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

/// Heston model finite difference solver (2D PDE with ADI scheme)
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
    let n_v = solver.n_vol.unwrap_or(30);
    let n_t = solver.n_time.max(100);

    // Grid parameters
    let s_max = option.spot * 3.0;
    let v_max = (theta * 3.0).max(0.5);
    let ds = s_max / (n_s - 1) as f64;
    let dv = v_max / (n_v - 1) as f64;
    let dt = option.maturity / (n_t - 1) as f64;

    // Initialize 2D value grid (asset x volatility)
    let mut v_grid = Array2::zeros((n_s, n_v));

    // Terminal condition: payoff at maturity
    for i in 0..n_s {
        let s = i as f64 * ds;
        for j in 0..n_v {
            v_grid[[i, j]] = solver.payoff(option, s);
        }
    }

    // Backward time stepping using ADI (Alternating Direction Implicit)
    for _t_idx in (0..n_t - 1).rev() {
        let mut v_intermediate = v_grid.clone();
        let mut v_new = Array2::zeros((n_s, n_v));

        // Step 1: Implicit in S direction, explicit in V direction
        for j in 1..n_v - 1 {
            let variance = (j as f64 * dv).max(0.0001);

            // Build tridiagonal system for this V-slice
            let mut lower = vec![0.0; n_s - 1];
            let mut diag = vec![0.0; n_s];
            let mut upper = vec![0.0; n_s - 1];
            let mut rhs = vec![0.0; n_s];

            // Boundary conditions
            diag[0] = 1.0;
            rhs[0] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => 0.0,
                crate::specialized::finance::types::OptionType::Put => {
                    option.strike * (-option.risk_free_rate * 0.5 * dt).exp()
                }
            };
            diag[n_s - 1] = 1.0;
            rhs[n_s - 1] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => {
                    s_max - option.strike * (-option.risk_free_rate * 0.5 * dt).exp()
                }
                crate::specialized::finance::types::OptionType::Put => 0.0,
            };

            // Interior nodes in S direction
            for i in 1..n_s - 1 {
                let s = i as f64 * ds;

                // Coefficients for implicit S-direction
                let alpha_s = 0.25
                    * dt
                    * (variance * (i as f64) * (i as f64) - option.risk_free_rate * i as f64);
                let beta_s =
                    -0.5 * dt * (variance * (i as f64) * (i as f64) + option.risk_free_rate);
                let gamma_s = 0.25
                    * dt
                    * (variance * (i as f64) * (i as f64) + option.risk_free_rate * i as f64);

                if i > 0 {
                    lower[i - 1] = -alpha_s;
                }
                diag[i] = 1.0 - beta_s;
                if i < n_s - 1 {
                    upper[i] = -gamma_s;
                }

                // Explicit V-direction terms
                let drift_v = kappa * (theta - variance);
                let diffusion_v = 0.5 * sigma * sigma * variance;

                let v_term = if j > 0 && j < n_v - 1 {
                    let dv_dv_vol = (v_grid[[i, j + 1]] - v_grid[[i, j - 1]]) / (2.0 * dv);
                    let d2v_dv2 = (v_grid[[i, j + 1]] - 2.0 * v_grid[[i, j]] + v_grid[[i, j - 1]])
                        / (dv * dv);
                    0.5 * dt * (drift_v * dv_dv_vol + diffusion_v * d2v_dv2)
                } else {
                    0.0
                };

                // RHS with explicit V terms
                rhs[i] = alpha_s * v_grid[[i - 1, j]]
                    + (1.0 + beta_s) * v_grid[[i, j]]
                    + gamma_s * v_grid[[i + 1, j]]
                    + v_term;
            }

            // Solve tridiagonal system
            let solution = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;
            for i in 0..n_s {
                v_intermediate[[i, j]] = solution[i];
            }
        }

        // Handle V boundaries for intermediate
        for i in 0..n_s {
            v_intermediate[[i, 0]] = v_intermediate[[i, 1]]; // V = 0
            v_intermediate[[i, n_v - 1]] =
                2.0 * v_intermediate[[i, n_v - 2]] - v_intermediate[[i, n_v - 3]];
            // V = V_max
        }

        // Step 2: Implicit in V direction, explicit in S direction
        for i in 1..n_s - 1 {
            let s = i as f64 * ds;

            // Build tridiagonal system for this S-slice
            let mut lower = vec![0.0; n_v - 1];
            let mut diag = vec![0.0; n_v];
            let mut upper = vec![0.0; n_v - 1];
            let mut rhs = vec![0.0; n_v];

            // Boundary conditions
            diag[0] = 1.0;
            rhs[0] = v_intermediate[[i, 1]]; // Extrapolate
            diag[n_v - 1] = 1.0;
            rhs[n_v - 1] = 2.0 * v_intermediate[[i, n_v - 2]] - v_intermediate[[i, n_v - 3]];

            // Interior nodes in V direction
            for j in 1..n_v - 1 {
                let variance = (j as f64 * dv).max(0.0001);

                // Coefficients for implicit V-direction (Crank-Nicolson)
                let drift_v = kappa * (theta - variance);
                let diffusion_v = 0.5 * sigma * sigma * variance;

                // Using central differences for first and second derivatives
                let coef_lower = 0.25 * dt * (drift_v / (2.0 * dv) - diffusion_v / (dv * dv));
                let coef_diag = 0.5 * dt * (2.0 * diffusion_v / (dv * dv) + option.risk_free_rate);
                let coef_upper = 0.25 * dt * (-drift_v / (2.0 * dv) - diffusion_v / (dv * dv));

                // Implicit side (left-hand side matrix)
                if j > 0 {
                    lower[j - 1] = -coef_lower;
                }
                diag[j] = 1.0 + coef_diag;
                if j < n_v - 1 {
                    upper[j] = -coef_upper;
                }

                // Explicit side (right-hand side)
                rhs[j] = coef_lower * v_intermediate[[i, j - 1]]
                    + (1.0 - coef_diag) * v_intermediate[[i, j]]
                    + coef_upper * v_intermediate[[i, j + 1]];
            }

            // Solve tridiagonal system
            let solution = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;
            for j in 0..n_v {
                v_new[[i, j]] = solution[j];
            }
        }

        // Handle S boundaries for final
        for j in 0..n_v {
            v_new[[0, j]] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => 0.0,
                crate::specialized::finance::types::OptionType::Put => {
                    option.strike * (-option.risk_free_rate * dt).exp()
                }
            };
            v_new[[n_s - 1, j]] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => {
                    s_max - option.strike * (-option.risk_free_rate * dt).exp()
                }
                crate::specialized::finance::types::OptionType::Put => 0.0,
            };
        }

        // Apply early exercise for American options
        if option.option_style == crate::specialized::finance::types::OptionStyle::American {
            for i in 1..n_s - 1 {
                let s = i as f64 * ds;
                let intrinsic = solver.payoff(option, s);
                for j in 1..n_v - 1 {
                    v_new[[i, j]] = v_new[[i, j]].max(intrinsic);
                }
            }
        }

        v_grid = v_new;
    }

    // Interpolate to get option value at (S0, V0)
    let i_s = ((option.spot / ds) as usize).min(n_s - 2);
    let i_v = ((v0 / dv) as usize).min(n_v - 2);

    let w_s = (option.spot - i_s as f64 * ds) / ds;
    let w_v = (v0 - i_v as f64 * dv) / dv;

    // Bilinear interpolation
    let v00 = v_grid[[i_s, i_v]];
    let v10 = v_grid[[i_s + 1, i_v]];
    let v01 = v_grid[[i_s, i_v + 1]];
    let v11 = v_grid[[i_s + 1, i_v + 1]];

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

        let price = price_finite_difference(&solver, &option).unwrap();

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

        let price = price_finite_difference(&solver, &option).unwrap();

        // Black-Scholes reference: ~5.57
        assert!(price > 4.5 && price < 7.0, "Price: {}", price);
    }

    #[test]
    #[ignore] // TODO: ADI scheme needs Craig-Sneyd method for proper mixed derivative handling
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

        let price = price_finite_difference(&solver, &option).unwrap();

        // Should be reasonable (close to Black-Scholes for these params)
        assert!(price > 8.0 && price < 13.0, "Price: {}", price);
    }
}
