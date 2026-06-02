//! Monte Carlo methods for option pricing

use crate::error::IntegrateResult;
use crate::specialized::finance::models::VolatilityModel;
use crate::specialized::finance::solvers::StochasticPDESolver;
use crate::specialized::finance::types::{FinancialOption, OptionStyle, OptionType};
use scirs2_core::random::{Distribution, Rng, StandardNormal, Uniform};

/// Monte Carlo pricing implementation with variance reduction techniques
pub fn price_monte_carlo(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    match &solver.volatility_model {
        VolatilityModel::Constant(sigma) => {
            monte_carlo_black_scholes(option, *sigma, n_paths, antithetic)
        }
        VolatilityModel::Heston {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => monte_carlo_heston(
            option, *v0, *theta, *kappa, *sigma, *rho, n_paths, antithetic,
        ),
        VolatilityModel::SABR {
            alpha,
            beta,
            nu,
            rho,
        } => monte_carlo_sabr(option, *alpha, *beta, *nu, *rho, n_paths, antithetic),
        VolatilityModel::HullWhite {
            v0,
            alpha,
            beta,
            rho,
        } => monte_carlo_hull_white(option, *v0, *alpha, *beta, *rho, n_paths, antithetic),
        VolatilityModel::ThreeHalves {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => monte_carlo_three_halves(
            option, *v0, *theta, *kappa, *sigma, *rho, n_paths, antithetic,
        ),
        VolatilityModel::Bates {
            v0,
            theta,
            kappa,
            sigma,
            rho,
            lambda_v,
            mu_v,
            sigma_v,
        } => monte_carlo_bates(
            option, *v0, *theta, *kappa, *sigma, *rho, *lambda_v, *mu_v, *sigma_v, n_paths,
            antithetic,
        ),
        VolatilityModel::LocalVolatility(local_vol_fn) => {
            monte_carlo_local_vol(option, local_vol_fn, n_paths, antithetic)
        }
    }
}

/// Monte Carlo pricing for Black-Scholes model
fn monte_carlo_black_scholes(
    option: &FinancialOption,
    sigma: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0; // Daily time steps
    let n_steps = 252;
    let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        // Standard path
        let mut s = option.spot;
        let mut path_sum = 0.0; // For Asian options

        for _ in 0..n_steps {
            let z: f64 = rng.sample(normal);
            s *= (drift + vol_sqrt_dt * z).exp();
            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        // Antithetic path (use -z)
        if antithetic {
            let mut s_anti = option.spot;
            let mut path_sum_anti = 0.0;

            for _ in 0..n_steps {
                let z: f64 = rng.sample(normal);
                s_anti *= (drift - vol_sqrt_dt * z).exp(); // Note: -z
                path_sum_anti += s_anti;
            }

            let payoff_anti = calculate_payoff(option, s_anti, path_sum_anti / n_steps as f64);
            payoff_sum += payoff_anti;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    let discounted_payoff =
        (payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp();

    Ok(discounted_payoff)
}

/// Monte Carlo pricing for Heston stochastic volatility model
fn monte_carlo_heston(
    option: &FinancialOption,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0;
    let n_steps = 252;
    let sqrt_dt = dt.sqrt();
    let sqrt_1_minus_rho2 = (1.0 - rho * rho).sqrt();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        // Standard path
        let mut s = option.spot;
        let mut v = v0.max(0.0);
        let mut path_sum = 0.0;

        for _ in 0..n_steps {
            let z1: f64 = rng.sample(normal);
            let z2: f64 = rng.sample(normal);

            // Correlated Brownian motions
            let dw_s = z1;
            let dw_v = rho * z1 + sqrt_1_minus_rho2 * z2;

            // Update variance (Euler-Maruyama with full truncation)
            let sqrt_v = v.sqrt();
            v += kappa * (theta - v) * dt + sigma * sqrt_v * sqrt_dt * dw_v;
            v = v.max(0.0); // Ensure non-negative variance

            // Update stock price
            let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v) * dt;
            s *= (drift + sqrt_v * sqrt_dt * dw_s).exp();
            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        // Antithetic path
        if antithetic {
            let mut s_anti = option.spot;
            let mut v_anti = v0.max(0.0);
            let mut path_sum_anti = 0.0;

            for _ in 0..n_steps {
                let z1: f64 = rng.sample(normal);
                let z2: f64 = rng.sample(normal);

                let dw_s = -z1; // Antithetic
                let dw_v = -rho * z1 - sqrt_1_minus_rho2 * z2; // Antithetic

                let sqrt_v = v_anti.sqrt();
                v_anti += kappa * (theta - v_anti) * dt + sigma * sqrt_v * sqrt_dt * dw_v;
                v_anti = v_anti.max(0.0);

                let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v_anti) * dt;
                s_anti *= (drift + sqrt_v * sqrt_dt * dw_s).exp();
                path_sum_anti += s_anti;
            }

            let payoff_anti = calculate_payoff(option, s_anti, path_sum_anti / n_steps as f64);
            payoff_sum += payoff_anti;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    let discounted_payoff =
        (payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp();

    Ok(discounted_payoff)
}

/// Monte Carlo pricing for SABR stochastic volatility model.
///
/// The SABR model is: dF = α F^β dW₁,  dα = ν α dW₂,  with corr(dW₁,dW₂)=ρ.
/// We simulate the forward F and stochastic vol α using the Euler-Maruyama scheme
/// and then discount the payoff at the risk-free rate.
fn monte_carlo_sabr(
    option: &FinancialOption,
    alpha0: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0;
    let n_steps = 252;
    let sqrt_dt = dt.sqrt();
    let sqrt_1_minus_rho2 = (1.0 - rho * rho).sqrt();

    // SABR forward price starts at F₀ ≈ S₀·exp((r-q)·T)/exp((r-q)·T) ≈ S₀ for short T
    // Use the futures price as F₀
    let f0 =
        option.spot * ((option.risk_free_rate - option.dividend_yield) * option.maturity).exp();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        // Standard path
        let mut f = f0;
        let mut alpha = alpha0;
        let mut path_sum = 0.0;

        for _ in 0..n_steps {
            let z1: f64 = rng.sample(normal);
            let z2: f64 = rng.sample(normal);
            let dw1 = z1;
            let dw2 = rho * z1 + sqrt_1_minus_rho2 * z2;

            // SABR dynamics: dF = α · |F|^β · dW₁,  dα = ν · α · dW₂
            let f_beta = f.abs().powf(beta);
            f += alpha * f_beta * sqrt_dt * dw1;
            f = f.max(1e-8); // reflection boundary
            alpha = (alpha * (nu * sqrt_dt * dw2 - 0.5 * nu * nu * dt).exp()).max(1e-8);
            path_sum += f;
        }

        // Discount payoff: SABR is a forward model, so discount by exp(-rT)
        let payoff = match option.option_type {
            OptionType::Call => (f - option.strike).max(0.0),
            OptionType::Put => (option.strike - f).max(0.0),
        };
        let avg = path_sum / n_steps as f64;
        let path_payoff = match option.option_style {
            OptionStyle::Asian => match option.option_type {
                OptionType::Call => (avg - option.strike).max(0.0),
                OptionType::Put => (option.strike - avg).max(0.0),
            },
            _ => payoff,
        };
        payoff_sum += path_payoff;

        if antithetic {
            let mut f_a = f0;
            let mut alpha_a = alpha0;
            let mut path_sum_a = 0.0;

            for _ in 0..n_steps {
                let z1: f64 = rng.sample(normal);
                let z2: f64 = rng.sample(normal);
                let dw1 = -z1;
                let dw2 = -(rho * z1 + sqrt_1_minus_rho2 * z2);

                let f_beta = f_a.abs().powf(beta);
                f_a += alpha_a * f_beta * sqrt_dt * dw1;
                f_a = f_a.max(1e-8);
                alpha_a = (alpha_a * (nu * sqrt_dt * dw2 - 0.5 * nu * nu * dt).exp()).max(1e-8);
                path_sum_a += f_a;
            }

            let payoff_a = match option.option_type {
                OptionType::Call => (f_a - option.strike).max(0.0),
                OptionType::Put => (option.strike - f_a).max(0.0),
            };
            let avg_a = path_sum_a / n_steps as f64;
            let path_payoff_a = match option.option_style {
                OptionStyle::Asian => match option.option_type {
                    OptionType::Call => (avg_a - option.strike).max(0.0),
                    OptionType::Put => (option.strike - avg_a).max(0.0),
                },
                _ => payoff_a,
            };
            payoff_sum += path_payoff_a;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    Ok((payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp())
}

/// Monte Carlo pricing for Hull-White (1987) stochastic volatility model.
///
/// Hull-White: dS = r·S·dt + V·S·dW₁,  dV = α·V·dt + β·V·dW₂,  corr=ρ.
/// V follows geometric Brownian motion (the variance is lognormal).
fn monte_carlo_hull_white(
    option: &FinancialOption,
    v0: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0;
    let n_steps = 252;
    let sqrt_dt = dt.sqrt();
    let sqrt_1_minus_rho2 = (1.0 - rho * rho).sqrt();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        // Standard path
        let mut s = option.spot;
        let mut v = v0.max(1e-8); // current instantaneous vol (not variance)
        let mut path_sum = 0.0;

        for _ in 0..n_steps {
            let z1: f64 = rng.sample(normal);
            let z2: f64 = rng.sample(normal);
            let dw1 = z1;
            let dw2 = rho * z1 + sqrt_1_minus_rho2 * z2;

            // dV = α·V·dt + β·V·dW₂ — GBM for instantaneous vol
            v *= ((alpha - 0.5 * beta * beta) * dt + beta * sqrt_dt * dw2).exp();
            v = v.max(1e-8);

            let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v * v) * dt;
            s *= (drift + v * sqrt_dt * dw1).exp();
            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        if antithetic {
            let mut s_a = option.spot;
            let mut v_a = v0.max(1e-8);
            let mut path_sum_a = 0.0;

            for _ in 0..n_steps {
                let z1: f64 = rng.sample(normal);
                let z2: f64 = rng.sample(normal);
                let dw1 = -z1;
                let dw2 = -(rho * z1 + sqrt_1_minus_rho2 * z2);

                v_a *= ((alpha - 0.5 * beta * beta) * dt + beta * sqrt_dt * dw2).exp();
                v_a = v_a.max(1e-8);

                let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v_a * v_a) * dt;
                s_a *= (drift + v_a * sqrt_dt * dw1).exp();
                path_sum_a += s_a;
            }

            let payoff_a = calculate_payoff(option, s_a, path_sum_a / n_steps as f64);
            payoff_sum += payoff_a;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    Ok((payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp())
}

/// Monte Carlo pricing for the 3/2 stochastic volatility model.
///
/// 3/2 model: dS = r·S·dt + √V·S·dW₁,  dV = κ·V·(θ-V)·dt + σ·V^(3/2)·dW₂,  corr=ρ.
fn monte_carlo_three_halves(
    option: &FinancialOption,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0;
    let n_steps = 252;
    let sqrt_dt = dt.sqrt();
    let sqrt_1_minus_rho2 = (1.0 - rho * rho).sqrt();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        let mut s = option.spot;
        let mut v = v0.max(1e-8);
        let mut path_sum = 0.0;

        for _ in 0..n_steps {
            let z1: f64 = rng.sample(normal);
            let z2: f64 = rng.sample(normal);
            let dw1 = z1;
            let dw2 = rho * z1 + sqrt_1_minus_rho2 * z2;

            // dV = κ·V·(θ-V)·dt + σ·V^(3/2)·dW₂  (Euler-Maruyama with reflection)
            v += kappa * v * (theta - v) * dt + sigma * v.powf(1.5) * sqrt_dt * dw2;
            v = v.max(1e-8);

            let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v) * dt;
            s *= (drift + v.sqrt() * sqrt_dt * dw1).exp();
            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        if antithetic {
            let mut s_a = option.spot;
            let mut v_a = v0.max(1e-8);
            let mut path_sum_a = 0.0;

            for _ in 0..n_steps {
                let z1: f64 = rng.sample(normal);
                let z2: f64 = rng.sample(normal);
                let dw1 = -z1;
                let dw2 = -(rho * z1 + sqrt_1_minus_rho2 * z2);

                v_a += kappa * v_a * (theta - v_a) * dt + sigma * v_a.powf(1.5) * sqrt_dt * dw2;
                v_a = v_a.max(1e-8);

                let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * v_a) * dt;
                s_a *= (drift + v_a.sqrt() * sqrt_dt * dw1).exp();
                path_sum_a += s_a;
            }

            let payoff_a = calculate_payoff(option, s_a, path_sum_a / n_steps as f64);
            payoff_sum += payoff_a;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    Ok((payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp())
}

/// Monte Carlo pricing for the Bates model (Heston + compound Poisson jumps).
///
/// Bates: dS = (r - q - λ·μ̄)·S·dt + √V·S·dW₁ + J·S·dN,
/// dV = κ(θ-V)dt + σ√V·dW₂, corr(dW₁,dW₂)=ρ,
/// where J~lognormal(μ_v, σ_v²) and N is Poisson with intensity λ_v.
fn monte_carlo_bates(
    option: &FinancialOption,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
    lambda_v: f64,
    mu_v: f64,
    sigma_v: f64,
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let dt = option.maturity / 252.0;
    let n_steps = 252;
    let sqrt_dt = dt.sqrt();
    let sqrt_1_minus_rho2 = (1.0 - rho * rho).sqrt();

    // Mean jump size: E[J-1] = exp(mu_v + 0.5*sigma_v^2) - 1
    let mean_jump = (mu_v + 0.5 * sigma_v * sigma_v).exp() - 1.0;

    // Expected number of jumps per step
    let lambda_dt = lambda_v * dt;

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        let mut s = option.spot;
        let mut v = v0.max(0.0);
        let mut path_sum = 0.0;

        for _ in 0..n_steps {
            let z1: f64 = rng.sample(normal);
            let z2: f64 = rng.sample(normal);
            let dw1 = z1;
            let dw2 = rho * z1 + sqrt_1_minus_rho2 * z2;

            // Variance dynamics (Heston)
            let sqrt_v = v.sqrt();
            v += kappa * (theta - v) * dt + sigma * sqrt_v * sqrt_dt * dw2;
            v = v.max(0.0);

            // Drift adjustment for jump risk: r - q - lambda*(E[J]-1)
            let drift =
                (option.risk_free_rate - option.dividend_yield - lambda_v * mean_jump - 0.5 * v)
                    * dt;
            s *= (drift + sqrt_v * sqrt_dt * dw1).exp();

            // Poisson jump: use Bernoulli approx for small lambda_dt
            let u: f64 = Uniform::new(0.0_f64, 1.0_f64)
                .expect("uniform [0,1) is always a valid range")
                .sample(&mut rng);
            if u < lambda_dt {
                // Sample log-normal jump size
                let z_j: f64 = rng.sample(normal);
                let log_j = mu_v + sigma_v * z_j;
                s *= log_j.exp();
                s = s.max(1e-8);
            }

            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        if antithetic {
            let mut s_a = option.spot;
            let mut v_a = v0.max(0.0);
            let mut path_sum_a = 0.0;

            for _ in 0..n_steps {
                let z1: f64 = rng.sample(normal);
                let z2: f64 = rng.sample(normal);
                let dw1 = -z1;
                let dw2 = -(rho * z1 + sqrt_1_minus_rho2 * z2);

                let sqrt_v_a = v_a.sqrt();
                v_a += kappa * (theta - v_a) * dt + sigma * sqrt_v_a * sqrt_dt * dw2;
                v_a = v_a.max(0.0);

                let drift = (option.risk_free_rate
                    - option.dividend_yield
                    - lambda_v * mean_jump
                    - 0.5 * v_a)
                    * dt;
                s_a *= (drift + sqrt_v_a * sqrt_dt * dw1).exp();

                let u: f64 = Uniform::new(0.0_f64, 1.0_f64)
                    .expect("uniform [0,1) is always a valid range")
                    .sample(&mut rng);
                if u < lambda_dt {
                    let z_j: f64 = rng.sample(normal);
                    let log_j = mu_v + sigma_v * z_j;
                    s_a *= log_j.exp();
                    s_a = s_a.max(1e-8);
                }

                path_sum_a += s_a;
            }

            let payoff_a = calculate_payoff(option, s_a, path_sum_a / n_steps as f64);
            payoff_sum += payoff_a;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    Ok((payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp())
}

/// Monte Carlo pricing for local volatility model.
///
/// Local vol: dS = r·S·dt + σ(S,t)·S·dW
/// We evaluate the local vol function at each step using the current (S, t).
fn monte_carlo_local_vol(
    option: &FinancialOption,
    local_vol_fn: &(dyn Fn(f64, f64) -> f64 + Send + Sync),
    n_paths: usize,
    antithetic: bool,
) -> IntegrateResult<f64> {
    let mut rng = scirs2_core::random::thread_rng();
    let normal = StandardNormal;

    let n_steps = 252;
    let dt = option.maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();

    let mut payoff_sum = 0.0;
    let effective_paths = if antithetic { n_paths / 2 } else { n_paths };

    for _ in 0..effective_paths {
        let mut s = option.spot;
        let mut path_sum = 0.0;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            let z: f64 = rng.sample(normal);
            let sigma = local_vol_fn(s, t).max(1e-8);
            let drift = (option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma) * dt;
            s *= (drift + sigma * sqrt_dt * z).exp();
            path_sum += s;
        }

        let payoff = calculate_payoff(option, s, path_sum / n_steps as f64);
        payoff_sum += payoff;

        if antithetic {
            let mut s_a = option.spot;
            let mut path_sum_a = 0.0;

            for step in 0..n_steps {
                let t = step as f64 * dt;
                let z: f64 = rng.sample(normal);
                let sigma = local_vol_fn(s_a, t).max(1e-8);
                let drift =
                    (option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma) * dt;
                s_a *= (drift - sigma * sqrt_dt * z).exp(); // antithetic: -z
                path_sum_a += s_a;
            }

            let payoff_a = calculate_payoff(option, s_a, path_sum_a / n_steps as f64);
            payoff_sum += payoff_a;
        }
    }

    let actual_paths = if antithetic { n_paths } else { effective_paths };
    Ok((payoff_sum / actual_paths as f64) * (-option.risk_free_rate * option.maturity).exp())
}

/// Calculate payoff based on option style
fn calculate_payoff(option: &FinancialOption, final_price: f64, average_price: f64) -> f64 {
    match option.option_style {
        OptionStyle::European | OptionStyle::American => match option.option_type {
            OptionType::Call => (final_price - option.strike).max(0.0),
            OptionType::Put => (option.strike - final_price).max(0.0),
        },
        OptionStyle::Asian => match option.option_type {
            OptionType::Call => (average_price - option.strike).max(0.0),
            OptionType::Put => (option.strike - average_price).max(0.0),
        },
        OptionStyle::Barrier {
            barrier,
            is_up,
            is_knock_in,
        } => {
            // Simplified barrier check (only checks final price)
            let barrier_hit = if is_up {
                final_price >= barrier
            } else {
                final_price <= barrier
            };

            let barrier_active = if is_knock_in {
                barrier_hit
            } else {
                !barrier_hit
            };

            if barrier_active {
                match option.option_type {
                    OptionType::Call => (final_price - option.strike).max(0.0),
                    OptionType::Put => (option.strike - final_price).max(0.0),
                }
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::models::VolatilityModel;
    use crate::specialized::finance::types::{FinanceMethod, OptionType};

    #[test]
    fn test_monte_carlo_european_call() {
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
            FinanceMethod::MonteCarlo {
                n_paths: 10000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 10000, true).expect("Operation failed");

        // Black-Scholes reference: ~10.45
        assert!(price > 8.0 && price < 13.0, "Price: {}", price);
    }

    #[test]
    fn test_monte_carlo_european_put() {
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
            FinanceMethod::MonteCarlo {
                n_paths: 10000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 10000, true).expect("Operation failed");

        // Black-Scholes reference: ~5.57
        assert!(price > 4.0 && price < 7.5, "Price: {}", price);
    }

    #[test]
    fn test_monte_carlo_asian_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::Asian,
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
            FinanceMethod::MonteCarlo {
                n_paths: 10000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 10000, true).expect("Operation failed");

        // Asian options are cheaper than European
        assert!(price > 3.0 && price < 8.0, "Price: {}", price);
    }

    #[test]
    fn test_monte_carlo_sabr_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        // SABR with beta=0.5 and alpha=0.2: the initial vol ≈ 0.2/sqrt(100)=0.02 → small prices.
        // Use alpha=0.2, beta=1.0 (log-normal SABR) so sigma_BS=alpha=0.2.
        let solver = StochasticPDESolver::new(
            50,
            30,
            VolatilityModel::SABR {
                alpha: 0.2,
                beta: 1.0, // log-normal limit: sigma_BS ≈ alpha = 0.2
                nu: 0.3,
                rho: -0.3,
            },
            FinanceMethod::MonteCarlo {
                n_paths: 5000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 5000, true).expect("Operation failed");
        // With beta=1, alpha=0.2 → effective vol ~0.2, price in reasonable range
        assert!(price > 3.0 && price < 20.0, "SABR MC price: {}", price);
    }

    #[test]
    fn test_monte_carlo_hull_white_call() {
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
            VolatilityModel::HullWhite {
                v0: 0.2,
                alpha: 0.0,
                beta: 0.3,
                rho: -0.5,
            },
            FinanceMethod::MonteCarlo {
                n_paths: 5000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 5000, true).expect("Operation failed");
        assert!(price > 4.0 && price < 20.0, "HullWhite price: {}", price);
    }

    #[test]
    fn test_monte_carlo_three_halves_call() {
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
            VolatilityModel::ThreeHalves {
                v0: 0.04,
                theta: 0.04,
                kappa: 2.0,
                sigma: 0.5,
                rho: -0.7,
            },
            FinanceMethod::MonteCarlo {
                n_paths: 5000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 5000, true).expect("Operation failed");
        assert!(price > 4.0 && price < 20.0, "ThreeHalves price: {}", price);
    }

    #[test]
    fn test_monte_carlo_local_vol_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        // Flat local vol surface (equivalent to constant Black-Scholes)
        let solver = StochasticPDESolver::new(
            50,
            30,
            VolatilityModel::LocalVolatility(Box::new(|_s: f64, _t: f64| 0.2)),
            FinanceMethod::MonteCarlo {
                n_paths: 10000,
                antithetic: true,
            },
        );

        let price = price_monte_carlo(&solver, &option, 10000, true).expect("Operation failed");
        // Should be similar to Black-Scholes with sigma=0.2: ~10.45
        assert!(price > 7.0 && price < 14.0, "LocalVol price: {}", price);
    }
}
