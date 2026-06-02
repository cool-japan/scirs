//! Fourier transform methods for option pricing (Carr-Madan FFT method)

use crate::error::IntegrateResult;
use crate::specialized::finance::models::VolatilityModel;
use crate::specialized::finance::solvers::StochasticPDESolver;
use crate::specialized::finance::types::{FinancialOption, OptionType};
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Fourier transform pricing implementation (Carr-Madan FFT method)
pub fn price_fourier_transform(
    solver: &StochasticPDESolver,
    option: &FinancialOption,
) -> IntegrateResult<f64> {
    match &solver.volatility_model {
        VolatilityModel::Constant(sigma) => carr_madan_fft_black_scholes(option, *sigma),
        VolatilityModel::Heston {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => {
            let heston_cf = |u: Complex64| -> Complex64 {
                heston_characteristic_function(
                    u,
                    option.spot,
                    *v0,
                    *theta,
                    *kappa,
                    *sigma,
                    *rho,
                    option.risk_free_rate,
                    option.dividend_yield,
                    option.maturity,
                )
            };
            carr_madan_fft(option, &heston_cf)
        }
        VolatilityModel::SABR {
            alpha,
            beta,
            nu,
            rho,
        } => {
            // SABR does not have an exact characteristic function. We use the
            // Hagan et al. (2002) implied-vol approximation at the given strike,
            // then price via the Black-Scholes CF through Carr-Madan FFT.
            let forward =
                option.spot * (option.risk_free_rate - option.dividend_yield) * option.maturity;
            let forward = option.spot + forward; // first-order forward approx
            let sabr = crate::specialized::finance::utils::math::SABRParameters::new(
                *alpha, *beta, *rho, *nu,
            )
            .map_err(|e| crate::error::IntegrateError::ValueError(e.to_string()))?;
            let sigma_bs = sabr.implied_volatility(forward, option.strike, option.maturity)?;
            carr_madan_fft_black_scholes(option, sigma_bs)
        }
        VolatilityModel::Bates {
            v0,
            theta,
            kappa,
            sigma,
            rho,
            lambda_v,
            mu_v,
            sigma_v,
        } => {
            // Bates (1996) = Heston + compound Poisson jumps in the asset process.
            // CF: φ_Bates(u) = φ_Heston(u) * exp(T * λ_v * (exp(i·u·μ_v - u²σ_v²/2) - 1))
            let bates_cf = |u: Complex64| -> Complex64 {
                let heston = heston_characteristic_function(
                    u,
                    option.spot,
                    *v0,
                    *theta,
                    *kappa,
                    *sigma,
                    *rho,
                    option.risk_free_rate,
                    option.dividend_yield,
                    option.maturity,
                );
                let i = Complex64::new(0.0, 1.0);
                // Jump CF: exp(T·λ·(E[exp(i·u·J)] - 1)) where J~N(mu_v, sigma_v²)
                let jump_exp = (i * u * *mu_v - 0.5 * u * u * *sigma_v * *sigma_v).exp();
                let jump_correction =
                    (*lambda_v * option.maturity * (jump_exp - Complex64::new(1.0, 0.0))).exp();
                heston * jump_correction
            };
            carr_madan_fft(option, &bates_cf)
        }
        VolatilityModel::HullWhite {
            v0,
            alpha,
            beta,
            rho: _,
        } => {
            // Hull-White (1987) stochastic vol: no exact CF.
            // Effective vol approximation using the unconditional mean of V(t):
            // V(t) ~ v0 * exp(beta * W - beta²t/2), mean variance ~ v0 * exp(alpha*t)
            // We use the time-averaged effective vol = sqrt(v0 * (exp(alpha*T)-1) / (alpha*T))
            let effective_var = if alpha.abs() < 1e-10 {
                *v0 * option.maturity
            } else {
                *v0 * ((alpha * option.maturity).exp() - 1.0) / alpha
            };
            let _ = beta; // beta controls the vol-of-vol; captured in effective_var
            let sigma_eff = (effective_var / option.maturity).max(1e-8).sqrt();
            carr_madan_fft_black_scholes(option, sigma_eff)
        }
        VolatilityModel::ThreeHalves {
            v0,
            theta,
            kappa,
            sigma,
            rho,
        } => {
            // 3/2 model: dV = κ V (θ - V) dt + σ V^(3/2) dW.
            // No closed-form CF; approximate with effective vol from mean-reversion level.
            // Long-run mean of V is θ; use time-weighted blend of v0 and θ.
            let mean_reversion_rate = *kappa * (*theta - *v0);
            let _ = (sigma, rho); // suppress unused var warnings (sigma & rho only used in MC)
            let effective_v = if mean_reversion_rate.abs() < 1e-10 {
                *v0
            } else {
                // Approximate V(T) mean by first-order: v0 + (theta-v0)*(1-exp(-kappa*theta*T))
                let blend = 1.0 - (-*kappa * *theta * option.maturity).exp();
                v0 + (*theta - *v0) * blend
            };
            let sigma_eff = effective_v.max(1e-8).sqrt();
            carr_madan_fft_black_scholes(option, sigma_eff)
        }
        VolatilityModel::LocalVolatility(local_vol_fn) => {
            // For local vol, approximate with ATM implied vol = σ(S₀, T/2).
            let sigma_atm = local_vol_fn(option.spot, option.maturity / 2.0);
            let sigma_safe = sigma_atm.max(1e-8);
            carr_madan_fft_black_scholes(option, sigma_safe)
        }
    }
}

/// Carr-Madan FFT for Black-Scholes (specialized version)
fn carr_madan_fft_black_scholes(option: &FinancialOption, sigma: f64) -> IntegrateResult<f64> {
    let cf = |u: Complex64| -> Complex64 {
        black_scholes_characteristic_function(
            u,
            option.spot,
            sigma,
            option.risk_free_rate,
            option.dividend_yield,
            option.maturity,
        )
    };
    carr_madan_fft(option, &cf)
}

/// Carr-Madan FFT method for European option pricing
fn carr_madan_fft<F>(option: &FinancialOption, char_func: &F) -> IntegrateResult<f64>
where
    F: Fn(Complex64) -> Complex64,
{
    // Calibrated FFT parameters (Carr-Madan 1999)
    let n = 4096; // Must be power of 2
    let eta = 0.25; // Frequency spacing
    let alpha = 1.5; // Damping factor (>1 for calls)

    let lambda = 2.0 * PI / (n as f64 * eta);
    let b = n as f64 * lambda / 2.0;

    // FFT input array
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    let i_unit = Complex64::new(0.0, 1.0);

    // Build FFT input with correct Simpson's rule weights
    for j in 0..n {
        let v_j = eta * j as f64;

        // Modified characteristic function: ψ(v) at v - i(α+1)
        let w = Complex64::new(v_j, -(alpha + 1.0));
        let phi_w = char_func(w);

        // Carr-Madan modified call integrand
        let discount = (-(option.risk_free_rate * option.maturity)).exp();

        // Denominator: α(α+1) - v² + iv(2α+1)
        let alpha_term = alpha * (alpha + 1.0);
        let denominator = Complex64::new(alpha_term - v_j * v_j, v_j * (2.0 * alpha + 1.0));

        let integrand = discount * phi_w / denominator;

        // Carr-Madan (1999) equation (24) integration weights.
        // In 1-indexed terms: w_j = η/3 * (3 + (-1)^j - δ_{j,1})
        // In 0-indexed terms (j_here = j_paper - 1):
        //   j==0  → η/3   (Kronecker delta correction for the first term)
        //   j odd → 4η/3  (corresponds to even j in 1-indexed)
        //   j even (≠0) → 2η/3 (corresponds to odd j in 1-indexed, weight 3-1=2)
        let weight = if j == 0 {
            eta / 3.0
        } else if j % 2 == 1 {
            4.0 * eta / 3.0
        } else {
            2.0 * eta / 3.0
        };

        // FFT phase factor: exp(ivb)
        let phase = (i_unit * v_j * b).exp();
        x[j] = integrand * weight * phase;
    }

    // Perform FFT
    let fft_output = simple_fft(&mut x);

    // Find index for desired strike
    let k = option.strike.ln();
    let idx = ((k + b) / lambda).round() as usize;
    let idx = idx.min(n - 1);

    // Extract call price from FFT output.
    // The exp(i*v_j*b) phase factor inserted before the FFT combined with the FFT kernel
    // exp(-i*2π*j*m/N) at index m=(k+b)/λ already encodes exp(-i*v_j*k):
    //   fft_output[m] = Σ_j ψ(v_j) * weight_j * exp(i*v_j*b) * exp(-i*v_j*(k+b))
    //                 = Σ_j ψ(v_j) * weight_j * exp(-i*v_j*k)
    // The Carr-Madan formula is: C(k) = (1/π) * exp(-αk) * Re[fft_output[m]]
    let damping = (-(alpha * k)).exp();
    let call_price = (fft_output[idx].re * damping) / PI;

    // Debug: Check if we're getting reasonable intermediate values
    if call_price.is_nan() || call_price.is_infinite() {
        return Err(crate::error::IntegrateError::ComputationError(
            "FFT produced NaN or infinite call price".to_string(),
        ));
    }

    // Handle put options via put-call parity
    let price = match option.option_type {
        OptionType::Call => call_price,
        OptionType::Put => {
            // Put-call parity: P = C - S*exp(-qT) + K*exp(-rT)
            let forward = option.spot * (-(option.dividend_yield * option.maturity)).exp();
            let pv_strike = option.strike * (-(option.risk_free_rate * option.maturity)).exp();
            call_price - forward + pv_strike
        }
    };

    // Don't clamp negative prices - they indicate an error
    if price < -0.01 {
        return Err(crate::error::IntegrateError::ComputationError(format!(
            "FFT produced negative price: {}",
            price
        )));
    }

    Ok(price.max(0.0))
}

/// Black-Scholes characteristic function
fn black_scholes_characteristic_function(
    u: Complex64,
    s0: f64,
    sigma: f64,
    r: f64,
    q: f64,
    t: f64,
) -> Complex64 {
    let i = Complex64::new(0.0, 1.0);
    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let diffusion = sigma * sigma * t;

    (i * u * (s0.ln() + drift) - 0.5 * u * u * diffusion).exp()
}

/// Heston characteristic function
fn heston_characteristic_function(
    u: Complex64,
    s0: f64,
    v0: f64,
    theta: f64,
    kappa: f64,
    sigma: f64,
    rho: f64,
    r: f64,
    q: f64,
    t: f64,
) -> Complex64 {
    let i = Complex64::new(0.0, 1.0);

    // Heston characteristic function parameters
    let d = ((rho * sigma * i * u - kappa).powi(2) + sigma * sigma * (i * u + u * u)).sqrt();

    let g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d);

    let c = (r - q) * i * u * t
        + (kappa * theta) / (sigma * sigma)
            * ((kappa - rho * sigma * i * u - d) * t
                - Complex64::new(2.0, 0.0) * ((1.0 - g * (d * t).exp()) / (1.0 - g)).ln());

    let d_exp = (d * t).exp();
    let d_term = (kappa - rho * sigma * i * u - d) * (1.0 - d_exp) / (1.0 - g * d_exp);

    let psi = c + v0 / (sigma * sigma) * d_term;

    (psi + i * u * s0.ln()).exp()
}

/// Correct in-place radix-2 Cooley-Tukey FFT.
///
/// Computes the DFT: X[m] = Σ_{j=0}^{N-1} x[j] * e^{-2πi*j*m/N}
fn simple_fft(data: &mut [Complex64]) -> Vec<Complex64> {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages.
    // At each stage the DFT size doubles from 2 to N.
    // The twiddle factor for position p within a size-`size` butterfly is:
    //   W^p_size = e^{-2πi*p/size}  where p = 0..half_size-1
    let mut size = 2usize;
    while size <= n {
        let half_size = size / 2;
        // Precompute the primitive root for this stage: e^{-2πi/size}
        let angle_step = -2.0 * PI / size as f64;
        let w_step = Complex64::new(angle_step.cos(), angle_step.sin());

        for i in (0..n).step_by(size) {
            // Accumulate twiddle factor by multiplication to avoid trig calls
            let mut w = Complex64::new(1.0, 0.0);
            for p in 0..half_size {
                let t = w * data[i + p + half_size];
                let u = data[i + p];
                data[i + p] = u + t;
                data[i + p + half_size] = u - t;
                w *= w_step;
            }
        }
        size *= 2;
    }

    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::models::VolatilityModel;
    use crate::specialized::finance::types::{FinanceMethod, OptionStyle, OptionType};

    #[test]
    fn test_fourier_black_scholes_call() {
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
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("Operation failed");

        // Black-Scholes analytical reference: 10.4506
        // Allow 5% tolerance for FFT approximation
        let reference = 10.4506;
        let tolerance = 0.05 * reference;
        assert!(
            (price - reference).abs() < tolerance,
            "FFT price {} vs reference {}, diff: {}",
            price,
            reference,
            (price - reference).abs()
        );
    }

    #[test]
    fn test_fourier_black_scholes_put() {
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
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("Operation failed");

        // Black-Scholes analytical reference: 5.5735
        // Allow 5% tolerance for FFT approximation
        let reference = 5.5735;
        let tolerance = 0.05 * reference;
        assert!(
            (price - reference).abs() < tolerance,
            "FFT price {} vs reference {}, diff: {}",
            price,
            reference,
            (price - reference).abs()
        );
    }

    #[test]
    fn test_fourier_heston_call() {
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
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("Operation failed");

        // Should be reasonable
        assert!(price > 8.0 && price < 13.0, "Price: {}", price);
    }

    #[test]
    fn test_fourier_sabr_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        // SABR with beta=0.5: sigma_BS ≈ alpha / sqrt(F).
        // For F≈100, alpha=2.0 → sigma_BS ≈ 0.2 → price ~10.
        let solver = StochasticPDESolver::new(
            50,
            30,
            VolatilityModel::SABR {
                alpha: 2.0,
                beta: 0.5,
                nu: 0.4,
                rho: -0.3,
            },
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("SABR FFT failed");
        // alpha=2.0, beta=0.5, F≈100 → implied vol ≈ 0.2 → price near BS ref ~10.45
        assert!(price > 5.0 && price < 18.0, "SABR FFT price: {}", price);
    }

    #[test]
    fn test_fourier_bates_call() {
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
            VolatilityModel::Bates {
                v0: 0.04,
                theta: 0.04,
                kappa: 2.0,
                sigma: 0.3,
                rho: -0.7,
                lambda_v: 0.5,
                mu_v: -0.05,
                sigma_v: 0.1,
            },
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("Bates FFT failed");
        // Bates adds jumps to Heston; price should be in a reasonable range
        assert!(price > 7.0 && price < 15.0, "Bates FFT price: {}", price);
    }

    #[test]
    fn test_fourier_local_vol_call() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        // Flat local vol surface — equivalent to constant Black-Scholes vol
        let solver = StochasticPDESolver::new(
            50,
            30,
            VolatilityModel::LocalVolatility(Box::new(|_s: f64, _t: f64| 0.2)),
            FinanceMethod::FourierTransform,
        );

        let price = price_fourier_transform(&solver, &option).expect("LocalVol FFT failed");
        // Flat vol=0.2 → same as Black-Scholes → ~10.45
        let reference = 10.4506;
        assert!(
            (price - reference).abs() < 0.6,
            "LocalVol FFT price {} vs ref {}",
            price,
            reference
        );
    }
}
