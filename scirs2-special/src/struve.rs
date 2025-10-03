//! Struve functions
//!
//! This module provides implementations of the Struve functions H_v(x) and L_v(x).
//! Struve functions are solutions to certain differential equations that arise
//! in cylindrical wave propagation problems and are related to Bessel functions.
//!
//! The implementation follows the approach used in SciPy's special module,
//! using series expansions for small arguments and asymptotic forms for large arguments.

use std::f64::consts::{FRAC_2_PI, PI};
// use scirs2_core::numeric::Float;

use crate::bessel::{j0, j1, y0, y1, yn};
use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;

// Constants
const LN_PI: f64 = 1.1447298858494002; // ln(π)

/// Compute the Struve function H_v(x)
///
/// The Struve function H_v(x) is defined as:
///
/// H_v(x) = (z/2)^(v+1) / (sqrt(π) * Γ(v+3/2)) * ∫_0_^π sin(z*cos(θ)) * (sin(θ))^(2v+1) dθ
///
/// where Γ is the gamma function.
///
/// # Arguments
///
/// * `v` - Order parameter (can be any real number)
/// * `x` - Argument (must be non-negative)
///
/// # Returns
///
/// * `f64` - Value of the Struve function H_v(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::struve;
///
/// let h0_1 = struve(0.0, 1.0).unwrap();
/// println!("H_0(1.0) = {}", h0_1);
/// ```
#[allow(dead_code)]
pub fn struve(v: f64, x: f64) -> SpecialResult<f64> {
    if x.is_nan() || v.is_nan() {
        return Err(SpecialError::DomainError("NaN input to struve".to_string()));
    }

    // Special case for x=0
    if x == 0.0 {
        if v > -1.0 {
            return Ok(0.0);
        } else if v == -1.0 {
            return Ok(FRAC_2_PI); // 2/π
        } else {
            return Ok(f64::INFINITY);
        }
    }

    // Special case for v=0, integer case
    if v == 0.0 {
        return struve_h0(x);
    }

    // Special case for v=1, integer case
    if v == 1.0 {
        return struve_h1(x);
    }

    // For small x, use the power series
    if x.abs() < 20.0 {
        return struve_series(v, x);
    }

    // For large x, use the asymptotic approximation
    struve_asymptotic(v, x)
}

/// Struve function H_0(x) for order v=0.
#[allow(dead_code)]
fn struve_h0(x: f64) -> SpecialResult<f64> {
    if x.abs() < 20.0 {
        // Use the series expansion for small x
        let mut sum: f64 = 0.0;
        let x2 = x * x;

        for k in 0..30 {
            let factor = if k == 0 { 1.0 } else { -1.0f64.powi(k as i32) };
            let term = factor * x2.powi(k as i32) / ((2.0 * k as f64 + 1.0) * fact_squared(k));
            sum += term;

            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        Ok(sum * 2.0 * x / PI)
    } else {
        // For large x, use the asymptotic approximation with Bessel functions
        let y0_val = y0(x);
        let _j0_val = j0(x);

        Ok(y0_val + 2.0 / (PI * x))
    }
}

/// Struve function H_1(x) for order v=1.
#[allow(dead_code)]
fn struve_h1(x: f64) -> SpecialResult<f64> {
    if x.abs() < 20.0 {
        // Use the series expansion for small x
        let mut sum: f64 = 0.0;
        let x2 = x * x;

        for k in 0..30 {
            let factor = if k % 2 == 0 { 1.0 } else { -1.0 };
            let term = factor * x2.powi(k as i32) / ((2.0 * k as f64 + 2.0) * fact_squared(k));
            sum += term;

            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        Ok(sum * 2.0 * x * x / PI)
    } else {
        // For large x, use the asymptotic approximation with Bessel functions
        let y1_val = y1(x);
        let _j1_val = j1(x);

        Ok(y1_val - 2.0 / (PI * x) * (1.0 - 2.0 / (x * x)))
    }
}

/// Compute the Struve function H_v(x) using series expansion.
#[allow(dead_code)]
fn struve_series(v: f64, x: f64) -> SpecialResult<f64> {
    let z = 0.5 * x;
    let z_squared = z * z;

    // From https://dlmf.nist.gov/11.2.E1
    // H_v(x) = (x/2)^(v+1) Σ_{k=0}^∞ (-1)^k (x/2)^(2k) / [Γ(k + 3/2) Γ(k + v + 3/2)]

    // Check for potential overflow or high exponents
    if z.abs() > 100.0 && v > 10.0 {
        // For large arguments with high order, the series might not converge well
        // and could lead to overflow, so use asymptotic form instead
        return struve_asymptotic(v, x);
    }

    // Special case for x near zero
    if z.abs() < 1e-15 {
        if v > -1.0 {
            return Ok(0.0);
        }
        if v == -1.0 {
            return Ok(FRAC_2_PI);
        }
        return Ok(f64::INFINITY);
    }

    let mut sum: f64 = 0.0;
    let mut k = 0;
    let mut prev_sum = 0.0;
    let mut num_equal_terms = 0;

    // Pre-compute the first gamma value outside the loop since it's used frequently
    let v_plus_1_5 = v + 1.5;

    // Use a more robust convergence approach
    loop {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let k_f64 = k as f64;

        // Compute gamma values with protection against overflow
        let g1 = if k > 170 {
            // For large k, use Stirling's approximation or log-gamma
            let log_g1 = LN_PI / 2.0 + (k_f64 + 1.5) * (k_f64 + 1.5).ln() - (k_f64 + 1.5);
            log_g1.exp()
        } else {
            gamma(k_f64 + 1.5)
        };

        let g2 = if k_f64 + v_plus_1_5 > 170.0 {
            let log_g2 = LN_PI / 2.0 + (k_f64 + v_plus_1_5) * (k_f64 + v_plus_1_5).ln()
                - (k_f64 + v_plus_1_5);
            log_g2.exp()
        } else {
            gamma(k_f64 + v_plus_1_5)
        };

        // Guard against division by zero or overflow
        if g1.is_infinite() || g2.is_infinite() || (g1 * g2).abs() < 1e-300 {
            break;
        }

        // Calculate term with handling for potential overflow
        let pow_result = if k > 100 {
            // For large k, use logarithms to compute power
            (2.0 * k as f64 * z_squared.ln()).exp()
        } else {
            z_squared.powi(k)
        };

        let term = sign * pow_result / (g1 * g2);

        // Check for NaN or Inf
        if !term.is_finite() {
            break;
        }

        sum += term;

        // Convergence check with multiple conditions
        let abs_term = term.abs();
        let abs_sum = sum.abs();

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance (avoiding division by zero)
        let rel_tol = 1e-15 * abs_sum.max(1e-300);

        // Early exit criteria if we hit machine precision
        if abs_term < abs_tol || abs_term < rel_tol {
            break;
        }

        // Check if the sum is no longer changing significantly
        if (sum - prev_sum).abs() < 1e-15 * abs_sum {
            num_equal_terms += 1;
            if num_equal_terms > 3 {
                // If sum doesn't change for several iterations, exit
                break;
            }
        } else {
            num_equal_terms = 0;
        }

        prev_sum = sum;

        // Safety limit
        if k > 70 {
            break;
        }

        k += 1;
    }

    // Final calculation with overflow protection
    let z_pow_v_plus_1 = if v + 1.0 > 700.0 / z.ln() {
        // If v is very large, this could overflow
        f64::INFINITY
    } else {
        z.powf(v + 1.0)
    };

    Ok(sum * z_pow_v_plus_1)
}

/// Compute the Struve function H_v(x) using asymptotic approximation for large x.
#[allow(dead_code)]
fn struve_asymptotic(v: f64, x: f64) -> SpecialResult<f64> {
    // Enhanced asymptotic expansion for large x
    // Uses multiple approaches based on the parameter regime

    // Safety checks
    if x.is_nan() || v.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to struve_asymptotic".to_string(),
        ));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "Positive argument expected for asymptotic approximation".to_string(),
        ));
    }

    // Choose the best asymptotic method based on parameters
    if x > 50.0 && v.abs() < x / 4.0 {
        // Use full asymptotic series for very large x and moderate v
        struve_asymptotic_series(v, x)
    } else if x > 20.0 {
        // Use Bessel-based approximation with corrections for large x
        struve_asymptotic_bessel_based(v, x)
    } else {
        // For borderline cases, use series expansion which is more reliable
        struve_series(v, x)
    }
}

/// Full asymptotic series expansion for Struve functions
#[allow(dead_code)]
fn struve_asymptotic_series(v: f64, x: f64) -> SpecialResult<f64> {
    // For large x, use the complete asymptotic expansion:
    // H_v(x) = Y_v(x) + (2/π) * Σ_{k=0}^∞ (-1)^k * U_k(v) / x^{2k+1}
    // where U_k(v) are the asymptotic coefficients

    // Get the Bessel Y function
    let bessel_y = if v == 0.0 {
        y0(x)
    } else if v == 1.0 {
        y1(x)
    } else if v.fract() == 0.0 && v.abs() < 20.0 {
        yn(v as i32, x)
    } else {
        // Use asymptotic form of Y_v for non-integer v
        struve_bessel_y_asymptotic(v, x)?
    };

    // Compute the asymptotic series correction
    let mut correction = 0.0;
    let x_inv = 1.0 / x;
    let x_inv_sq = x_inv * x_inv;
    let mut x_power = x_inv; // 1/x

    // Compute first several terms of the asymptotic series
    for k in 0..8 {
        let u_k = struve_asymptotic_coefficient(k, v);
        let term = u_k * x_power;

        correction += if k % 2 == 0 { term } else { -term };

        // Check for convergence
        if term.abs() < 1e-15 * correction.abs() || term.abs() < 1e-15 {
            break;
        }

        x_power *= x_inv_sq; // Update for next term
    }

    let result = bessel_y + (2.0 / PI) * correction;

    // Final check for NaN (which might happen from 0*∞ or ∞-∞)
    if result.is_nan() {
        Err(SpecialError::DomainError(
            "Result is NaN in asymptotic approximation".to_string(),
        ))
    } else {
        Ok(result)
    }
}

/// Bessel-based approximation with higher-order corrections
#[allow(dead_code)]
fn struve_asymptotic_bessel_based(v: f64, x: f64) -> SpecialResult<f64> {
    // Get the Bessel Y function
    let bessel_y = if v == 0.0 {
        y0(x)
    } else if v == 1.0 {
        y1(x)
    } else if v.fract() == 0.0 && v.abs() < 20.0 {
        yn(v as i32, x)
    } else {
        struve_bessel_y_asymptotic(v, x)?
    };

    // Enhanced correction term with higher-order terms
    let correction = struve_correction_term_enhanced(v, x)?;

    Ok(bessel_y + correction)
}

/// Asymptotic form of Y_v(x) for non-integer v
#[allow(dead_code)]
fn struve_bessel_y_asymptotic(v: f64, x: f64) -> SpecialResult<f64> {
    // For large x: Y_v(x) ~ sqrt(2/(πx)) * sin(x - vπ/2 - π/4)
    if x < 10.0 {
        return Err(SpecialError::DomainError(
            "x too small for Y_v asymptotic approximation".to_string(),
        ));
    }

    let phase = x - v * PI / 2.0 - PI / 4.0;
    let amplitude = (2.0 / (PI * x)).sqrt();

    // Add first-order correction for better accuracy
    let correction = -(v * v - 0.25) / (8.0 * x);

    Ok(amplitude * (phase.sin() + correction * phase.cos()))
}

/// Compute asymptotic coefficients U_k(v) for the Struve function expansion
#[allow(dead_code)]
fn struve_asymptotic_coefficient(k: usize, v: f64) -> f64 {
    // The coefficients U_k(v) in the asymptotic expansion
    // These are related to the Euler polynomials and Bernoulli numbers

    match k {
        0 => {
            // U_0(v) = (4v^2 - 1) / 8
            (4.0 * v * v - 1.0) / 8.0
        }
        1 => {
            // U_1(v) = ((4v^2 - 1)(4v^2 - 9)) / 128
            let v2 = v * v;
            (4.0 * v2 - 1.0) * (4.0 * v2 - 9.0) / 128.0
        }
        2 => {
            // U_2(v) = ((4v^2 - 1)(4v^2 - 9)(4v^2 - 25)) / 3072
            let v2 = v * v;
            (4.0 * v2 - 1.0) * (4.0 * v2 - 9.0) * (4.0 * v2 - 25.0) / 3072.0
        }
        3 => {
            // U_3(v) = higher-order term
            let v2 = v * v;
            let factor1 = 4.0 * v2 - 1.0;
            let factor2 = 4.0 * v2 - 9.0;
            let factor3 = 4.0 * v2 - 25.0;
            let factor4 = 4.0 * v2 - 49.0;
            factor1 * factor2 * factor3 * factor4 / 73728.0
        }
        _ => {
            // For higher orders, use the general recurrence relation
            // U_k(v) = product of (4v^2 - (2n-1)^2) for n=1 to k+1, divided by appropriate factorial
            let mut product = 1.0;
            let v2 = v * v;

            for n in 1..=(k + 1) {
                let odd_num = (2 * n - 1) as f64;
                let odd_square = odd_num * odd_num;
                product *= 4.0 * v2 - odd_square;
            }

            // Divide by 8^{k+1} * (2k+1)!/(k!)
            let denominator = 8.0_f64.powi(k as i32 + 1) * factorial_ratio(2 * k + 1, k);
            product / denominator
        }
    }
}

/// Enhanced correction term with multiple orders
#[allow(dead_code)]
fn struve_correction_term_enhanced(v: f64, x: f64) -> SpecialResult<f64> {
    if v == 0.0 {
        // H_0(x) special case with higher-order corrections
        let leading = 2.0 / (PI * x);
        let correction = -1.0 / (2.0 * PI * x.powi(3)); // Next order term
        Ok(leading + correction)
    } else if v == 1.0 {
        // H_1(x) special case with higher-order corrections
        let leading = -2.0 / (PI * x) * (1.0 - 2.0 / (x * x));
        let correction = 8.0 / (3.0 * PI * x.powi(5)); // Next order term
        Ok(leading + correction)
    } else {
        // General case: compute (2/π) * (x/2)^{v-1} / [Γ(v+1/2) * √π]
        // with higher-order corrections

        // Use log space for stability
        let log_gamma_v_plus_half = log_gamma(v + 0.5)?;
        let log_sqrt_pi = 0.5 * PI.ln();
        let log_x_half_pow = (v - 1.0) * (0.5 * x).ln();

        let log_leading = (2.0 / PI).ln() + log_x_half_pow - log_gamma_v_plus_half - log_sqrt_pi;

        // Check for overflow
        if log_leading > 700.0 {
            return Ok(f64::INFINITY);
        } else if log_leading < -700.0 {
            return Ok(0.0);
        }

        let leading_term = log_leading.exp();

        // Add first-order correction: multiply by (1 - (v^2 - 1/4)/(2x^2))
        let correction_factor = 1.0 - (v * v - 0.25) / (2.0 * x * x);

        Ok(leading_term * correction_factor)
    }
}

/// Compute factorial ratios efficiently to avoid overflow
#[allow(dead_code)]
fn factorial_ratio(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }

    let mut result = 1.0;
    for i in (k + 1)..=n {
        result *= i as f64;
    }
    result
}

/// Enhanced asymptotic approximation for integrated modified Struve function of order 1
#[allow(dead_code)]
fn it_mod_struve_1_asymptotic(x: f64) -> SpecialResult<f64> {
    // For large x, use the more sophisticated asymptotic expansion
    // that properly accounts for the oscillatory behavior and higher-order terms

    if x < 5.0 {
        return Err(SpecialError::DomainError(
            "x too small for asymptotic approximation".to_string(),
        ));
    }

    // Use the relation with L_0(x) and the asymptotic properties
    // ∫L_1(t)dt = x*L_0(x) - ∫L_0(t)dt + corrections

    let mod_struve_0 = mod_struve(0.0, x)?;
    let it_mod_struve_0 = it_mod_struve0(x)?;

    // Main asymptotic term
    let main_term = x * mod_struve_0 - it_mod_struve_0;

    // Add higher-order corrections for better accuracy
    let correction1 = 2.0 / PI; // Constant correction
    let correction2 = -1.0 / (PI * x); // 1/x correction
    let correction3 = 1.0 / (2.0 * PI * x * x); // 1/x^2 correction

    Ok(main_term + correction1 + correction2 + correction3)
}

/// Calculate the logarithm of the gamma function for large arguments.
/// This is more stable than calling gamma and then taking the log.
#[allow(dead_code)]
fn log_gamma(x: f64) -> SpecialResult<f64> {
    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "log_gamma requires positive argument".to_string(),
        ));
    }

    if x > 170.0 {
        // Use Stirling's approximation for large arguments
        // log(Γ(x)) ≈ (x-0.5)*log(x) - x + 0.5*log(2π) + 1/(12x) - ...
        let log_2pi = PI.ln() + 2.0_f64.ln();
        let result = (x - 0.5) * x.ln() - x + 0.5 * log_2pi + 1.0 / (12.0 * x);
        Ok(result)
    } else {
        // For smaller arguments, compute gamma and take log
        let g = gamma(x);
        Ok(g.ln())
    }
}

/// Compute the Modified Struve function L_v(x)
///
/// The Modified Struve function L_v(x) is related to the Struve function H_v(x) but
/// uses hyperbolic instead of trigonometric functions in the integral representation.
///
/// # Arguments
///
/// * `v` - Order parameter (can be any real number)
/// * `x` - Argument (must be non-negative)
///
/// # Returns
///
/// * `f64` - Value of the Modified Struve function L_v(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::mod_struve;
///
/// let l0_1 = mod_struve(0.0, 1.0).unwrap();
/// println!("L_0(1.0) = {}", l0_1);
/// ```
#[allow(dead_code)]
pub fn mod_struve(v: f64, x: f64) -> SpecialResult<f64> {
    if x.is_nan() || v.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to mod_struve".to_string(),
        ));
    }

    // Special case for x=0
    if x == 0.0 {
        if v > -1.0 {
            return Ok(0.0);
        } else if v == -1.0 {
            return Ok(FRAC_2_PI); // 2/π
        } else {
            return Ok(f64::INFINITY);
        }
    }

    // Modified Struve function follows similar computation to Struve function
    // with changes to the sign pattern in the series

    if x.abs() < 20.0 {
        // Use the series expansion for small x
        let z = 0.5 * x;
        let z_squared = z * z;

        // Special case for x near zero
        if z.abs() < 1e-15 {
            if v > -1.0 {
                return Ok(0.0);
            }
            if v == -1.0 {
                return Ok(FRAC_2_PI);
            }
            return Ok(f64::INFINITY);
        }

        // From DLMF: L_v(x) = (x/2)^(v+1) Σ_{k=0}^∞ (x/2)^(2k) / [Γ(k + 3/2) Γ(k + v + 3/2)]
        let mut sum: f64 = 0.0;
        let mut k = 0;
        let mut prev_sum = 0.0;
        let mut num_equal_terms = 0;

        // Pre-compute the first gamma value outside the loop
        let v_plus_1_5 = v + 1.5;

        loop {
            let k_f64 = k as f64;

            // Compute gamma values with protection against overflow
            let g1 = if k > 170 {
                // For large k, use Stirling's approximation
                let log_g1 = LN_PI / 2.0 + (k_f64 + 1.5) * (k_f64 + 1.5).ln() - (k_f64 + 1.5);
                log_g1.exp()
            } else {
                gamma(k_f64 + 1.5)
            };

            let g2 = if k_f64 + v_plus_1_5 > 170.0 {
                let log_g2 = LN_PI / 2.0 + (k_f64 + v_plus_1_5) * (k_f64 + v_plus_1_5).ln()
                    - (k_f64 + v_plus_1_5);
                log_g2.exp()
            } else {
                gamma(k_f64 + v_plus_1_5)
            };

            // Guard against division by zero or overflow
            if g1.is_infinite() || g2.is_infinite() || (g1 * g2).abs() < 1e-300 {
                break;
            }

            // Calculate term with handling for potential overflow
            let pow_result = if k > 100 {
                // For large k, use logarithms to compute power
                (2.0 * k as f64 * z_squared.ln()).exp()
            } else {
                z_squared.powi(k)
            };

            let term = pow_result / (g1 * g2);

            // Check for NaN or Inf
            if !term.is_finite() {
                break;
            }

            sum += term;

            // Convergence check with multiple conditions
            let abs_term = term.abs();
            let abs_sum = sum.abs();

            // Absolute tolerance
            let abs_tol = 1e-15;

            // Relative tolerance (avoiding division by zero)
            let rel_tol = 1e-15 * abs_sum.max(1e-300);

            // Early exit criteria if we hit machine precision
            if abs_term < abs_tol || abs_term < rel_tol {
                break;
            }

            // Check if the sum is no longer changing significantly
            if (sum - prev_sum).abs() < 1e-15 * abs_sum {
                num_equal_terms += 1;
                if num_equal_terms > 3 {
                    // If sum doesn't change for several iterations, exit
                    break;
                }
            } else {
                num_equal_terms = 0;
            }

            prev_sum = sum;

            // Safety limit
            if k > 70 {
                break;
            }

            k += 1;
        }

        // Final calculation with overflow protection
        let z_pow_v_plus_1 = if v + 1.0 > 700.0 / z.ln() {
            // If v is very large, this could overflow
            f64::INFINITY
        } else {
            z.powf(v + 1.0)
        };

        Ok(sum * z_pow_v_plus_1)
    } else {
        // For large x, use the asymptotic approximation
        if x > 100.0 && v > 20.0 {
            // For very large x and v, the asymptotic form may not be accurate
            // Fall back to series calculation with extended precision
            let z = 0.5 * x;

            // Use a stabilized series calculation in log space
            let mut log_sum = f64::NEG_INFINITY; // log(0)
            let mut k = 0;

            while k < 50 {
                let k_f64 = k as f64;

                // Calculate log of the term
                let log_g1 = log_gamma(k_f64 + 1.5)?;
                let log_g2 = log_gamma(k_f64 + v + 1.5)?;
                let log_term = 2.0 * k as f64 * z.ln() - log_g1 - log_g2;

                // Add terms in log space using log-sum-exp trick
                if log_sum == f64::NEG_INFINITY {
                    log_sum = log_term;
                } else {
                    let max_log = log_sum.max(log_term);
                    log_sum = max_log
                        + (log_sum - max_log).exp().ln_1p()
                        + (log_term - max_log).exp().ln();
                }

                // Check for convergence in log space
                if (log_term - log_sum).exp() < 1e-15 && k > 10 {
                    break;
                }

                k += 1;
            }

            // Final result in log space, then convert back
            let log_result = log_sum + (v + 1.0) * z.ln();

            // Check for overflow before converting from log space
            if log_result > 700.0 {
                return Ok(f64::INFINITY);
            }

            return Ok(log_result.exp());
        }

        // Standard asymptotic approximation for moderate to large x
        if v == 0.0
            || v == 1.0
            || (v.fract() == 0.0 && v.abs() < 20.0)
            || (v > 0.0 && v < 100.0 && x > 10.0)
        {
            // Use the asymptotic relation between modified Struve and Bessel functions
            // L_v(x) ≈ I_v(x) - 2/(π*Γ(v+1/2)*Γ(1/2)) * (x/2)^(v-1)

            // Get the modified Bessel function I_v(x)
            let i_v = bessel_i_approximation(v, x)?;

            // Calculate the correction term with safe handling of large values
            let correction_term = if v == 0.0 {
                // For v=0, the correction is different
                2.0 / (PI * x)
            } else if v == 1.0 {
                // For v=1, another specific correction
                // For v=1, use special form that handles precision better
                let base_term = 2.0 / (PI * x);

                // Be careful with the formula (1.0 + 2.0/(x*x))
                // For large x, we can lose precision
                if x > 10.0 {
                    base_term * (1.0 + 2.0 / (x * x))
                } else {
                    // For smaller x, compute more carefully
                    let x_sq = x * x;
                    base_term * ((x_sq + 2.0) / x_sq)
                }
            } else {
                // For other values, compute in log space to avoid overflow
                let log_gamma_v_plus_half = log_gamma(v + 0.5)?;
                let log_sqrt_pi = 0.5 * PI.ln(); // log(Γ(1/2)) = log(√π)

                // Calculate (x/2)^(v-1) in log space
                let log_x_half_pow = (v - 1.0) * (0.5 * x).ln();

                // Combine all terms in log space
                let log_correction =
                    (2.0 / PI).ln() + log_x_half_pow - log_gamma_v_plus_half - log_sqrt_pi;

                // Check for overflow before converting back from log space
                if log_correction > 700.0 {
                    f64::INFINITY
                } else if log_correction < -700.0 {
                    0.0 // Effectively zero due to underflow
                } else {
                    log_correction.exp()
                }
            };

            // Combine the Bessel function and correction carefully
            // I_v can get very large for large x, so check carefully
            if i_v.is_infinite() && correction_term.is_infinite() {
                Err(SpecialError::DomainError(
                    "Indeterminate result (∞ - ∞) in mod_struve".to_string(),
                ))
            } else if i_v.is_infinite() {
                Ok(i_v) // If I_v is infinite but correction is finite, return I_v
            } else {
                let result = i_v - correction_term;

                // Final result check
                if result.is_nan() {
                    Err(SpecialError::DomainError(
                        "Result is NaN in mod_struve calculation".to_string(),
                    ))
                } else {
                    Ok(result)
                }
            }
        } else {
            // For more complex cases, use the series expansion with greater care
            // We already have a good implementation for the series, so redirect
            mod_struve(v, x)
        }
    }
}

/// Helper function to approximate the modified Bessel function I_v(x) for large arguments.
#[allow(dead_code)]
fn bessel_i_approximation(v: f64, x: f64) -> SpecialResult<f64> {
    // Modified Bessel function I_v(x) for large x
    // Using the asymptotic expansion I_v(x) ~ e^x / sqrt(2πx) * (1 + O(1/x))

    if v.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to bessel_i_approximation".to_string(),
        ));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "Argument must be positive in bessel_i_approximation".to_string(),
        ));
    }

    // Handle small x separately to avoid inaccuracies in the asymptotic formula
    if x < 5.0 {
        // Use series expansion for small x
        let mut sum = 1.0; // First term (k=0)
        let x_half = x / 2.0;
        let x_half_sq = x_half * x_half;

        if v == 0.0 {
            // I₀(x) = Σ (x/2)^(2k) / (k!)^2
            let mut term: f64 = 1.0;

            for k in 1..40 {
                term *= x_half_sq / (k as f64 * k as f64);
                sum += term;

                if term < 1e-15 * sum && k > 10 {
                    break;
                }
            }

            return Ok(sum);
        } else if v == 1.0 {
            // I₁(x) = (x/2) * Σ (x/2)^(2k) / (k!(k+1)!)
            sum = 0.0;
            let mut term = x_half;
            sum += term;

            for k in 1..40 {
                term *= x_half_sq / (k as f64 * (k + 1) as f64);
                sum += term;

                if term < 1e-15 * sum && k > 10 {
                    break;
                }
            }

            return Ok(sum);
        }
    }

    // For large x, use asymptotic formula that's accurate for large arguments
    // I_v(x) ~ e^x / sqrt(2πx) * (1 + (4v²-1)/(8x) + (4v²-1)(4v²-9)/(2!(8x)²) + ...)

    // Compute the first few terms of the expansion
    let _one_over_x = 1.0 / x;
    let v_squared = v * v;
    let mu = 4.0 * v_squared - 1.0;

    // First few terms of the asymptotic series
    let mut series = 1.0;

    if x > 10.0 {
        // For very large x, include more terms
        series += mu / (8.0 * x);

        if x > 20.0 {
            let term2 = mu * (mu - 8.0) / (2.0 * 64.0 * x * x);
            series += term2;

            if x > 40.0 {
                let term3 = mu * (mu - 8.0) * (mu - 24.0) / (6.0 * 512.0 * x * x * x);
                series += term3;
            }
        }
    }

    // Main asymptotic formula
    // Using exp() with safeguards to prevent overflow
    let result = if x > 700.0 {
        // Handle potential overflow in exp(x)
        let log_result = x - 0.5 * (2.0 * PI * x).ln() + series.ln();
        if log_result > 700.0 {
            f64::INFINITY
        } else {
            log_result.exp()
        }
    } else {
        (x.exp() / (2.0 * PI * x).sqrt()) * series
    };

    Ok(result)
}

/// Compute squared factorial for small integers.
///
/// Returns (n!)² but computed in a way that avoids overflow for larger values of n.
/// Uses a log-space calculation for large values of n.
#[allow(dead_code)]
fn fact_squared(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }

    // For small n, use direct computation to avoid logarithm overhead
    if n <= 10 {
        let mut result = 1.0;
        for i in 1..=n {
            let i_f64 = i as f64;
            result *= i_f64 * i_f64;
        }
        return result;
    }

    // For larger n, use logarithms to avoid overflow
    let mut log_result = 0.0;
    for i in 1..=n {
        log_result += 2.0 * (i as f64).ln();
    }

    // Check for overflow before exponentiating
    if log_result > 700.0 {
        // Near ln(f64::MAX)
        return f64::INFINITY;
    }

    log_result.exp()
}

/// Compute the integrated Struve function of order 0
///
/// This function computes the integral of H_0(t) from 0 to x.
///
/// # Arguments
///
/// * `x` - Upper limit of integration
///
/// # Returns
///
/// * `f64` - Value of the integrated Struve function
///
/// # Examples
///
/// ```
/// use scirs2_special::it_struve0;
///
/// let its0_1 = it_struve0(1.0).unwrap();
/// println!("∫_0_^1 H_0(t) dt = {}", its0_1);
/// ```
#[allow(dead_code)]
pub fn it_struve0(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to it_struve0".to_string(),
        ));
    }

    if x == 0.0 {
        return Ok(0.0);
    }

    // Special case for x near zero - with extended precision
    if x.abs() < 1e-15 {
        return Ok(0.0);
    }

    if x.abs() < 20.0 {
        // Use the series expansion with improved numerical stability
        let mut sum: f64 = 0.0;
        let x2 = x * x;

        // Starting conditions to ensure stable calculation
        let mut prev_sum = 0.0;
        let mut num_equal_terms = 0;

        for k in 0..50 {
            // Extended iteration limit for better precision
            let factor = if k % 2 == 0 { 1.0 } else { -1.0 };
            let k_f64 = k as f64;

            // Calculate the term with protection against overflow
            let denom1 = 2.0 * k_f64 + 1.0;
            let denom2 = 2.0 * k_f64 + 3.0;

            // Calculate term value - using scoping to ensure it's available throughout the loop
            let term = if k > 15 {
                // For large k, use log-space calculation
                // Compute x^(2k) carefully to avoid overflow
                let log_x2k = 2.0 * k_f64 * x2.ln();
                let log_fact_squared = fact_squared_log(k);
                let log_term = log_x2k - (denom1 * denom2).ln() - log_fact_squared;

                // Convert back to linear space
                factor * log_term.exp()
            } else {
                // For small k, direct calculation is stable
                let fac_sq = fact_squared(k);
                factor * x2.powi(k as i32) / (denom1 * denom2 * fac_sq)
            };

            // Skip terms that could cause numerical instability
            if !term.is_finite() {
                break;
            }

            sum += term;

            // Multiple convergence criteria for robustness
            let abs_term = term.abs();
            let abs_sum = sum.abs().max(1e-300); // Avoid division by zero

            // Absolute and relative tolerance checks
            if (abs_term < 1e-15) || (abs_term < 1e-15 * abs_sum) {
                break;
            }

            // Check if sum is stabilizing
            if (sum - prev_sum).abs() < 1e-15 * abs_sum {
                num_equal_terms += 1;
                if num_equal_terms > 3 {
                    // Exit after several iterations without significant change
                    break;
                }
            } else {
                num_equal_terms = 0;
            }

            prev_sum = sum;
        }

        Ok(sum * 2.0 * x * x / PI)
    } else {
        // For large x, use improved asymptotic approximation
        let struve_0 = struve(0.0, x)?;
        let bessel_j1 = j1(x);

        // For very large x, the subtraction can lose precision, so compute more carefully
        if x > 100.0 {
            // For very large x, struve_0 ~ y0(x) and j1(x) ~ sqrt(2/πx) * cos(x-3π/4)
            // The asymptotic behavior becomes approximately:
            // it_struve0(x) ~ (y0(x) - j1(x)/x) * x + 1.0 - 2.0/π

            // Compute in a way that preserves significant digits
            let term1 = struve_0 * x;
            let term2 = bessel_j1 * x;
            let term3 = 1.0 - 2.0 / PI;

            // Add terms in order of increasing magnitude
            let result = term3 + (term1 - term2);

            // Check for NaN (which could occur from 0*∞ or ∞-∞)
            if result.is_nan() {
                Err(SpecialError::DomainError(
                    "NaN result in it_struve0 calculation".to_string(),
                ))
            } else {
                Ok(result)
            }
        } else {
            // For moderate x, the standard formula works well
            Ok(struve_0 * x - bessel_j1 * x + 1.0 - 2.0 / PI)
        }
    }
}

/// Compute the logarithm of the squared factorial for larger values
#[allow(dead_code)]
fn fact_squared_log(n: usize) -> f64 {
    if n == 0 {
        return 0.0; // log(1) = 0
    }

    let mut result = 0.0;
    for i in 1..=n {
        result += 2.0 * (i as f64).ln();
    }

    result
}

/// Compute the second integration of the Struve function of order 0
///
/// This function computes the second integral of H_0(t) from 0 to x.
///
/// # Arguments
///
/// * `x` - Upper limit of integration
///
/// # Returns
///
/// * `f64` - Value of the second integration of the Struve function
///
/// # Examples
///
/// ```
/// use scirs2_special::it2_struve0;
///
/// let it2s0_1 = it2_struve0(1.0).unwrap();
/// println!("∫_0_^x ∫_0_^t H_0(s) ds dt = {}", it2s0_1);
/// ```
#[allow(dead_code)]
pub fn it2_struve0(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to it2_struve0".to_string(),
        ));
    }

    if x == 0.0 {
        return Ok(0.0);
    }

    if x.abs() < 20.0 {
        // Use the series expansion
        let mut sum: f64 = 0.0;
        let x2 = x * x;

        for k in 0..30 {
            let factor = if k % 2 == 0 { 1.0 } else { -1.0 };
            let term = factor * x2.powi(k as i32)
                / ((2.0 * k as f64 + 1.0)
                    * (2.0 * k as f64 + 3.0)
                    * (2.0 * k as f64 + 5.0)
                    * fact_squared(k));
            sum += term;

            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        Ok(sum * 2.0 * x * x * x / PI)
    } else {
        // For large x, use enhanced asymptotic approximation
        it_mod_struve_1_asymptotic(x)
    }
}

/// Compute the integrated modified Struve function of order 0
///
/// This function computes the integral of L_0(t) from 0 to x.
///
/// # Arguments
///
/// * `x` - Upper limit of integration
///
/// # Returns
///
/// * `f64` - Value of the integrated modified Struve function
///
/// # Examples
///
/// ```
/// use scirs2_special::it_mod_struve0;
///
/// let itl0_1 = it_mod_struve0(1.0).unwrap();
/// println!("∫_0_^1 L_0(t) dt = {}", itl0_1);
/// ```
#[allow(dead_code)]
pub fn it_mod_struve0(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to it_mod_struve0".to_string(),
        ));
    }

    if x == 0.0 {
        return Ok(0.0);
    }

    if x.abs() < 20.0 {
        // Use the series expansion
        let mut sum: f64 = 0.0;
        let x2 = x * x;

        for k in 0..30 {
            let term = x2.powi(k as i32)
                / ((2.0 * k as f64 + 1.0) * (2.0 * k as f64 + 3.0) * fact_squared(k));
            sum += term;

            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        Ok(sum * 2.0 * x * x / PI)
    } else {
        // For large x, use asymptotic approximation
        // This is a simplified version
        let mod_struve_0 = mod_struve(0.0, x)?;
        let bessel_i1 = bessel_i_approximation(1.0, x)?;

        Ok(mod_struve_0 * x - bessel_i1 * x + 2.0 / PI * (x.exp() - 1.0))
    }
}
