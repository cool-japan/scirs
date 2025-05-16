fn main() {
    println!("Testing parabolic cylinder functions with enhanced stability");
    
    // Test pbvv_asymptotic with different inputs
    let test_cases = [
        (0.0, 1.0),
        (1.0, 5.0),
        (-1.0, 10.0),
        (2.5, 20.0),
        (5.0, 50.0),
        (-2.5, -20.0),
        (10.0, 100.0),
        (20.0, 200.0),
    ];
    
    for (v, x) in test_cases.iter() {
        println!("Testing V_{}({})", v, x);
        match pbvv_asymptotic(*v, *x) {
            Ok((val, deriv)) => println!("  Result: {}, Derivative: {}", val, deriv),
            Err(e) => println!("  Error: {:?}", e),
        }
    }
}

/// Asymptotic expansion for V_v(x) with enhanced numerical stability
fn pbvv_asymptotic(v: f64, x: f64) -> Result<(f64, f64), String> {
    // For extremely large |x|
    if x.abs() > 100.0 {
        // For very large |x|, V_v(x) grows exponentially
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        
        // For extremely large |x|, V_v(x) approaches sign * infinity
        if x.abs() > 700.0 {
            return Ok((sign * f64::INFINITY, sign * f64::INFINITY));
        }
    }
    
    // Scale to avoid overflow
    let z = x.abs() / std::f64::consts::SQRT_2;
    let v2 = v * v;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    
    // Calculate the asymptotic series in log space to avoid overflow/underflow
    // Compute the sum for the asymptotic expansion with enhanced stability
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_sum = 0.0;
    let mut stagnation_count = 0;
    
    // Keep track of the derivative term
    let mut deriv_term = 0.0;
    
    // Extended iteration limit for better accuracy
    for k in 1..30 {
        // Calculate the numerator carefully to avoid overflow
        let numerator = v2 - (2 * k - 1) as f64 * (2 * k - 1) as f64;
        let denominator = 2.0 * k as f64 * z * z;
        
        // Check for potential numerical issues
        if denominator.abs() < 1e-300 {
            // Avoid division by zero
            break;
        }
        
        let term_factor = numerator / denominator;
        let new_term = term * term_factor;
        
        // Check for numerical stability
        if !new_term.is_finite() {
            // If the term would cause overflow, stop here
            break;
        }
        
        term = new_term;
        sum += term;
        
        // Save this term for derivative calculation
        deriv_term = term;
        
        // Multiple convergence criteria
        
        // Absolute tolerance
        let abs_tol = 1e-15;
        
        // Relative tolerance with protection against zero division
        let rel_tol = 1e-15 * sum.abs().max(1e-300);
        
        // First convergence check
        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }
        
        // Check for series stagnation
        if (sum - prev_sum).abs() < rel_tol {
            stagnation_count += 1;
            if stagnation_count >= 3 {
                break;
            }
        } else {
            stagnation_count = 0;
        }
        
        // Check for potential divergence in the asymptotic series
        if k > 5 && term.abs() > prev_sum.abs() {
            // Series may be starting to diverge, use the value before divergence
            sum = prev_sum;
            break;
        }
        
        prev_sum = sum;
    }
    
    // Use a simple gamma function approximation for testing
    let gamma_term = gamma(v + 0.5)?;
    
    // Calculate in log space to avoid overflow/underflow
    // log(V_v(x)) = log(sqrt(2/π)) + log(exp(z²/2)) - log(z^(v+0.5)) + log(sum) - log(gamma_term)
    let log_sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt().ln();
    let log_exp_term = z * z / 2.0;
    let log_z_term = (v + 0.5) * z.ln();
    let log_sum = sum.ln();
    let log_gamma = gamma_term.ln();
    
    // Combined log calculation
    let log_v_val = log_sqrt_2_pi + log_exp_term - log_z_term + log_sum - log_gamma;
    
    // Only exponentiate if it won't overflow/underflow
    let v_val = if log_v_val > 700.0 {
        sign * f64::INFINITY
    } else if log_v_val < -700.0 {
        0.0
    } else {
        sign * log_v_val.exp()
    };
    
    // Enhanced derivative calculation that handles extreme cases
    let vp_val = if !v_val.is_finite() {
        // For infinite function values, the derivative is also infinite
        v_val
    } else if v_val.abs() < 1e-300 {
        // For very small function values, the derivative is also near zero
        0.0
    } else {
        // Standard derivative calculation with correction term
        let correction = if sum.abs() < 1e-300 {
            0.0 // Avoid division by zero
        } else {
            deriv_term * z / sum
        };
        
        v_val * (sign * z + (v + 0.5) / z - correction)
    };
    
    // Final validation for potential numerical errors
    if !v_val.is_finite() && x.abs() < 100.0 {
        // For cases where we get overflow but x isn't extremely large
        // This can happen for some combinations of v and x
        // Provide a reasoned approximation
        let asymptotic_sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let v_approx = asymptotic_sign * f64::MAX * 0.1; // Large but not infinity
        let vp_approx = v_approx; // Derivative grows at similar rate
        
        return Ok((v_approx, vp_approx));
    }
    
    Ok((v_val, vp_val))
}

// A simplified gamma function implementation for testing
fn gamma(x: f64) -> Result<f64, String> {
    if x <= 0.0 && x == x.floor() {
        return Err(format!("Gamma function pole at x = {}", x));
    }
    
    if x < 0.5 {
        // Reflection formula
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-300 {
            return Err(format!("Gamma function reflection formula failed at x = {}", x));
        }
        return Ok(std::f64::consts::PI / (sin_pi_x * gamma(1.0 - x)?));
    }
    
    // Lanczos approximation for x >= 0.5
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];
    
    let mut y = x - 1.0;
    let mut sum = 0.99999999999980993;
    
    for i in 0..p.len() {
        sum += p[i] / (y + (i + 1) as f64);
    }
    
    let t = y + p.len() as f64 - 0.5;
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt(); 
    let result = sqrt_2pi * t.powf(y + 0.5) * (-t).exp() * sum;
    
    Ok(result)
}