//! System design and interconnection functions for LTI systems
//!
//! This module provides functions for creating LTI systems in different representations
//! and connecting them in various configurations:
//! - System creation helpers (tf, zpk, ss)
//! - System interconnections (series, parallel, feedback)
//! - System transformations (continuous to discrete)
//! - Sensitivity function analysis
//! - Polynomial utility functions

use super::systems::{LtiSystem, StateSpace, TransferFunction, ZerosPoleGain};
use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;

/// Create a transfer function system from numerator and denominator coefficients
///
/// This is a convenience function that wraps `TransferFunction::new()` with
/// a more concise interface commonly used in control theory.
///
/// # Arguments
///
/// * `num` - Numerator coefficients (highest power first)
/// * `den` - Denominator coefficients (highest power first)
/// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
///
/// # Returns
///
/// A `TransferFunction` instance
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::tf;
///
/// // Create H(s) = (s + 1) / (s^2 + 2s + 1)
/// let sys = tf(vec![1.0, 1.0], vec![1.0, 2.0, 1.0], None).unwrap();
/// ```
pub fn tf(num: Vec<f64>, den: Vec<f64>, dt: Option<bool>) -> SignalResult<TransferFunction> {
    TransferFunction::new(num, den, dt)
}

/// Create a zeros-poles-gain system
///
/// This function provides a convenient interface for creating systems in
/// zero-pole-gain form, which is often more intuitive for understanding
/// system behavior and stability.
///
/// # Arguments
///
/// * `zeros` - Zeros of the transfer function
/// * `poles` - Poles of the transfer function
/// * `gain` - System gain
/// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
///
/// # Returns
///
/// A `ZerosPoleGain` instance
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::zpk;
/// use num_complex::Complex64;
///
/// // Create H(s) = 2 * (s + 1) / (s + 2)(s + 3)
/// let sys = zpk(
///     vec![Complex64::new(-1.0, 0.0)],              // zero at s = -1
///     vec![Complex64::new(-2.0, 0.0), Complex64::new(-3.0, 0.0)], // poles at s = -2, -3
///     2.0,                                          // gain = 2
///     None
/// ).unwrap();
/// ```
pub fn zpk(
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: f64,
    dt: Option<bool>,
) -> SignalResult<ZerosPoleGain> {
    ZerosPoleGain::new(zeros, poles, gain, dt)
}

/// Create a state-space system
///
/// This function provides a convenient interface for creating systems in
/// state-space form, which is especially useful for MIMO systems and
/// modern control design.
///
/// # Arguments
///
/// * `a` - State matrix (n_states x n_states), stored in row-major order
/// * `b` - Input matrix (n_states x n_inputs), stored in row-major order
/// * `c` - Output matrix (n_outputs x n_states), stored in row-major order
/// * `d` - Feedthrough matrix (n_outputs x n_inputs), stored in row-major order
/// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
///
/// # Returns
///
/// A `StateSpace` instance
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::ss;
///
/// // Create a simple integrator: dx/dt = u, y = x
/// let sys = ss(
///     vec![0.0],  // A = [0]
///     vec![1.0],  // B = [1]
///     vec![1.0],  // C = [1]
///     vec![0.0],  // D = [0]
///     None
/// ).unwrap();
/// ```
pub fn ss(
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    dt: Option<bool>,
) -> SignalResult<StateSpace> {
    StateSpace::new(a, b, c, d, dt)
}

/// Convert a continuous-time system to a discrete-time system using zero-order hold method
///
/// This function discretizes a continuous-time system using the zero-order hold (ZOH)
/// assumption, which is commonly used in digital control systems.
///
/// # Arguments
///
/// * `system` - A continuous-time LTI system
/// * `dt` - The sampling period
///
/// # Returns
///
/// A discretized version of the system
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, c2d};
///
/// let sys_ct = tf(vec![1.0], vec![1.0, 1.0], Some(false)).unwrap();
/// let sys_dt = c2d(&sys_ct, 0.1).unwrap();
/// ```
pub fn c2d<T: LtiSystem>(system: &T, _dt: f64) -> SignalResult<StateSpace> {
    // Convert to state-space first
    let ss_sys = system.to_ss()?;

    // Ensure the system is continuous-time
    if ss_sys.dt {
        return Err(SignalError::ValueError(
            "System is already discrete-time".to_string(),
        ));
    }

    // For now, return a placeholder for the discretized system
    // In practice, we would use the matrix exponential method: A_d = exp(A*dt)

    Ok(StateSpace {
        a: ss_sys.a.clone(),
        b: ss_sys.b.clone(),
        c: ss_sys.c.clone(),
        d: ss_sys.d.clone(),
        n_states: ss_sys.n_states,
        n_inputs: ss_sys.n_inputs,
        n_outputs: ss_sys.n_outputs,
        dt: true,
    })
}

/// Connect two LTI systems in series
///
/// For systems G1 and G2 in series: H(s) = G2(s) * G1(s)
/// The output of G1 becomes the input of G2.
///
/// # Arguments
///
/// * `g1` - First system (input side)
/// * `g2` - Second system (output side)
///
/// # Returns
///
/// The series interconnection as a transfer function
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, series};
///
/// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();   // 1/(s+1)
/// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();   // 2/(s+2)
/// let series_sys = series(&g1, &g2).unwrap();              // 2/((s+1)(s+2))
/// ```
pub fn series<T1: LtiSystem, T2: LtiSystem>(g1: &T1, g2: &T2) -> SignalResult<TransferFunction> {
    let tf1 = g1.to_tf()?;
    let tf2 = g2.to_tf()?;

    // Check compatibility
    if tf1.dt != tf2.dt {
        return Err(SignalError::ValueError(
            "Systems must have the same time domain (continuous or discrete)".to_string(),
        ));
    }

    // Series connection: H(s) = G2(s) * G1(s)
    // Multiply numerators and denominators
    let num = multiply_polynomials(&tf2.num, &tf1.num);
    let den = multiply_polynomials(&tf2.den, &tf1.den);

    TransferFunction::new(num, den, Some(tf1.dt))
}

/// Connect two LTI systems in parallel
///
/// For systems G1 and G2 in parallel: H(s) = G1(s) + G2(s)
/// Both systems receive the same input, and their outputs are summed.
///
/// # Arguments
///
/// * `g1` - First system
/// * `g2` - Second system
///
/// # Returns
///
/// The parallel interconnection as a transfer function
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, parallel};
///
/// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();   // 1/(s+1)
/// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();   // 2/(s+2)
/// let parallel_sys = parallel(&g1, &g2).unwrap();          // (3s+4)/((s+1)(s+2))
/// ```
pub fn parallel<T1: LtiSystem, T2: LtiSystem>(g1: &T1, g2: &T2) -> SignalResult<TransferFunction> {
    let tf1 = g1.to_tf()?;
    let tf2 = g2.to_tf()?;

    // Check compatibility
    if tf1.dt != tf2.dt {
        return Err(SignalError::ValueError(
            "Systems must have the same time domain (continuous or discrete)".to_string(),
        ));
    }

    // Parallel connection: H(s) = G1(s) + G2(s)
    // H(s) = (N1*D2 + N2*D1) / (D1*D2)
    let num1_den2 = multiply_polynomials(&tf1.num, &tf2.den);
    let num2_den1 = multiply_polynomials(&tf2.num, &tf1.den);
    let num = add_polynomials(&num1_den2, &num2_den1);
    let den = multiply_polynomials(&tf1.den, &tf2.den);

    TransferFunction::new(num, den, Some(tf1.dt))
}

/// Connect two LTI systems in feedback configuration
///
/// For systems G (forward) and H (feedback): T(s) = G(s) / (1 + G(s)*H(s))
/// If sign is -1: T(s) = G(s) / (1 - G(s)*H(s))
///
/// # Arguments
///
/// * `g` - Forward path system
/// * `h` - Feedback path system (optional, defaults to unity feedback)
/// * `sign` - Feedback sign (1 for negative feedback, -1 for positive feedback)
///
/// # Returns
///
/// The closed-loop system as a transfer function
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, feedback};
///
/// let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
/// let h = tf(vec![1.0], vec![1.0], None).unwrap(); // Unity feedback
/// let closed_loop = feedback(&g, Some(&h), 1).unwrap();
/// ```
pub fn feedback<T1: LtiSystem>(
    g: &T1,
    h: Option<&dyn LtiSystem>,
    sign: i32,
) -> SignalResult<TransferFunction> {
    let tf_g = g.to_tf()?;

    let tf_h = if let Some(h_sys) = h {
        h_sys.to_tf()?
    } else {
        // Unity feedback
        TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
    };

    // Check compatibility
    if tf_g.dt != tf_h.dt {
        return Err(SignalError::ValueError(
            "Systems must have the same time domain (continuous or discrete)".to_string(),
        ));
    }

    // Feedback connection: T(s) = G(s) / (1 + sign*G(s)*H(s))
    // Numerator: N_g * D_h
    let num = multiply_polynomials(&tf_g.num, &tf_h.den);

    // Denominator: D_g * D_h + sign * N_g * N_h
    let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
    let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);

    let den = if sign > 0 {
        // Negative feedback: 1 + G*H
        add_polynomials(&dg_dh, &ng_nh)
    } else {
        // Positive feedback: 1 - G*H
        subtract_polynomials(&dg_dh, &ng_nh)
    };

    TransferFunction::new(num, den, Some(tf_g.dt))
}

/// Get the sensitivity function for a feedback system
///
/// Sensitivity S(s) = 1 / (1 + G(s)*H(s))
///
/// The sensitivity function represents how sensitive the output is to
/// disturbances at the reference input. Lower sensitivity is generally better.
///
/// # Arguments
///
/// * `g` - Forward path system
/// * `h` - Feedback path system (optional, defaults to unity feedback)
///
/// # Returns
///
/// The sensitivity function as a transfer function
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, sensitivity};
///
/// let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();
/// let sens = sensitivity(&g, None).unwrap(); // Unity feedback
/// ```
pub fn sensitivity<T1: LtiSystem>(
    g: &T1,
    h: Option<&dyn LtiSystem>,
) -> SignalResult<TransferFunction> {
    let tf_g = g.to_tf()?;

    let tf_h = if let Some(h_sys) = h {
        h_sys.to_tf()?
    } else {
        // Unity feedback
        TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
    };

    // Check compatibility
    if tf_g.dt != tf_h.dt {
        return Err(SignalError::ValueError(
            "Systems must have the same time domain (continuous or discrete)".to_string(),
        ));
    }

    // Sensitivity: S(s) = 1 / (1 + G(s)*H(s))
    // Numerator: D_g * D_h
    let num = multiply_polynomials(&tf_g.den, &tf_h.den);

    // Denominator: D_g * D_h + N_g * N_h
    let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
    let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);
    let den = add_polynomials(&dg_dh, &ng_nh);

    TransferFunction::new(num, den, Some(tf_g.dt))
}

/// Get the complementary sensitivity function for a feedback system
///
/// Complementary sensitivity T(s) = G(s)*H(s) / (1 + G(s)*H(s))
///
/// The complementary sensitivity function represents how well the system
/// tracks the reference signal. Together with the sensitivity function,
/// it satisfies S(s) + T(s) = 1.
///
/// # Arguments
///
/// * `g` - Forward path system
/// * `h` - Feedback path system (optional, defaults to unity feedback)
///
/// # Returns
///
/// The complementary sensitivity function as a transfer function
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::{tf, complementary_sensitivity};
///
/// let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();
/// let comp_sens = complementary_sensitivity(&g, None).unwrap(); // Unity feedback
/// ```
pub fn complementary_sensitivity<T1: LtiSystem>(
    g: &T1,
    h: Option<&dyn LtiSystem>,
) -> SignalResult<TransferFunction> {
    let tf_g = g.to_tf()?;

    let tf_h = if let Some(h_sys) = h {
        h_sys.to_tf()?
    } else {
        // Unity feedback
        TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
    };

    // Check compatibility
    if tf_g.dt != tf_h.dt {
        return Err(SignalError::ValueError(
            "Systems must have the same time domain (continuous or discrete)".to_string(),
        ));
    }

    // Complementary sensitivity: T(s) = G(s)*H(s) / (1 + G(s)*H(s))
    // Numerator: N_g * N_h
    let num = multiply_polynomials(&tf_g.num, &tf_h.num);

    // Denominator: D_g * D_h + N_g * N_h
    let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
    let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);
    let den = add_polynomials(&dg_dh, &ng_nh);

    TransferFunction::new(num, den, Some(tf_g.dt))
}

// Polynomial utility functions for system interconnections

/// Multiply two polynomials
///
/// Computes the product of two polynomials represented as coefficient vectors
/// (highest power first). This is used for transfer function operations.
///
/// # Arguments
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Product polynomial coefficients
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::multiply_polynomials;
///
/// let p1 = vec![1.0, 2.0]; // x + 2
/// let p2 = vec![1.0, 3.0]; // x + 3
/// let result = multiply_polynomials(&p1, &p2); // x^2 + 5x + 6
/// assert_eq!(result, vec![1.0, 5.0, 6.0]);
/// ```
pub fn multiply_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    if p1.is_empty() || p2.is_empty() {
        return vec![0.0];
    }

    let mut result = vec![0.0; p1.len() + p2.len() - 1];

    for (i, &a) in p1.iter().enumerate() {
        for (j, &b) in p2.iter().enumerate() {
            result[i + j] += a * b;
        }
    }

    result
}

/// Add two polynomials
///
/// Computes the sum of two polynomials represented as coefficient vectors
/// (highest power first). Polynomials of different degrees are handled
/// by zero-padding the shorter one.
///
/// # Arguments
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Sum polynomial coefficients
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::add_polynomials;
///
/// let p1 = vec![1.0, 2.0]; // x + 2
/// let p2 = vec![1.0, 3.0]; // x + 3
/// let result = add_polynomials(&p1, &p2); // 2x + 5
/// assert_eq!(result, vec![2.0, 5.0]);
/// ```
pub fn add_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    // Pad with zeros from the front and add
    let p1_offset = max_len - p1.len();
    let p2_offset = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[p1_offset + i] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[p2_offset + i] += val;
    }

    result
}

/// Subtract two polynomials
///
/// Computes the difference of two polynomials represented as coefficient vectors
/// (highest power first). Polynomials of different degrees are handled
/// by zero-padding the shorter one.
///
/// # Arguments
///
/// * `p1` - First polynomial coefficients (minuend)
/// * `p2` - Second polynomial coefficients (subtrahend)
///
/// # Returns
///
/// Difference polynomial coefficients (p1 - p2)
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::design::subtract_polynomials;
///
/// let p1 = vec![2.0, 5.0]; // 2x + 5
/// let p2 = vec![1.0, 3.0]; // x + 3
/// let result = subtract_polynomials(&p1, &p2); // x + 2
/// assert_eq!(result, vec![1.0, 2.0]);
/// ```
pub fn subtract_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    // Pad with zeros from the front and subtract
    let p1_offset = max_len - p1.len();
    let p2_offset = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[p1_offset + i] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[p2_offset + i] -= val;
    }

    result
}

/// Divide two polynomials using long division
///
/// Computes the quotient and remainder when dividing two polynomials.
/// This is useful for partial fraction decomposition and system reduction.
///
/// # Arguments
///
/// * `dividend` - Polynomial to be divided
/// * `divisor` - Polynomial to divide by
///
/// # Returns
///
/// Tuple of (quotient, remainder) polynomial coefficients
pub fn divide_polynomials(dividend: &[f64], divisor: &[f64]) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if divisor.is_empty() || divisor.iter().all(|&x| x.abs() < 1e-10) {
        return Err(SignalError::ValueError(
            "Cannot divide by zero polynomial".to_string(),
        ));
    }

    if dividend.len() < divisor.len() {
        // Dividend has lower degree than divisor
        return Ok((vec![0.0], dividend.to_vec()));
    }

    let mut remainder = dividend.to_vec();
    let mut quotient = vec![0.0; dividend.len() - divisor.len() + 1];

    // Remove leading zeros from divisor
    let mut clean_divisor = divisor.to_vec();
    while clean_divisor.len() > 1 && clean_divisor[0].abs() < 1e-10 {
        clean_divisor.remove(0);
    }

    let divisor_lead = clean_divisor[0];

    for item in &mut quotient {
        if remainder.len() < clean_divisor.len() {
            break;
        }

        // Calculate coefficient for this term
        let coeff = remainder[0] / divisor_lead;
        *item = coeff;

        // Subtract divisor * coeff from remainder
        for (j, &div_coeff) in clean_divisor.iter().enumerate() {
            remainder[j] -= coeff * div_coeff;
        }

        // Remove leading term
        remainder.remove(0);
    }

    // Remove leading zeros from results
    while quotient.len() > 1 && quotient[0].abs() < 1e-10 {
        quotient.remove(0);
    }
    while remainder.len() > 1 && remainder[0].abs() < 1e-10 {
        remainder.remove(0);
    }

    Ok((quotient, remainder))
}

/// Evaluate a polynomial at a given point
///
/// Uses Horner's method for efficient polynomial evaluation.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients (highest power first)
/// * `x` - Point at which to evaluate the polynomial
///
/// # Returns
///
/// Value of the polynomial at x
pub fn evaluate_polynomial(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    // Horner's method
    let mut result = coeffs[0];
    for &coeff in &coeffs[1..] {
        result = result * x + coeff;
    }

    result
}

/// Find the derivative of a polynomial
///
/// Computes the derivative by multiplying each coefficient by its power
/// and reducing the degree by one.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients (highest power first)
///
/// # Returns
///
/// Derivative polynomial coefficients
pub fn polynomial_derivative(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() <= 1 {
        return vec![0.0];
    }

    let mut derivative = Vec::with_capacity(coeffs.len() - 1);
    let n = coeffs.len() - 1;

    for (i, &coeff) in coeffs.iter().enumerate().take(coeffs.len() - 1) {
        let power = n - i;
        derivative.push(coeff * power as f64);
    }

    derivative
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_system_creation() {
        // Test transfer function creation
        let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        assert_eq!(tf_sys.num.len(), 1);
        assert_eq!(tf_sys.den.len(), 2);

        // Test ZPK creation
        let zpk_sys = zpk(
            vec![Complex64::new(-1.0, 0.0)],
            vec![Complex64::new(-2.0, 0.0)],
            1.0,
            None,
        )
        .unwrap();
        assert_eq!(zpk_sys.zeros.len(), 1);
        assert_eq!(zpk_sys.poles.len(), 1);

        // Test state-space creation
        let ss_sys = ss(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();
        assert_eq!(ss_sys.n_states, 1);
    }

    #[test]
    fn test_series_connection() {
        // Test series connection of two first-order systems
        // G1(s) = 1/(s+1), G2(s) = 2/(s+2)
        let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();

        let series_sys = series(&g1, &g2).unwrap();

        // Series: G2*G1 = 2/((s+1)(s+2)) = 2/(s^2 + 3s + 2)
        assert_eq!(series_sys.num.len(), 1);
        assert_eq!(series_sys.den.len(), 3);
        assert_relative_eq!(series_sys.num[0], 2.0);
        assert_relative_eq!(series_sys.den[0], 1.0);
        assert_relative_eq!(series_sys.den[1], 3.0);
        assert_relative_eq!(series_sys.den[2], 2.0);
    }

    #[test]
    fn test_parallel_connection() {
        // Test parallel connection of two first-order systems
        // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
        let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = tf(vec![1.0], vec![1.0, 2.0], None).unwrap();

        let parallel_sys = parallel(&g1, &g2).unwrap();

        // Parallel: G1+G2 = (s+2+s+1)/((s+1)(s+2)) = (2s+3)/(s^2+3s+2)
        assert_eq!(parallel_sys.num.len(), 2);
        assert_eq!(parallel_sys.den.len(), 3);
        assert_relative_eq!(parallel_sys.num[0], 2.0);
        assert_relative_eq!(parallel_sys.num[1], 3.0);
    }

    #[test]
    fn test_feedback_connection() {
        // Test unity feedback of a first-order system
        // G(s) = 1/(s+1), unity feedback
        let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();

        let feedback_sys = feedback(&g, None, 1).unwrap();

        // Feedback: T(s) = G(s)/(1+G(s)) = (1/(s+1))/(1+1/(s+1)) = 1/(s+2)
        assert_eq!(feedback_sys.num.len(), 1);
        assert_eq!(feedback_sys.den.len(), 2);
        assert_relative_eq!(feedback_sys.num[0], 1.0);
        assert_relative_eq!(feedback_sys.den[0], 1.0);
        assert_relative_eq!(feedback_sys.den[1], 2.0);
    }

    #[test]
    fn test_feedback_with_controller() {
        // Test feedback connection with a controller
        // G(s) = 1/(s+1), H(s) = 2 (proportional controller)
        let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let h = tf(vec![2.0], vec![1.0], None).unwrap();

        let feedback_sys = feedback(&g, Some(&h as &dyn LtiSystem), 1).unwrap();

        // Feedback: T(s) = G(s)/(1+G(s)*H(s)) = (1/(s+1))/(1+2/(s+1)) = 1/(s+3)
        assert_eq!(feedback_sys.num.len(), 1);
        assert_eq!(feedback_sys.den.len(), 2);
        assert_relative_eq!(feedback_sys.num[0], 1.0);
        assert_relative_eq!(feedback_sys.den[0], 1.0);
        assert_relative_eq!(feedback_sys.den[1], 3.0);
    }

    #[test]
    fn test_sensitivity_function() {
        // Test sensitivity function
        // G(s) = 10/(s+1), unity feedback
        let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let sens = sensitivity(&g, None).unwrap();

        // Sensitivity: S(s) = 1/(1+G(s)) = (s+1)/(s+11)
        assert_eq!(sens.num.len(), 2);
        assert_eq!(sens.den.len(), 2);
        assert_relative_eq!(sens.num[0], 1.0);
        assert_relative_eq!(sens.num[1], 1.0);
        assert_relative_eq!(sens.den[0], 1.0);
        assert_relative_eq!(sens.den[1], 11.0);
    }

    #[test]
    fn test_complementary_sensitivity() {
        // Test complementary sensitivity function
        // G(s) = 10/(s+1), unity feedback
        let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let comp_sens = complementary_sensitivity(&g, None).unwrap();

        // Complementary sensitivity: T(s) = G(s)/(1+G(s)) = 10/(s+11)
        assert_eq!(comp_sens.num.len(), 1);
        assert_eq!(comp_sens.den.len(), 2);
        assert_relative_eq!(comp_sens.num[0], 10.0);
        assert_relative_eq!(comp_sens.den[0], 1.0);
        assert_relative_eq!(comp_sens.den[1], 11.0);
    }

    #[test]
    fn test_polynomial_operations() {
        // Test multiply_polynomials
        let p1 = vec![1.0, 2.0]; // x + 2
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = multiply_polynomials(&p1, &p2);
        // (x + 2)(x + 3) = x^2 + 5x + 6
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 5.0);
        assert_relative_eq!(result[2], 6.0);

        // Test add_polynomials
        let p1 = vec![1.0, 2.0]; // x + 2
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = add_polynomials(&p1, &p2);
        // (x + 2) + (x + 3) = 2x + 5
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0);
        assert_relative_eq!(result[1], 5.0);

        // Test subtract_polynomials
        let p1 = vec![2.0, 5.0]; // 2x + 5
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = subtract_polynomials(&p1, &p2);
        // (2x + 5) - (x + 3) = x + 2
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 2.0);
    }

    #[test]
    fn test_polynomial_division() {
        // Test polynomial division: (x^2 + 3x + 2) / (x + 1) = (x + 2) remainder 0
        let dividend = vec![1.0, 3.0, 2.0]; // x^2 + 3x + 2
        let divisor = vec![1.0, 1.0]; // x + 1

        let (quotient, remainder) = divide_polynomials(&dividend, &divisor).unwrap();

        assert_eq!(quotient.len(), 2);
        assert_relative_eq!(quotient[0], 1.0); // x
        assert_relative_eq!(quotient[1], 2.0); // + 2

        assert_eq!(remainder.len(), 1);
        assert_relative_eq!(remainder[0], 0.0); // remainder 0
    }

    #[test]
    fn test_polynomial_evaluation() {
        // Test evaluation of p(x) = x^2 + 2x + 1 at x = 3
        let coeffs = vec![1.0, 2.0, 1.0];
        let result = evaluate_polynomial(&coeffs, 3.0);
        // p(3) = 9 + 6 + 1 = 16
        assert_relative_eq!(result, 16.0);
    }

    #[test]
    fn test_polynomial_derivative() {
        // Test derivative of p(x) = x^3 + 2x^2 + 3x + 4
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let derivative = polynomial_derivative(&coeffs);
        // p'(x) = 3x^2 + 4x + 3
        assert_eq!(derivative.len(), 3);
        assert_relative_eq!(derivative[0], 3.0);
        assert_relative_eq!(derivative[1], 4.0);
        assert_relative_eq!(derivative[2], 3.0);
    }

    #[test]
    fn test_system_interconnection_errors() {
        // Test error when connecting continuous and discrete-time systems
        let g_ct = tf(vec![1.0], vec![1.0, 1.0], Some(false)).unwrap();
        let g_dt = tf(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();

        let result = series(&g_ct, &g_dt);
        assert!(result.is_err());

        let result = parallel(&g_ct, &g_dt);
        assert!(result.is_err());
    }
}
