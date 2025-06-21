//! Linear Time-Invariant (LTI) Systems Module
//!
//! This module provides comprehensive support for Linear Time-Invariant systems analysis
//! and design. It offers three different system representations and extensive analysis tools.
//!
//! # System Representations
//!
//! - **Transfer Function**: Ratio of polynomials H(s) = N(s)/D(s)
//! - **Zero-Pole-Gain**: Factored form H(s) = K * ∏(s-zi)/∏(s-pi)
//! - **State-Space**: Matrix form dx/dt = Ax + Bu, y = Cx + Du
//!
//! # Quick Start
//!
//! ## Creating Systems
//!
//! ```rust
//! use scirs2_signal::lti::{design, systems::TransferFunction};
//! use num_complex::Complex64;
//!
//! // Transfer function: H(s) = 1/(s+1)
//! let sys1 = design::tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
//!
//! // Zero-pole-gain: H(s) = 2(s+1)/(s+2)
//! let sys2 = design::zpk(
//!     vec![Complex64::new(-1.0, 0.0)], // zeros
//!     vec![Complex64::new(-2.0, 0.0)], // poles
//!     2.0,                             // gain
//!     None
//! ).unwrap();
//!
//! // State-space: dx/dt = -x + u, y = x
//! let sys3 = design::ss(
//!     vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None
//! ).unwrap();
//! ```
//!
//! ## System Analysis
//!
//! ```rust
//! use scirs2_signal::lti::{design::{tf, ss}, analysis::{bode, analyze_controllability}};
//!
//! let sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
//!
//! // Frequency response analysis
//! let freqs = vec![0.1, 1.0, 10.0];
//! let (w, mag, phase) = bode(&sys, Some(&freqs)).unwrap();
//!
//! // State-space analysis using direct state-space creation
//! // For H(s) = 1/(s+1), state-space form: dx/dt = -x + u, y = x
//! let ss_sys = ss(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();
//! let ctrl_analysis = analyze_controllability(&ss_sys).unwrap();
//! println!("System is controllable: {}", ctrl_analysis.is_controllable);
//! ```
//!
//! ## System Interconnections
//!
//! ```rust
//! use scirs2_signal::lti::design::{tf, series, parallel, feedback};
//!
//! let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
//! let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
//!
//! // Series connection
//! let series_sys = series(&g1, &g2).unwrap();
//!
//! // Parallel connection
//! let parallel_sys = parallel(&g1, &g2).unwrap();
//!
//! // Feedback connection
//! let feedback_sys = feedback(&g1, None, 1).unwrap(); // Unity feedback
//! ```
//!
//! # Module Organization
//!
//! - [`systems`] - Core system types and trait definitions
//! - [`analysis`] - System analysis functions (Bode plots, controllability, etc.)
//! - [`design`] - System creation and interconnection functions

// Re-export all public modules
pub mod analysis;
pub mod design;
pub mod systems;

// Re-export core system types for convenience
pub use systems::{LtiSystem, StateSpace, TransferFunction, ZerosPoleGain};

// Re-export analysis functions and types
pub use analysis::{
    analyze_control_observability, analyze_controllability, analyze_observability, bode,
    complete_kalman_decomposition, compute_lyapunov_gramians, matrix_condition_number,
    systems_equivalent, ControlObservabilityAnalysis, ControllabilityAnalysis, GramianPair,
    KalmanDecomposition, KalmanStructure, ObservabilityAnalysis,
};

// Re-export design functions for convenience
pub use design::{
    add_polynomials, c2d, complementary_sensitivity, divide_polynomials, evaluate_polynomial,
    feedback, multiply_polynomials, parallel, polynomial_derivative, sensitivity, series, ss,
    subtract_polynomials, tf, zpk,
};

// Keep the system module for backward compatibility
/// Functions for creating and manipulating LTI systems
///
/// This module provides convenience functions for system creation and interconnection.
/// It is maintained for backward compatibility with existing code.
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::system;
///
/// // Create systems
/// let g1 = system::tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
/// let g2 = system::tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
///
/// // Connect systems
/// let series_connection = system::series(&g1, &g2).unwrap();
/// ```
pub mod system {
    pub use super::design::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_module_api_compatibility() {
        // Test that the main API functions are accessible

        // System creation
        let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let _zpk_sys = zpk(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();
        let ss_sys = ss(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();

        // Analysis
        let freqs = vec![1.0];
        let (w, mag, phase) = bode(&tf_sys, Some(&freqs)).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(mag.len(), 1);
        assert_eq!(phase.len(), 1);

        let ctrl_analysis = analyze_controllability(&ss_sys).unwrap();
        assert!(ctrl_analysis.is_controllable);

        // Interconnections
        let tf2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
        let series_sys = series(&tf_sys, &tf2).unwrap();
        let parallel_sys = parallel(&tf_sys, &tf2).unwrap();
        let feedback_sys = feedback(&tf_sys, None, 1).unwrap();

        // Verify these operations produce valid systems
        assert!(!series_sys.num.is_empty());
        assert!(!parallel_sys.num.is_empty());
        assert!(!feedback_sys.num.is_empty());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the system module (for backward compatibility) works
        let g1 = system::tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = system::tf(vec![2.0], vec![1.0, 2.0], None).unwrap();

        let series_connection = system::series(&g1, &g2).unwrap();
        let parallel_connection = system::parallel(&g1, &g2).unwrap();

        assert!(!series_connection.num.is_empty());
        assert!(!parallel_connection.num.is_empty());
    }

    #[test]
    fn test_system_conversions() {
        // Test conversions between different representations
        let tf_sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Convert to other representations
        let zpk_sys = tf_sys.to_zpk().unwrap();
        let ss_sys = tf_sys.to_ss().unwrap();

        // Verify basic properties are maintained
        assert_eq!(tf_sys.dt, zpk_sys.dt);
        assert_eq!(tf_sys.dt, ss_sys.dt);
    }

    #[test]
    fn test_comprehensive_analysis() {
        // Test a complete analysis workflow
        let ss_sys = ss(
            vec![-1.0, 0.0, 1.0, -2.0], // 2x2 A matrix
            vec![1.0, 0.0],             // 2x1 B matrix
            vec![1.0, 0.0],             // 1x2 C matrix
            vec![0.0],                  // 1x1 D matrix
            None,
        )
        .unwrap();

        // Structural analysis
        let ctrl_analysis = analyze_controllability(&ss_sys).unwrap();
        let obs_analysis = analyze_observability(&ss_sys).unwrap();
        let combined_analysis = analyze_control_observability(&ss_sys).unwrap();

        // Verify analysis results are consistent
        assert_eq!(
            ctrl_analysis.is_controllable,
            combined_analysis.controllability.is_controllable
        );
        assert_eq!(
            obs_analysis.is_observable,
            combined_analysis.observability.is_observable
        );
        assert_eq!(
            combined_analysis.is_minimal,
            ctrl_analysis.is_controllable && obs_analysis.is_observable
        );

        // Energy-based analysis
        let (wc, wo) = compute_lyapunov_gramians(&ss_sys).unwrap();
        assert_eq!(wc.len(), ss_sys.n_states);
        assert_eq!(wo.len(), ss_sys.n_states);

        // Complete decomposition
        let kalman_decomp = complete_kalman_decomposition(&ss_sys).unwrap();
        assert_eq!(
            kalman_decomp.co_dimension
                + kalman_decomp.c_no_dimension
                + kalman_decomp.nc_o_dimension
                + kalman_decomp.nc_no_dimension,
            ss_sys.n_states
        );
    }

    #[test]
    fn test_polynomial_utilities() {
        // Test polynomial operations
        let p1 = vec![1.0, -2.0]; // s - 2
        let p2 = vec![1.0, -3.0]; // s - 3

        let product = multiply_polynomials(&p1, &p2); // (s-2)(s-3) = s^2 - 5s + 6
        assert_eq!(product.len(), 3);
        assert_relative_eq!(product[0], 1.0);
        assert_relative_eq!(product[1], -5.0);
        assert_relative_eq!(product[2], 6.0);

        let sum = add_polynomials(&p1, &p2); // (s-2) + (s-3) = 2s - 5
        assert_eq!(sum.len(), 2);
        assert_relative_eq!(sum[0], 2.0);
        assert_relative_eq!(sum[1], -5.0);

        let diff = subtract_polynomials(&p1, &p2); // (s-2) - (s-3) = 1
        assert_eq!(diff.len(), 2);
        assert_relative_eq!(diff[0], 0.0);
        assert_relative_eq!(diff[1], 1.0);

        // Test evaluation
        let value = evaluate_polynomial(&product, 1.0); // (1)^2 - 5(1) + 6 = 2
        assert_relative_eq!(value, 2.0);

        // Test derivative
        let deriv = polynomial_derivative(&product); // d/ds(s^2 - 5s + 6) = 2s - 5
        assert_eq!(deriv.len(), 2);
        assert_relative_eq!(deriv[0], 2.0);
        assert_relative_eq!(deriv[1], -5.0);
    }

    #[test]
    fn test_sensitivity_analysis() {
        // Test sensitivity and complementary sensitivity functions
        let g = tf(vec![10.0], vec![1.0, 1.0], None).unwrap(); // 10/(s+1)

        let s_func = sensitivity(&g, None).unwrap();
        let t_func = complementary_sensitivity(&g, None).unwrap();

        // Verify S(s) + T(s) = 1 at s = 0
        let s_val = s_func.evaluate(Complex64::new(0.0, 0.0));
        let t_val = t_func.evaluate(Complex64::new(0.0, 0.0));
        let sum = s_val + t_val;

        assert_relative_eq!(sum.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_system_equivalence() {
        // Test system equivalence checking
        let sys1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let sys2 = tf(vec![2.0], vec![2.0, 2.0], None).unwrap(); // Same after normalization

        assert!(systems_equivalent(&sys1, &sys2, 1e-6).unwrap());

        // Different systems should not be equivalent
        let sys3 = tf(vec![1.0], vec![1.0, 2.0], None).unwrap();
        assert!(!systems_equivalent(&sys1, &sys3, 1e-6).unwrap());
    }

    #[test]
    fn test_frequency_response() {
        // Test frequency response computation
        let sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap(); // 1/(s+1)

        let freqs = vec![0.0, 1.0, 10.0];
        let response = sys.frequency_response(&freqs).unwrap();

        assert_eq!(response.len(), 3);

        // At s = 0: H(0) = 1/(0+1) = 1
        assert_relative_eq!(response[0].re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(response[0].im, 0.0, epsilon = 1e-6);

        // At s = j: H(j) = 1/(j+1), |H(j)| = 1/sqrt(2)
        assert_relative_eq!(response[1].norm(), 1.0 / (2.0_f64.sqrt()), epsilon = 1e-6);
    }

    #[test]
    fn test_impulse_and_step_response() {
        // Test time-domain response computation
        let _sys = tf(vec![1.0], vec![1.0, 1.0], None).unwrap(); // 1/(s+1)

        let t = vec![0.0, 0.1, 0.2, 0.5, 1.0];

        // For discrete-time system, test impulse response
        let sys_dt = tf(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();
        let impulse = sys_dt.impulse_response(&t).unwrap();
        let step = sys_dt.step_response(&t).unwrap();

        assert_eq!(impulse.len(), t.len());
        assert_eq!(step.len(), t.len());
    }
}
