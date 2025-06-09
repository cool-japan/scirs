//! Linear Time-Invariant (LTI) Systems
//!
//! This module provides types and functions for working with Linear Time-Invariant
//! systems, which are a fundamental concept in control theory and signal processing.
//!
//! Three different representations are provided:
//! - Transfer function representation: numerator and denominator polynomials
//! - Zero-pole-gain representation: zeros, poles, and gain
//! - State-space representation: A, B, C, D matrices
//!
//! These representations can be converted between each other, and used to analyze
//! system behavior through techniques such as impulse response, step response,
//! frequency response, and Bode plots.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::Zero;
use std::fmt::Debug;

/// A trait for all LTI system representations
pub trait LtiSystem {
    /// Get the transfer function representation of the system
    fn to_tf(&self) -> SignalResult<TransferFunction>;

    /// Get the zero-pole-gain representation of the system
    fn to_zpk(&self) -> SignalResult<ZerosPoleGain>;

    /// Get the state-space representation of the system
    fn to_ss(&self) -> SignalResult<StateSpace>;

    /// Calculate the system's frequency response
    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>>;

    /// Calculate the system's impulse response
    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Calculate the system's step response
    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Check if the system is stable
    fn is_stable(&self) -> SignalResult<bool>;
}

/// Transfer function representation of an LTI system
///
/// The transfer function is represented as a ratio of two polynomials:
/// H(s) = (b[0] * s^n + b[1] * s^(n-1) + ... + b[n]) / (a[0] * s^m + a[1] * s^(m-1) + ... + a[m])
///
/// Where:
/// - b: numerator coefficients (highest power first)
/// - a: denominator coefficients (highest power first)
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator coefficients (highest power first)
    pub num: Vec<f64>,

    /// Denominator coefficients (highest power first)
    pub den: Vec<f64>,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl TransferFunction {
    /// Create a new transfer function
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `TransferFunction` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::TransferFunction;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// ```
    pub fn new(mut num: Vec<f64>, mut den: Vec<f64>, dt: Option<bool>) -> SignalResult<Self> {
        // Remove leading zeros from numerator and denominator
        while num.len() > 1 && num[0].abs() < 1e-10 {
            num.remove(0);
        }

        while den.len() > 1 && den[0].abs() < 1e-10 {
            den.remove(0);
        }

        // Check if denominator is all zeros
        if den.iter().all(|&x| x.abs() < 1e-10) {
            return Err(SignalError::ValueError(
                "Denominator polynomial cannot be zero".to_string(),
            ));
        }

        // Normalize the denominator so that the leading coefficient is 1
        if !den.is_empty() && den[0].abs() > 1e-10 {
            let den_lead = den[0];
            for coef in &mut den {
                *coef /= den_lead;
            }

            // Also scale the numerator accordingly
            for coef in &mut num {
                *coef /= den_lead;
            }
        }

        Ok(TransferFunction {
            num,
            den,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get the order of the numerator polynomial
    pub fn num_order(&self) -> usize {
        self.num.len().saturating_sub(1)
    }

    /// Get the order of the denominator polynomial
    pub fn den_order(&self) -> usize {
        self.den.len().saturating_sub(1)
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Evaluate numerator polynomial
        let mut num_val = Complex64::zero();
        for (i, &coef) in self.num.iter().enumerate() {
            let power = (self.num.len() - 1 - i) as i32;
            num_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Evaluate denominator polynomial
        let mut den_val = Complex64::zero();
        for (i, &coef) in self.den.iter().enumerate() {
            let power = (self.den.len() - 1 - i) as i32;
            den_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Return the ratio
        if den_val.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num_val / den_val
        }
    }
}

impl LtiSystem for TransferFunction {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        Ok(self.clone())
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert transfer function to ZPK form by finding roots of numerator and denominator
        // This is a basic implementation - a production version would use more robust methods

        let gain = if self.num.is_empty() {
            0.0
        } else {
            self.num[0]
        };

        // Note: In practice, we would use a reliable polynomial root-finding algorithm
        // For now, returning placeholder with empty zeros and poles
        Ok(ZerosPoleGain {
            zeros: Vec::new(), // Replace with actual roots of numerator
            poles: Vec::new(), // Replace with actual roots of denominator
            gain,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert transfer function to state-space form
        // For a SISO system, this involves creating a controllable canonical form

        // This is a placeholder implementation - a full implementation would
        // properly handle the controllable canonical form construction

        // For now, return an empty state-space system
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        // For continuous-time systems, we use numerical simulation by
        // converting to state-space form and then simulating the response.
        if !self.dt {
            // Convert to state-space form if it's not already available
            let ss = self.to_ss()?;

            // Get time step (assume uniform sampling)
            let dt = if t.len() > 1 { t[1] - t[0] } else { 0.001 };

            // Simulate impulse response
            let mut response = vec![0.0; t.len()];

            if !ss.b.is_empty() && !ss.c.is_empty() {
                // Initial state is zero
                let mut x = vec![0.0; ss.n_states];

                // For an impulse, the input at t[0] is 1/dt, and 0 otherwise
                // Inject impulse: u[0] = 1/dt, which approximates a continuous impulse
                for (j, _) in (0..ss.n_inputs).enumerate() {
                    for (i, x_i) in x.iter_mut().enumerate().take(ss.n_states) {
                        *x_i += ss.b[i * ss.n_inputs + j] * (1.0 / dt);
                    }
                }

                // Record initial output
                for i in 0..ss.n_outputs {
                    let mut y = 0.0;
                    for (j, &x_j) in x.iter().enumerate().take(ss.n_states) {
                        y += ss.c[i * ss.n_states + j] * x_j;
                    }
                    if i == 0 {
                        // For SISO systems
                        response[0] = y;
                    }
                }

                // Simulate the system response for the rest of the time points
                for (_k, response_k) in response.iter_mut().enumerate().skip(1).take(t.len() - 1) {
                    // Update state: dx/dt = Ax + Bu, use forward Euler for simplicity
                    let mut x_new = vec![0.0; ss.n_states];

                    for (i, x_new_val) in x_new.iter_mut().enumerate().take(ss.n_states) {
                        for (j, &x_val) in x.iter().enumerate().take(ss.n_states) {
                            *x_new_val += ss.a[i * ss.n_states + j] * x_val * dt;
                        }
                        // No input term (Bu) after initial impulse
                    }

                    // Copy updated state
                    x = x_new;

                    // Calculate output: y = Cx + Du (u is zero after initial impulse)
                    for i in 0..ss.n_outputs {
                        let mut y = 0.0;
                        for (j, &x_j) in x.iter().enumerate().take(ss.n_states) {
                            y += ss.c[i * ss.n_states + j] * x_j;
                        }
                        if i == 0 {
                            // For SISO systems
                            *response_k = y;
                        }
                    }
                }
            }

            Ok(response)
        } else {
            // For discrete-time systems, impulse response h[n] is equivalent to
            // the inverse Z-transform of the transfer function H(z)
            // For a DT system H(z) = B(z)/A(z), the impulse response is given by
            // the coefficients of the series expansion of H(z)

            let n = t.len();
            let mut response = vec![0.0; n];

            // Check if we have the right number of coefficients
            if self.num.is_empty() || self.den.is_empty() {
                return Ok(response);
            }

            // For a proper transfer function with normalized denominator,
            // the first impulse response value is b[0]/a[0]
            response[0] = if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                self.num[0] / self.den[0]
            } else {
                self.num[0]
            };

            // For later samples, we use the recurrence relation:
            // h[n] = (b[n] - sum_{k=1}^n a[k]*h[n-k])/a[0]
            for n in 1..response.len() {
                // Add numerator contribution
                if n < self.num.len() {
                    response[n] = self.num[n];
                }

                // Subtract denominator * past outputs
                for k in 1..std::cmp::min(n + 1, self.den.len()) {
                    response[n] -= self.den[k] * response[n - k];
                }

                // Normalize by a[0]
                if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                    response[n] /= self.den[0];
                }
            }

            Ok(response)
        }
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        if !self.dt {
            // For continuous-time systems:
            // 1. Get impulse response
            let impulse = self.impulse_response(t)?;

            // 2. Integrate the impulse response to get the step response
            // Using the trapezoidal rule for integration
            let mut step = vec![0.0; t.len()];

            if t.len() > 1 {
                let dt = t[1] - t[0];

                // Initialize with the first value
                step[0] = impulse[0] * dt / 2.0;

                // Accumulate the integral
                for i in 1..t.len() {
                    step[i] = step[i - 1] + (impulse[i - 1] + impulse[i]) * dt / 2.0;
                }
            }

            Ok(step)
        } else {
            // For discrete-time systems:
            // The step response can be calculated either by:
            // 1. Convolving the impulse response with a step input
            // 2. Directly simulating with a step input
            // We'll use approach 1 for simplicity

            let impulse = self.impulse_response(t)?;
            let mut step = vec![0.0; t.len()];

            // Convolve with a unit step (running sum of impulse response)
            for (i, step_val) in step.iter_mut().enumerate().take(t.len()) {
                for &impulse_val in impulse.iter().take(i + 1) {
                    *step_val += impulse_val;
                }
            }

            Ok(step)
        }
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would check the poles from to_zpk()
        Ok(true)
    }
}

/// Zeros-poles-gain representation of an LTI system
///
/// The transfer function is represented as:
/// H(s) = gain * (s - zeros[0]) * (s - zeros[1]) * ... / ((s - poles[0]) * (s - poles[1]) * ...)
#[derive(Debug, Clone)]
pub struct ZerosPoleGain {
    /// Zeros of the transfer function
    pub zeros: Vec<Complex64>,

    /// Poles of the transfer function
    pub poles: Vec<Complex64>,

    /// System gain
    pub gain: f64,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl ZerosPoleGain {
    /// Create a new zeros-poles-gain representation
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
    /// * A new `ZerosPoleGain` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::ZerosPoleGain;
    /// use num_complex::Complex64;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let zpk = ZerosPoleGain::new(
    ///     Vec::new(),  // No zeros
    ///     vec![Complex64::new(-1.0, 0.0)],  // One pole at s = -1
    ///     1.0,  // Gain = 1
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
        gain: f64,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        Ok(ZerosPoleGain {
            zeros,
            poles,
            gain,
            dt: dt.unwrap_or(false),
        })
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Compute the numerator product (s - zeros[i])
        let mut num = Complex64::new(self.gain, 0.0);
        for &zero in &self.zeros {
            num *= s - zero;
        }

        // Compute the denominator product (s - poles[i])
        let mut den = Complex64::new(1.0, 0.0);
        for &pole in &self.poles {
            den *= s - pole;
        }

        // Return the ratio
        if den.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num / den
        }
    }
}

impl LtiSystem for ZerosPoleGain {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert ZPK to transfer function by expanding the polynomial products
        // This is a basic implementation - a production version would use more robust methods

        // For now, return a placeholder
        // In practice, we would expand (s - zero_1) * (s - zero_2) * ... for the numerator
        // and (s - pole_1) * (s - pole_2) * ... for the denominator

        Ok(TransferFunction {
            num: vec![self.gain],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        Ok(self.clone())
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert ZPK to state-space
        // Typically done by first converting to transfer function, then to state-space

        // For now, return a placeholder
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        for &pole in &self.poles {
            if self.dt {
                // For discrete-time systems, check if poles are inside the unit circle
                if pole.norm() >= 1.0 {
                    return Ok(false);
                }
            } else {
                // For continuous-time systems, check if poles have negative real parts
                if pole.re >= 0.0 {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// State-space representation of an LTI system
///
/// The system is represented as:
/// dx/dt = A*x + B*u  (for continuous-time systems)
/// x[k+1] = A*x[k] + B*u[k]  (for discrete-time systems)
/// y = C*x + D*u
///
/// Where:
/// - x is the state vector
/// - u is the input vector
/// - y is the output vector
/// - A, B, C, D are matrices of appropriate dimensions
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State matrix (n_states x n_states)
    pub a: Vec<f64>,

    /// Input matrix (n_states x n_inputs)
    pub b: Vec<f64>,

    /// Output matrix (n_outputs x n_states)
    pub c: Vec<f64>,

    /// Feedthrough matrix (n_outputs x n_inputs)
    pub d: Vec<f64>,

    /// Number of state variables
    pub n_states: usize,

    /// Number of inputs
    pub n_inputs: usize,

    /// Number of outputs
    pub n_outputs: usize,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl StateSpace {
    /// Create a new state-space system
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `StateSpace` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::StateSpace;
    ///
    /// // Create a simple first-order system: dx/dt = -x + u, y = x
    /// let ss = StateSpace::new(
    ///     vec![-1.0],  // A = [-1]
    ///     vec![1.0],   // B = [1]
    ///     vec![1.0],   // C = [1]
    ///     vec![0.0],   // D = [0]
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        d: Vec<f64>,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        // Determine the system dimensions from the matrix shapes
        let n_states = (a.len() as f64).sqrt() as usize;

        // Check if A is square
        if n_states * n_states != a.len() {
            return Err(SignalError::ValueError(
                "A matrix must be square".to_string(),
            ));
        }

        // Infer n_inputs from B
        let n_inputs = if n_states == 0 { 0 } else { b.len() / n_states };

        // Check consistency of B
        if n_states * n_inputs != b.len() {
            return Err(SignalError::ValueError(
                "B matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Infer n_outputs from C
        let n_outputs = if n_states == 0 { 0 } else { c.len() / n_states };

        // Check consistency of C
        if n_outputs * n_states != c.len() {
            return Err(SignalError::ValueError(
                "C matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Check consistency of D
        if n_outputs * n_inputs != d.len() {
            return Err(SignalError::ValueError(
                "D matrix has inconsistent dimensions".to_string(),
            ));
        }

        Ok(StateSpace {
            a,
            b,
            c,
            d,
            n_states,
            n_inputs,
            n_outputs,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get an element of the A matrix
    pub fn a(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for A matrix".to_string(),
            ));
        }

        Ok(self.a[i * self.n_states + j])
    }

    /// Get an element of the B matrix
    pub fn b(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for B matrix".to_string(),
            ));
        }

        Ok(self.b[i * self.n_inputs + j])
    }

    /// Get an element of the C matrix
    pub fn c(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for C matrix".to_string(),
            ));
        }

        Ok(self.c[i * self.n_states + j])
    }

    /// Get an element of the D matrix
    pub fn d(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for D matrix".to_string(),
            ));
        }

        Ok(self.d[i * self.n_inputs + j])
    }
}

impl LtiSystem for StateSpace {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert state-space to transfer function
        // For SISO systems, TF(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse and polynomial expansion

        Ok(TransferFunction {
            num: vec![1.0],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert state-space to ZPK
        // Typically done by first converting to transfer function, then factoring

        // For now, return a placeholder
        Ok(ZerosPoleGain {
            zeros: Vec::new(),
            poles: Vec::new(),
            gain: 1.0,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        Ok(self.clone())
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        // Calculate the frequency response for state-space system
        // H(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse for each frequency

        let response = vec![Complex64::new(1.0, 0.0); w.len()];
        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A state-space system is stable if all eigenvalues of A have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would calculate the eigenvalues of A

        Ok(true)
    }
}

/// Calculate the Bode plot data (magnitude and phase) for an LTI system
///
/// # Arguments
///
/// * `system` - The LTI system to analyze
/// * `w` - The frequency points at which to evaluate the response
///
/// # Returns
///
/// * A tuple containing (frequencies, magnitude in dB, phase in degrees)
pub fn bode<T: LtiSystem>(
    system: &T,
    w: Option<&[f64]>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Default frequencies if none provided
    let frequencies = match w {
        Some(freq) => freq.to_vec(),
        None => {
            // Generate logarithmically spaced frequencies between 0.01 and 100 rad/s
            let n = 100;
            let mut w_out = Vec::with_capacity(n);

            let w_min = 0.01;
            let w_max = 100.0;
            let log_step = f64::powf(w_max / w_min, 1.0 / (n - 1) as f64);

            let mut w_val = w_min;
            for _ in 0..n {
                w_out.push(w_val);
                w_val *= log_step;
            }

            w_out
        }
    };

    // Calculate frequency response
    let resp = system.frequency_response(&frequencies)?;

    // Convert to magnitude (dB) and phase (degrees)
    let mut mag = Vec::with_capacity(resp.len());
    let mut phase = Vec::with_capacity(resp.len());

    for &val in &resp {
        // Magnitude in dB: 20 * log10(|H(jw)|)
        let mag_db = 20.0 * val.norm().log10();
        mag.push(mag_db);

        // Phase in degrees: arg(H(jw)) * 180/pi
        let phase_deg = val.arg() * 180.0 / std::f64::consts::PI;
        phase.push(phase_deg);
    }

    Ok((frequencies, mag, phase))
}

/// Functions for creating and manipulating LTI systems
pub mod system {
    use super::*;

    /// Create a transfer function system from numerator and denominator coefficients
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `TransferFunction` instance
    pub fn tf(num: Vec<f64>, den: Vec<f64>, dt: Option<bool>) -> SignalResult<TransferFunction> {
        TransferFunction::new(num, den, dt)
    }

    /// Create a zeros-poles-gain system
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
    /// * A `ZerosPoleGain` instance
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
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `StateSpace` instance
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
    /// # Arguments
    ///
    /// * `system` - A continuous-time LTI system
    /// * `dt` - The sampling period
    ///
    /// # Returns
    ///
    /// * A discretized version of the system
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
    ///
    /// # Arguments
    ///
    /// * `g1` - First system (input side)
    /// * `g2` - Second system (output side)
    ///
    /// # Returns
    ///
    /// * The series interconnection as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
    ///
    /// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
    /// let series_sys = series(&g1, &g2).unwrap();
    /// ```
    pub fn series<T1: LtiSystem, T2: LtiSystem>(
        g1: &T1,
        g2: &T2,
    ) -> SignalResult<TransferFunction> {
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
    ///
    /// # Arguments
    ///
    /// * `g1` - First system
    /// * `g2` - Second system
    ///
    /// # Returns
    ///
    /// * The parallel interconnection as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
    ///
    /// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
    /// let parallel_sys = parallel(&g1, &g2).unwrap();
    /// ```
    pub fn parallel<T1: LtiSystem, T2: LtiSystem>(
        g1: &T1,
        g2: &T2,
    ) -> SignalResult<TransferFunction> {
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
    /// * The closed-loop system as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
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
    /// # Arguments
    ///
    /// * `g` - Forward path system
    /// * `h` - Feedback path system (optional, defaults to unity feedback)
    ///
    /// # Returns
    ///
    /// * The sensitivity function as a transfer function
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
    /// # Arguments
    ///
    /// * `g` - Forward path system
    /// * `h` - Feedback path system (optional, defaults to unity feedback)
    ///
    /// # Returns
    ///
    /// * The complementary sensitivity function as a transfer function
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
}

/// Helper functions for polynomial operations
fn multiply_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
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

fn add_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
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

fn subtract_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tf_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        assert_eq!(tf.num.len(), 1);
        assert_eq!(tf.den.len(), 2);
        assert_relative_eq!(tf.num[0], 1.0);
        assert_relative_eq!(tf.den[0], 1.0);
        assert_relative_eq!(tf.den[1], 1.0);
        assert!(!tf.dt);

        // Test normalization
        let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();
        assert_relative_eq!(tf2.num[0], 1.0);
        assert_relative_eq!(tf2.den[0], 1.0);
        assert_relative_eq!(tf2.den[1], 1.0);
    }

    #[test]
    fn test_tf_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Evaluate at s = 0
        let result = tf.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = tf.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_zpk_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        assert_eq!(zpk.zeros.len(), 0);
        assert_eq!(zpk.poles.len(), 1);
        assert_relative_eq!(zpk.poles[0].re, -1.0);
        assert_relative_eq!(zpk.poles[0].im, 0.0);
        assert_relative_eq!(zpk.gain, 1.0);
        assert!(!zpk.dt);
    }

    #[test]
    fn test_zpk_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        // Evaluate at s = 0
        let result = zpk.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = zpk.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_ss_creation() {
        // Create a simple first-order system: dx/dt = -x + u, y = x
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        assert_eq!(ss.n_states, 1);
        assert_eq!(ss.n_inputs, 1);
        assert_eq!(ss.n_outputs, 1);
        assert_relative_eq!(ss.a[0], -1.0);
        assert_relative_eq!(ss.b[0], 1.0);
        assert_relative_eq!(ss.c[0], 1.0);
        assert_relative_eq!(ss.d[0], 0.0);
        assert!(!ss.dt);
    }

    #[test]
    fn test_bode() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Compute Bode plot at omega = 0.1, 1, 10
        let freqs = vec![0.1, 1.0, 10.0];
        let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();

        // Check frequencies
        assert_eq!(w.len(), 3);
        assert_relative_eq!(w[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(w[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(w[2], 10.0, epsilon = 1e-6);

        // Check magnitudes (in dB)
        assert_eq!(mag.len(), 3);
        // At omega = 0.1, |H| = 0.995, which is -0.043 dB
        assert_relative_eq!(mag[0], -0.043, epsilon = 0.01);
        // At omega = 1, |H| = 0.707, which is -3 dB
        assert_relative_eq!(mag[1], -3.0, epsilon = 0.1);
        // At omega = 10, |H| = 0.0995, which is -20.043 dB
        assert_relative_eq!(mag[2], -20.043, epsilon = 0.1);

        // Check phases (in degrees)
        assert_eq!(phase.len(), 3);
        // At omega = 0.1, phase is about -5.7 degrees
        assert_relative_eq!(phase[0], -5.7, epsilon = 0.1);
        // At omega = 1, phase is -45 degrees
        assert_relative_eq!(phase[1], -45.0, epsilon = 0.1);
        // At omega = 10, phase is about -84.3 degrees
        assert_relative_eq!(phase[2], -84.3, epsilon = 0.1);
    }

    #[test]
    fn test_is_stable() {
        // Stable continuous-time system: H(s) = 1 / (s + 1)
        let stable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();
        assert!(stable.is_stable().unwrap());

        // Unstable continuous-time system: H(s) = 1 / (s - 1)
        let unstable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None).unwrap();
        assert!(!unstable.is_stable().unwrap());

        // Stable discrete-time system: H(z) = 1 / (z - 0.5)
        let stable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(0.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(stable_dt.is_stable().unwrap());

        // Unstable discrete-time system: H(z) = 1 / (z - 1.5)
        let unstable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(!unstable_dt.is_stable().unwrap());
    }

    #[test]
    fn test_series_connection() {
        // Test series connection of two first-order systems
        // G1(s) = 1/(s+1), G2(s) = 2/(s+2)
        let g1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = TransferFunction::new(vec![2.0], vec![1.0, 2.0], None).unwrap();

        let series_sys = system::series(&g1, &g2).unwrap();

        // Series: H(s) = G2(s)*G1(s) = 2/((s+1)(s+2)) = 2/(s^2+3s+2)
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
        let g1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = TransferFunction::new(vec![1.0], vec![1.0, 2.0], None).unwrap();

        let parallel_sys = system::parallel(&g1, &g2).unwrap();

        // Parallel: H(s) = G1(s)+G2(s) = 1/(s+1) + 1/(s+2) = (s+2+s+1)/((s+1)(s+2))
        //         = (2s+3)/(s^2+3s+2)
        assert_eq!(parallel_sys.num.len(), 2);
        assert_eq!(parallel_sys.den.len(), 3);
        assert_relative_eq!(parallel_sys.num[0], 2.0);
        assert_relative_eq!(parallel_sys.num[1], 3.0);
        assert_relative_eq!(parallel_sys.den[0], 1.0);
        assert_relative_eq!(parallel_sys.den[1], 3.0);
        assert_relative_eq!(parallel_sys.den[2], 2.0);
    }

    #[test]
    fn test_feedback_connection() {
        // Test feedback connection with unity feedback
        // G(s) = 1/(s+1), unity feedback
        let g = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        let feedback_sys = system::feedback(&g, None, 1).unwrap();

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
        let g = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let h = TransferFunction::new(vec![2.0], vec![1.0], None).unwrap();

        let feedback_sys = system::feedback(&g, Some(&h as &dyn LtiSystem), 1).unwrap();

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
        let g = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let sens = system::sensitivity(&g, None).unwrap();

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
        let g = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let comp_sens = system::complementary_sensitivity(&g, None).unwrap();

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
    fn test_system_interconnection_errors() {
        // Test error when connecting continuous and discrete-time systems
        let g_ct = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Some(false)).unwrap();
        let g_dt = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();

        let result = system::series(&g_ct, &g_dt);
        assert!(result.is_err());

        let result = system::parallel(&g_ct, &g_dt);
        assert!(result.is_err());

        let result = system::feedback(&g_ct, Some(&g_dt as &dyn LtiSystem), 1);
        assert!(result.is_err());
    }
}
