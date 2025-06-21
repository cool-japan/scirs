//! Core LTI system representations and implementations
//!
//! This module provides the fundamental types for representing Linear Time-Invariant (LTI) systems:
//! - Transfer function representation (numerator/denominator polynomials)
//! - Zero-pole-gain representation (zeros, poles, and gain)
//! - State-space representation (A, B, C, D matrices)
//!
//! All system representations implement the `LtiSystem` trait for common operations.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::Zero;

/// A trait for all LTI system representations
///
/// This trait provides a common interface for different LTI system representations,
/// allowing conversions between forms and standard system analysis operations.
pub trait LtiSystem {
    /// Get the transfer function representation of the system
    fn to_tf(&self) -> SignalResult<TransferFunction>;

    /// Get the zero-pole-gain representation of the system
    fn to_zpk(&self) -> SignalResult<ZerosPoleGain>;

    /// Get the state-space representation of the system
    fn to_ss(&self) -> SignalResult<StateSpace>;

    /// Calculate the system's frequency response at given frequencies
    ///
    /// # Arguments
    ///
    /// * `w` - Array of frequencies at which to evaluate the response
    ///
    /// # Returns
    ///
    /// Complex frequency response values H(jω) for each frequency
    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>>;

    /// Calculate the system's impulse response at given time points
    ///
    /// # Arguments
    ///
    /// * `t` - Array of time points at which to evaluate the response
    ///
    /// # Returns
    ///
    /// Impulse response values h(t) for each time point
    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Calculate the system's step response at given time points
    ///
    /// # Arguments
    ///
    /// * `t` - Array of time points at which to evaluate the response
    ///
    /// # Returns
    ///
    /// Step response values s(t) for each time point
    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Check if the system is stable
    ///
    /// For continuous-time systems: all poles must have negative real parts
    /// For discrete-time systems: all poles must be inside the unit circle
    ///
    /// # Returns
    ///
    /// True if the system is stable, false otherwise
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
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::systems::TransferFunction;
///
/// // Create H(s) = (s + 2) / (s^2 + 3s + 2)
/// let tf = TransferFunction::new(
///     vec![1.0, 2.0],      // s + 2
///     vec![1.0, 3.0, 2.0], // s^2 + 3s + 2
///     None
/// ).unwrap();
/// ```
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
    /// A new `TransferFunction` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_signal::lti::systems::TransferFunction;
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
    ///
    /// # Arguments
    ///
    /// * `s` - Complex frequency at which to evaluate H(s)
    ///
    /// # Returns
    ///
    /// The complex value H(s)
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
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::systems::ZerosPoleGain;
/// use num_complex::Complex64;
///
/// // Create H(s) = 2 * (s + 1) / (s + 2)
/// let zpk = ZerosPoleGain::new(
///     vec![Complex64::new(-1.0, 0.0)], // zero at s = -1
///     vec![Complex64::new(-2.0, 0.0)], // pole at s = -2
///     2.0,                             // gain = 2
///     None
/// ).unwrap();
/// ```
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
    /// A new `ZerosPoleGain` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_signal::lti::systems::ZerosPoleGain;
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
    ///
    /// # Arguments
    ///
    /// * `s` - Complex frequency at which to evaluate H(s)
    ///
    /// # Returns
    ///
    /// The complex value H(s)
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
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::systems::StateSpace;
///
/// // Create a simple integrator: dx/dt = u, y = x
/// let ss = StateSpace::new(
///     vec![0.0],  // A = [0]
///     vec![1.0],  // B = [1]
///     vec![1.0],  // C = [1]
///     vec![0.0],  // D = [0]
///     None
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State matrix (n_states x n_states), stored in row-major order
    pub a: Vec<f64>,

    /// Input matrix (n_states x n_inputs), stored in row-major order
    pub b: Vec<f64>,

    /// Output matrix (n_outputs x n_states), stored in row-major order
    pub c: Vec<f64>,

    /// Feedthrough matrix (n_outputs x n_inputs), stored in row-major order
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
    /// * `a` - State matrix (n_states x n_states), stored in row-major order
    /// * `b` - Input matrix (n_states x n_inputs), stored in row-major order
    /// * `c` - Output matrix (n_outputs x n_states), stored in row-major order
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs), stored in row-major order
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// A new `StateSpace` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_signal::lti::systems::StateSpace;
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
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    ///
    /// The value A[i,j]
    pub fn a(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for A matrix".to_string(),
            ));
        }

        Ok(self.a[i * self.n_states + j])
    }

    /// Get an element of the B matrix
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    ///
    /// The value B[i,j]
    pub fn b(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for B matrix".to_string(),
            ));
        }

        Ok(self.b[i * self.n_inputs + j])
    }

    /// Get an element of the C matrix
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    ///
    /// The value C[i,j]
    pub fn c(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for C matrix".to_string(),
            ));
        }

        Ok(self.c[i * self.n_states + j])
    }

    /// Get an element of the D matrix
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    ///
    /// The value D[i,j]
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transfer_function_creation() {
        // Test creating a simple transfer function
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        assert_eq!(tf.num.len(), 1);
        assert_eq!(tf.den.len(), 2);
        assert_eq!(tf.num[0], 1.0);
        assert_eq!(tf.den[0], 1.0);
        assert_eq!(tf.den[1], 1.0);
        assert!(!tf.dt);
    }

    #[test]
    fn test_transfer_function_normalization() {
        // Test that denominator is normalized
        let tf = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();
        assert_relative_eq!(tf.num[0], 1.0);
        assert_relative_eq!(tf.den[0], 1.0);
        assert_relative_eq!(tf.den[1], 1.0);
    }

    #[test]
    fn test_transfer_function_evaluation() {
        // Test evaluating H(s) = 1/(s+1) at s = 0
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let result = tf.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0);
        assert_relative_eq!(result.im, 0.0);
    }

    #[test]
    fn test_zpk_creation() {
        let zpk = ZerosPoleGain::new(
            vec![Complex64::new(-1.0, 0.0)],
            vec![Complex64::new(-2.0, 0.0)],
            1.0,
            None,
        )
        .unwrap();

        assert_eq!(zpk.zeros.len(), 1);
        assert_eq!(zpk.poles.len(), 1);
        assert_eq!(zpk.gain, 1.0);
        assert!(!zpk.dt);
    }

    #[test]
    fn test_zpk_stability() {
        // Stable system (all poles in LHP)
        let zpk_stable = ZerosPoleGain::new(
            Vec::new(),
            vec![Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0)],
            1.0,
            None,
        )
        .unwrap();
        assert!(zpk_stable.is_stable().unwrap());

        // Unstable system (pole in RHP)
        let zpk_unstable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None).unwrap();
        assert!(!zpk_unstable.is_stable().unwrap());
    }

    #[test]
    fn test_state_space_creation() {
        let ss = StateSpace::new(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None).unwrap();

        assert_eq!(ss.n_states, 1);
        assert_eq!(ss.n_inputs, 1);
        assert_eq!(ss.n_outputs, 1);
        assert!(!ss.dt);
    }

    #[test]
    fn test_state_space_matrix_access() {
        let ss = StateSpace::new(
            vec![-1.0, 0.0, 1.0, -2.0], // 2x2 A matrix
            vec![1.0, 0.0],             // 2x1 B matrix
            vec![1.0, 0.0],             // 1x2 C matrix
            vec![0.0],                  // 1x1 D matrix
            None,
        )
        .unwrap();

        assert_eq!(ss.a(0, 0).unwrap(), -1.0);
        assert_eq!(ss.a(0, 1).unwrap(), 0.0);
        assert_eq!(ss.a(1, 0).unwrap(), 1.0);
        assert_eq!(ss.a(1, 1).unwrap(), -2.0);

        assert_eq!(ss.b(0, 0).unwrap(), 1.0);
        assert_eq!(ss.b(1, 0).unwrap(), 0.0);

        assert_eq!(ss.c(0, 0).unwrap(), 1.0);
        assert_eq!(ss.c(0, 1).unwrap(), 0.0);

        assert_eq!(ss.d(0, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_invalid_denominator() {
        let result = TransferFunction::new(vec![1.0], vec![0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inconsistent_state_space_dimensions() {
        // Invalid A matrix (not square)
        let result = StateSpace::new(
            vec![1.0, 2.0, 3.0], // 3 elements, not a perfect square
            vec![1.0],
            vec![1.0],
            vec![0.0],
            None,
        );
        assert!(result.is_err());
    }
}
