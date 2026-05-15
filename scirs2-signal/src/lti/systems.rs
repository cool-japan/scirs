// Core LTI system representations and implementations
//
// This module provides the fundamental types for representing Linear Time-Invariant (LTI) systems:
// - Transfer function representation (numerator/denominator polynomials)
// - Zero-pole-gain representation (zeros, poles, and gain)
// - State-space representation (A, B, C, D matrices)
//
// All system representations implement the `LtiSystem` trait for common operations.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;

#[allow(unused_imports)]
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
/// `H(s) = (b[0] * s^n + b[1] * s^(n-1) + ... + b[n]) / (a[0] * s^m + a[1] * s^(m-1) + ... + a[m])`
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
/// ).expect("Operation failed");
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
    /// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("Operation failed");
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
        if den.iter().all(|&x: &f64| x.abs() < 1e-10) {
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
        // using companion matrix eigenvalue method

        let gain = if self.num.is_empty() {
            0.0
        } else {
            self.num[0]
        };

        // Find zeros (roots of numerator)
        let zeros = if self.num.len() > 1 {
            poly_roots_real(&self.num)?
        } else {
            Vec::new()
        };

        // Find poles (roots of denominator)
        let poles = if self.den.len() > 1 {
            poly_roots_real(&self.den)?
        } else {
            Vec::new()
        };

        Ok(ZerosPoleGain {
            zeros,
            poles,
            gain,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert transfer function to state-space form using controllable canonical form
        // For a transfer function H(s) = N(s)/D(s) = (b[0]*s^n + ... + b[n])/(s^n + a[1]*s^(n-1) + ... + a[n])

        if self.den.is_empty() {
            return Err(SignalError::ValueError(
                "Denominator cannot be empty".to_string(),
            ));
        }

        // Get the order of the system (highest power of denominator)
        let n = self.den.len() - 1; // degree of denominator

        if n == 0 {
            // Zero-order system: just a constant gain
            let d_val = if !self.num.is_empty() {
                self.num[0] / self.den[0]
            } else {
                0.0
            };
            return Ok(StateSpace {
                a: Vec::new(),
                b: Vec::new(),
                c: Vec::new(),
                d: vec![d_val],
                n_inputs: 1,
                n_outputs: 1,
                n_states: 0,
                dt: self.dt,
            });
        }

        // Normalize denominator (ensure leading coefficient is 1)
        let mut den_norm = self.den.clone();
        let leading_coeff = den_norm[0];
        for coeff in &mut den_norm {
            *coeff /= leading_coeff;
        }

        let mut num_norm = self.num.clone();
        for coeff in &mut num_norm {
            *coeff /= leading_coeff;
        }

        // Pad numerator with zeros if necessary
        while num_norm.len() < den_norm.len() {
            num_norm.insert(0, 0.0);
        }

        // Controllable canonical form
        // A matrix (companion form)
        let mut a = vec![0.0; n * n];

        // Fill A matrix
        for i in 0..n {
            if i < n - 1 {
                // Super-diagonal of 1s
                a[i * n + (i + 1)] = 1.0;
            }
            // Bottom row contains -a_i coefficients
            a[(n - 1) * n + i] = -den_norm[n - i];
        }

        // B matrix (all zeros except last element = 1)
        let mut b = vec![0.0; n];
        if n > 0 {
            b[n - 1] = 1.0;
        }

        // C matrix contains numerator coefficients (after removing D term)
        let mut c = vec![0.0; n];
        for i in 0..n.min(num_norm.len()) {
            if i + 1 < num_norm.len() {
                c[i] = num_norm[i + 1];
            }
        }

        // D matrix (direct feedthrough)
        let d = if num_norm.len() > n {
            vec![num_norm[0]]
        } else {
            vec![0.0]
        };

        Ok(StateSpace {
            a,
            b,
            c,
            d,
            n_inputs: 1,
            n_outputs: 1,
            n_states: n,
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

        // For continuous-time systems, create an impulse input and use lsim
        if !self.dt {
            // Create impulse input: very short, high amplitude pulse
            let dt = if t.len() > 1 { t[1] - t[0] } else { 0.001 };
            let impulse_amplitude = 1.0 / dt;
            let mut u = vec![0.0; t.len()];
            if !u.is_empty() {
                u[0] = impulse_amplitude;
            }

            // Use the improved lsim function with RK4 integration
            crate::lti::response::lsim(self, &u, t)
        } else {
            // For discrete-time systems, use difference equation
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

            // For later samples, use the recurrence relation:
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

        // For both continuous and discrete-time systems, create a step input and use lsim
        let u = vec![1.0; t.len()]; // Unit step input

        // Use the improved lsim function
        crate::lti::response::lsim(self, &u, t)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time).
        // Obtain poles via ZPK conversion.
        let zpk = self.to_zpk()?;
        for pole in &zpk.poles {
            if self.dt {
                // Discrete-time: stable iff |pole| < 1
                if pole.norm() >= 1.0 {
                    return Ok(false);
                }
            } else {
                // Continuous-time: stable iff Re(pole) < 0
                if pole.re >= 0.0 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// Zeros-poles-gain representation of an LTI system
///
/// The transfer function is represented as:
/// `H(s) = gain * (s - zeros[0]) * (s - zeros[1]) * ... / ((s - poles[0]) * (s - poles[1]) * ...)`
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::systems::ZerosPoleGain;
/// use scirs2_core::numeric::Complex64;
///
/// // Create H(s) = 2 * (s + 1) / (s + 2)
/// let zpk = ZerosPoleGain::new(
///     vec![Complex64::new(-1.0, 0.0)], // zero at s = -1
///     vec![Complex64::new(-2.0, 0.0)], // pole at s = -2
///     2.0,                             // gain = 2
///     None
/// ).expect("Operation failed");
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
    /// use scirs2_core::numeric::Complex64;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let zpk = ZerosPoleGain::new(
    ///     Vec::new(),  // No zeros
    ///     vec![Complex64::new(-1.0, 0.0)],  // One pole at s = -1
    ///     1.0,  // Gain = 1
    ///     None,
    /// ).expect("Operation failed");
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
        // Expand (s - zero_1) * (s - zero_2) * ... for the numerator
        // and (s - pole_1) * (s - pole_2) * ... for the denominator

        let num_complex = poly_from_roots(&self.zeros);
        let den_complex = poly_from_roots(&self.poles);

        // Extract real parts (for real-coefficient systems, imaginary parts should be ~0)
        let mut num: Vec<f64> = num_complex.iter().map(|c| c.re).collect();
        let den: Vec<f64> = den_complex.iter().map(|c| c.re).collect();

        // Apply gain to numerator
        for coeff in &mut num {
            *coeff *= self.gain;
        }

        TransferFunction::new(num, den, Some(self.dt))
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        Ok(self.clone())
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert ZPK to state-space via transfer function
        let tf = self.to_tf()?;
        tf.to_ss()
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
        // Delegate to transfer function representation
        let tf = self.to_tf()?;
        tf.impulse_response(t)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Delegate to transfer function representation
        let tf = self.to_tf()?;
        tf.step_response(t)
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
/// `x[k+1] = A*x[k] + B*u[k]`  (for discrete-time systems)
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
/// ).expect("Operation failed");
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
    /// ).expect("Operation failed");
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
    /// The value `A[i,j]`
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
    /// The value `B[i,j]`
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
    /// The value `C[i,j]`
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
    /// The value `D[i,j]`
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
        // Convert state-space to transfer function using Faddeev-LeVerrier algorithm
        // For SISO systems, TF(s) = C * (sI - A)^-1 * B + D

        if self.n_states == 0 {
            // Static gain system
            let d_val = if !self.d.is_empty() { self.d[0] } else { 0.0 };
            return TransferFunction::new(vec![d_val], vec![1.0], Some(self.dt));
        }

        let n = self.n_states;

        // Compute characteristic polynomial of A using Faddeev-LeVerrier algorithm
        // det(sI - A) = s^n + c_{n-1}*s^{n-1} + ... + c_0
        let char_poly = characteristic_polynomial(&self.a, n);

        // Compute numerator: C * adj(sI - A) * B + D * det(sI - A)
        let num_poly = compute_ss_numerator(&self.a, &self.b, &self.c, &self.d, n, &char_poly);

        TransferFunction::new(num_poly, char_poly, Some(self.dt))
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert state-space to ZPK via transfer function
        let tf = self.to_tf()?;
        tf.to_zpk()
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        Ok(self.clone())
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        // Calculate H(jw) = C * (jwI - A)^-1 * B + D for each frequency
        // using complex linear system solve

        if self.n_states == 0 {
            let d_val = if !self.d.is_empty() { self.d[0] } else { 0.0 };
            return Ok(vec![Complex64::new(d_val, 0.0); w.len()]);
        }

        let n = self.n_states;
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                Complex64::new(0.0, freq).exp()
            } else {
                Complex64::new(0.0, freq)
            };

            // Build sI - A
            let mut si_minus_a = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                for j in 0..n {
                    let a_ij = self.a[i * n + j];
                    si_minus_a[i * n + j] = if i == j {
                        s - Complex64::new(a_ij, 0.0)
                    } else {
                        Complex64::new(-a_ij, 0.0)
                    };
                }
            }

            // Solve (sI - A) * x = B for x
            let b_complex: Vec<Complex64> =
                self.b.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            let x = solve_complex_system(&si_minus_a, &b_complex, n)?;

            // H(s) = C * x + D
            let mut h = Complex64::new(0.0, 0.0);
            for i in 0..n {
                h += Complex64::new(self.c[i], 0.0) * x[i];
            }
            if !self.d.is_empty() {
                h += Complex64::new(self.d[0], 0.0);
            }

            response.push(h);
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Convert to TF and use its impulse response (which uses lsim)
        let tf = self.to_tf()?;
        tf.impulse_response(t)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Convert to TF and use its step response (which uses lsim)
        let tf = self.to_tf()?;
        tf.step_response(t)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A state-space system is stable if all eigenvalues of A have negative real parts (continuous)
        // or are inside the unit circle (discrete)

        if self.n_states == 0 {
            return Ok(true);
        }

        // Compute eigenvalues via characteristic polynomial roots
        let char_poly = characteristic_polynomial(&self.a, self.n_states);
        let eigenvalues = poly_roots_real(&char_poly)?;

        for ev in &eigenvalues {
            if self.dt {
                if ev.norm() >= 1.0 {
                    return Ok(false);
                }
            } else {
                if ev.re >= 0.0 {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

// ============================================================================
// Helper functions for polynomial root finding and system conversions
// ============================================================================

/// Find roots of a polynomial using the companion matrix eigenvalue method.
/// Polynomial coefficients are in descending power order: p[0]*x^n + p[1]*x^(n-1) + ... + p[n]
fn poly_roots_real(coeffs: &[f64]) -> SignalResult<Vec<Complex64>> {
    if coeffs.is_empty() || coeffs.len() == 1 {
        return Ok(Vec::new());
    }

    // Normalize by leading coefficient
    let lead = coeffs[0];
    if lead.abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Leading coefficient is zero".to_string(),
        ));
    }

    let n = coeffs.len() - 1; // degree of polynomial
    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        // Linear: lead*x + coeffs[1] = 0 => x = -coeffs[1]/lead
        return Ok(vec![Complex64::new(-coeffs[1] / lead, 0.0)]);
    }

    if n == 2 {
        // Quadratic formula
        let a = lead;
        let b = coeffs[1];
        let c = coeffs[2];
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            return Ok(vec![
                Complex64::new((-b + sq) / (2.0 * a), 0.0),
                Complex64::new((-b - sq) / (2.0 * a), 0.0),
            ]);
        } else {
            let sq = (-disc).sqrt();
            return Ok(vec![
                Complex64::new(-b / (2.0 * a), sq / (2.0 * a)),
                Complex64::new(-b / (2.0 * a), -sq / (2.0 * a)),
            ]);
        }
    }

    // Build companion matrix for the monic polynomial
    // x^n + (c1)*x^(n-1) + ... + c_n = 0
    let mut companion = vec![0.0; n * n];
    for i in 0..n {
        // Last column contains negated coefficients
        companion[i * n + (n - 1)] = -coeffs[i + 1] / lead;
    }
    // Sub-diagonal of ones
    for i in 1..n {
        companion[i * n + (i - 1)] = 1.0;
    }

    // Compute eigenvalues using QR iteration
    qr_eigenvalues(&companion, n)
}

/// QR iteration for eigenvalues of a general matrix.
/// Uses Hessenberg reduction followed by implicit QR shifts.
fn qr_eigenvalues(matrix: &[f64], n: usize) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![Complex64::new(matrix[0], 0.0)]);
    }

    // First reduce to upper Hessenberg form
    let mut h = matrix.to_vec();
    hessenberg_reduce(&mut h, n);

    let max_iter = 200 * n;
    let mut eigenvalues = Vec::with_capacity(n);

    let mut p = n;
    let mut iter_count = 0;

    while p > 0 && iter_count < max_iter {
        if p == 1 {
            eigenvalues.push(Complex64::new(h[0], 0.0));
            break;
        }

        if p == 2 {
            let evs = eigenvalues_2x2(h[0], h[1], h[n], h[n + 1]);
            eigenvalues.push(evs.0);
            eigenvalues.push(evs.1);
            break;
        }

        // Check for convergence on subdiagonal
        let sub_idx = (p - 1) * n + (p - 2);
        let diag_sum = h[(p - 2) * n + (p - 2)].abs() + h[(p - 1) * n + (p - 1)].abs();
        let threshold = 1e-14 * diag_sum.max(1e-15);

        if h[sub_idx].abs() < threshold {
            // Deflation: eigenvalue found
            eigenvalues.push(Complex64::new(h[(p - 1) * n + (p - 1)], 0.0));
            p -= 1;
            iter_count = 0;
            continue;
        }

        // Check 2x2 block at bottom
        let sub_idx2 = if p >= 3 { (p - 2) * n + (p - 3) } else { 0 };
        if p >= 3
            && h[sub_idx2].abs()
                < 1e-14
                    * (h[(p - 3) * n + (p - 3)].abs() + h[(p - 2) * n + (p - 2)].abs()).max(1e-15)
        {
            // 2x2 block converged
            let r1 = (p - 2) * n + (p - 2);
            let r2 = (p - 2) * n + (p - 1);
            let r3 = (p - 1) * n + (p - 2);
            let r4 = (p - 1) * n + (p - 1);
            let evs = eigenvalues_2x2(h[r1], h[r2], h[r3], h[r4]);
            eigenvalues.push(evs.0);
            eigenvalues.push(evs.1);
            p -= 2;
            iter_count = 0;
            continue;
        }

        // Perform one implicit QR step with Wilkinson shift
        implicit_qr_step(&mut h, n, p);
        iter_count += 1;
    }

    // If we ran out of iterations, extract remaining diagonal elements
    if eigenvalues.len() < n {
        for i in 0..p {
            eigenvalues.push(Complex64::new(h[i * n + i], 0.0));
        }
    }

    Ok(eigenvalues)
}

/// Reduce a matrix to upper Hessenberg form using Householder reflections.
fn hessenberg_reduce(h: &mut [f64], n: usize) {
    for k in 0..n.saturating_sub(2) {
        // Compute Householder vector for column k, rows k+1..n
        let m = n - k - 1;
        if m == 0 {
            continue;
        }

        let mut v = vec![0.0; m];
        for i in 0..m {
            v[i] = h[(k + 1 + i) * n + k];
        }

        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            continue;
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm;
        let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-15 {
            continue;
        }
        for vi in &mut v {
            *vi /= v_norm;
        }

        // Apply H = I - 2*v*v^T from the left: H * (submatrix)
        for j in k..n {
            let mut dot = 0.0;
            for i in 0..m {
                dot += v[i] * h[(k + 1 + i) * n + j];
            }
            for i in 0..m {
                h[(k + 1 + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        // Apply from the right: (submatrix) * H
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..m {
                dot += h[i * n + (k + 1 + j)] * v[j];
            }
            for j in 0..m {
                h[i * n + (k + 1 + j)] -= 2.0 * dot * v[j];
            }
        }
    }
}

/// Compute eigenvalues of a 2x2 matrix [[a, b], [c, d]]
fn eigenvalues_2x2(a: f64, b: f64, c: f64, d: f64) -> (Complex64, Complex64) {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        let sq = disc.sqrt();
        (
            Complex64::new((trace + sq) / 2.0, 0.0),
            Complex64::new((trace - sq) / 2.0, 0.0),
        )
    } else {
        let sq = (-disc).sqrt();
        (
            Complex64::new(trace / 2.0, sq / 2.0),
            Complex64::new(trace / 2.0, -sq / 2.0),
        )
    }
}

/// Perform one implicit QR step with Wilkinson shift on upper Hessenberg matrix.
fn implicit_qr_step(h: &mut [f64], n: usize, p: usize) {
    // Wilkinson shift: eigenvalue of bottom-right 2x2 block closest to h[p-1,p-1]
    let a11 = h[(p - 2) * n + (p - 2)];
    let a12 = h[(p - 2) * n + (p - 1)];
    let a21 = h[(p - 1) * n + (p - 2)];
    let a22 = h[(p - 1) * n + (p - 1)];

    let trace = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = trace * trace - 4.0 * det;

    let shift = if disc >= 0.0 {
        let sq = disc.sqrt();
        let ev1 = (trace + sq) / 2.0;
        let ev2 = (trace - sq) / 2.0;
        if (ev1 - a22).abs() < (ev2 - a22).abs() {
            ev1
        } else {
            ev2
        }
    } else {
        // Complex eigenvalue pair - use real part of closest eigenvalue
        trace / 2.0
    };

    // Apply shifted QR step: QR factorize (H - shift*I), then H = R*Q + shift*I
    // Using Givens rotations for Hessenberg matrix
    let mut cos_vals = vec![0.0; p - 1];
    let mut sin_vals = vec![0.0; p - 1];

    // Shift diagonal
    for i in 0..p {
        h[i * n + i] -= shift;
    }

    // QR via Givens rotations
    for i in 0..p - 1 {
        let a_val = h[i * n + i];
        let b_val = h[(i + 1) * n + i];
        let r = (a_val * a_val + b_val * b_val).sqrt();
        if r < 1e-15 {
            cos_vals[i] = 1.0;
            sin_vals[i] = 0.0;
            continue;
        }
        let c = a_val / r;
        let s = b_val / r;
        cos_vals[i] = c;
        sin_vals[i] = s;

        // Apply rotation to rows i, i+1
        for j in i..p.min(n) {
            let t1 = h[i * n + j];
            let t2 = h[(i + 1) * n + j];
            h[i * n + j] = c * t1 + s * t2;
            h[(i + 1) * n + j] = -s * t1 + c * t2;
        }
    }

    // Multiply R * Q (apply rotations from the right)
    for i in 0..p - 1 {
        let c = cos_vals[i];
        let s = sin_vals[i];
        for j in 0..(i + 2).min(p) {
            let t1 = h[j * n + i];
            let t2 = h[j * n + (i + 1)];
            h[j * n + i] = c * t1 + s * t2;
            h[j * n + (i + 1)] = -s * t1 + c * t2;
        }
    }

    // Restore shift
    for i in 0..p {
        h[i * n + i] += shift;
    }
}

/// Expand polynomial from roots: (x - r1)(x - r2)...(x - rn) -> coefficient vector
/// Returns coefficients in descending order [1, -(r1+r2+...), ..., (-1)^n * r1*r2*...]
fn poly_from_roots(roots: &[Complex64]) -> Vec<Complex64> {
    let n = roots.len();
    if n == 0 {
        return vec![Complex64::new(1.0, 0.0)];
    }

    let mut coeffs = vec![Complex64::new(0.0, 0.0); n + 1];
    coeffs[0] = Complex64::new(1.0, 0.0);

    for (i, &root) in roots.iter().enumerate() {
        // Multiply current polynomial by (x - root)
        for j in (1..=i + 1).rev() {
            coeffs[j] = coeffs[j] - root * coeffs[j - 1];
        }
    }

    coeffs
}

/// Compute characteristic polynomial of matrix A using Faddeev-LeVerrier algorithm.
/// Returns coefficients in descending power order [1, c_{n-1}, ..., c_0].
fn characteristic_polynomial(a: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }

    let mut coeffs = vec![0.0; n + 1];
    coeffs[0] = 1.0;

    // M_k = A * M_{k-1} + c_{n-k} * I, starting with M_0 = I
    let mut m = vec![0.0; n * n];
    // Identity matrix
    for i in 0..n {
        m[i * n + i] = 1.0;
    }

    for k in 1..=n {
        // M = A * M_prev
        let m_prev = m.clone();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += a[i * n + l] * m_prev[l * n + j];
                }
                m[i * n + j] = sum;
            }
        }

        // c_{n-k} = -trace(M) / k
        let mut trace = 0.0;
        for i in 0..n {
            trace += m[i * n + i];
        }
        let c = -trace / k as f64;
        coeffs[k] = c;

        // M = M + c * I
        for i in 0..n {
            m[i * n + i] += c;
        }
    }

    coeffs
}

/// Compute numerator polynomial of SS -> TF conversion.
/// For SISO: num(s) = C * adj(sI - A) * B + D * det(sI - A)
fn compute_ss_numerator(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    n: usize,
    char_poly: &[f64],
) -> Vec<f64> {
    if n == 0 {
        return if !d.is_empty() { vec![d[0]] } else { vec![0.0] };
    }

    // Use Faddeev-LeVerrier to compute C * adj(sI - A) * B as a polynomial in s
    // adj(sI - A) = sum_{k=0}^{n-1} N_k * s^k
    // where N_{n-1} = I, N_{k} = A * N_{k+1} + c_{n-1-k} * I

    let mut adj_coeffs = vec![vec![0.0; n * n]; n]; // N_k matrices

    // N_{n-1} = I
    for i in 0..n {
        adj_coeffs[n - 1][i * n + i] = 1.0;
    }

    // Compute N_{k} = A * N_{k+1} + char_poly[n-1-k] * I
    for k_idx in (0..n - 1).rev() {
        let next = adj_coeffs[k_idx + 1].clone();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += a[i * n + l] * next[l * n + j];
                }
                adj_coeffs[k_idx][i * n + j] = sum;
                if i == j {
                    adj_coeffs[k_idx][i * n + j] += char_poly[n - 1 - k_idx];
                }
            }
        }
    }

    // Compute C * N_k * B for each k to get polynomial coefficients
    // num(s) = sum_{k=0}^{n-1} (C * N_k * B) * s^k + D * char_poly(s)
    let mut num = vec![0.0; n + 1];

    // D * char_poly contribution
    let d_val = if !d.is_empty() { d[0] } else { 0.0 };
    for (i, &cp) in char_poly.iter().enumerate() {
        num[i] += d_val * cp;
    }

    // C * N_k * B contribution: coefficient of s^k goes to position n-k
    for k in 0..n {
        let nk = &adj_coeffs[k];
        let mut cb = 0.0;
        for i in 0..n {
            let mut nb_i = 0.0;
            for j in 0..n {
                nb_i += nk[i * n + j] * b[j];
            }
            cb += c[i] * nb_i;
        }
        // s^k corresponds to index (n - k) in descending order
        num[n - k] += cb;
    }

    num
}

/// Solve a complex linear system A*x = b using Gaussian elimination with partial pivoting.
fn solve_complex_system(
    a: &[Complex64],
    b: &[Complex64],
    n: usize,
) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augmented matrix [A|b]
    let mut aug = vec![Complex64::new(0.0, 0.0); n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = col;
        for row in col..n {
            let val = aug[row * (n + 1) + col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(SignalError::ComputationError(
                "Singular matrix in complex system solve".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        if diag.norm() < 1e-15 {
            return Err(SignalError::ComputationError(
                "Zero diagonal in back substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lti::design::tf;
    use crate::TransferFunction;
    use approx::assert_relative_eq;
    #[test]
    fn test_transfer_function_creation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Test creating a simple transfer function
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("Operation failed");
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
        let tf = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).expect("Operation failed");
        assert_relative_eq!(tf.num[0], 1.0);
        assert_relative_eq!(tf.den[0], 1.0);
        assert_relative_eq!(tf.den[1], 1.0);
    }

    #[test]
    fn test_transfer_function_evaluation() {
        // Test evaluating H(s) = 1/(s+1) at s = 0
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("Operation failed");
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
        .expect("Operation failed");

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
        .expect("Operation failed");
        assert!(zpk_stable.is_stable().expect("Operation failed"));

        // Unstable system (pole in RHP)
        let zpk_unstable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None)
                .expect("Operation failed");
        assert!(!zpk_unstable.is_stable().expect("Operation failed"));
    }

    #[test]
    fn test_state_space_creation() {
        let ss = StateSpace::new(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None)
            .expect("Operation failed");

        assert_eq!(ss.n_states, 1);
        assert_eq!(ss.n_inputs, 1);
        assert_eq!(ss.n_outputs, 1);
        assert!(!ss.dt);
    }

    #[test]
    fn test_state_space_matrix_access() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        let ss = StateSpace::new(
            vec![-1.0, 0.0, 1.0, -2.0], // 2x2 A matrix
            vec![1.0, 0.0],             // 2x1 B matrix
            vec![1.0, 0.0],             // 1x2 C matrix
            vec![0.0],                  // 1x1 D matrix
            None,
        )
        .expect("Operation failed");

        assert_eq!(ss.a(0, 0).expect("Operation failed"), -1.0);
        assert_eq!(ss.a(0, 1).expect("Operation failed"), 0.0);
        assert_eq!(ss.a(1, 0).expect("Operation failed"), 1.0);
        assert_eq!(ss.a(1, 1).expect("Operation failed"), -2.0);

        assert_eq!(ss.b(0, 0).expect("Operation failed"), 1.0);
        assert_eq!(ss.b(1, 0).expect("Operation failed"), 0.0);

        assert_eq!(ss.c(0, 0).expect("Operation failed"), 1.0);
        assert_eq!(ss.c(0, 1).expect("Operation failed"), 0.0);

        assert_eq!(ss.d(0, 0).expect("Operation failed"), 0.0);
    }

    #[test]
    fn test_invalid_denominator() {
        let result = TransferFunction::new(vec![1.0], vec![0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inconsistent_state_space_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
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

#[allow(dead_code)]
fn tf(num: Vec<f64>, den: Vec<f64>) -> TransferFunction {
    TransferFunction::new(num, den, None).expect("Operation failed")
}
