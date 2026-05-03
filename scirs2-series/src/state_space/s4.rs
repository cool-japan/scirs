//! S4: Structured State Spaces for Sequence Modeling
//! (Gu, Goel, Re, 2022)
//!
//! Implements the S4 layer with both recurrent and convolutional modes.
//!
//! A continuous-time SSM  x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t)
//! is discretised to obtain
//!
//!   x_{k+1} = A_bar x_k + B_bar u_k
//!   y_k     = C x_k     + D u_k
//!
//! Two equivalent forward-pass modes are provided:
//!
//! 1. **Recurrent** -- O(LN) per step, sequential.
//! 2. **Convolutional** -- pre-compute kernel K of length L, then
//!    convolve: O(L log L) via FFT (here we do naive O(L^2) for clarity).

use crate::error::{Result, TimeSeriesError};
use crate::state_space::hippo::{hippo_matrix, HiPPOVariant};

// ---------------------------------------------------------------------------
// Discretization method
// ---------------------------------------------------------------------------

/// Discretization method for continuous-to-discrete conversion.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DiscretizationMethod {
    /// Zero-Order Hold (simplified): A_bar = I + A*dt, B_bar = B*dt.
    ///
    /// Exact ZOH would use matrix exponential; this first-order
    /// approximation is accurate for small dt.
    ZOH,

    /// Bilinear (Tustin) transform:
    ///   A_bar = (I + A*dt/2) (I - A*dt/2)^{-1}
    ///   B_bar = dt * (I - A*dt/2)^{-1} B
    Bilinear,

    /// Forward Euler: A_bar = I + A*dt, B_bar = B*dt.
    ///
    /// Identical to the simplified ZOH above.
    Euler,
}

// ---------------------------------------------------------------------------
// S4Config
// ---------------------------------------------------------------------------

/// S4 layer configuration.
#[derive(Debug, Clone)]
pub struct S4Config {
    /// N: state dimension (e.g. 64).
    pub state_dim: usize,
    /// H: input/output dimension.
    pub input_dim: usize,
    /// Discretization time step.
    pub dt: f64,
    /// Which HiPPO variant to use for initialising (A, B).
    pub hippo_variant: HiPPOVariant,
    /// Discretization method.
    pub discretization: DiscretizationMethod,
}

// ---------------------------------------------------------------------------
// S4Layer
// ---------------------------------------------------------------------------

/// S4 layer (after discretization).
///
/// Stores the discrete matrices and provides both recurrent and
/// convolutional forward passes.
#[derive(Debug, Clone)]
pub struct S4Layer {
    /// Discrete state matrix A_bar (N x N), row-major.
    pub a_bar: Vec<Vec<f64>>,
    /// Discrete input vector B_bar (N).
    pub b_bar: Vec<f64>,
    /// Output vector C (N).
    pub c: Vec<f64>,
    /// Skip-connection scalar D.
    pub d: f64,
    /// Configuration used to build this layer.
    pub config: S4Config,
}

impl S4Layer {
    /// Create a new S4 layer by constructing the HiPPO matrices
    /// and discretising them according to `config`.
    ///
    /// The output matrix C is initialised to all ones and D = 0.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`hippo_matrix`] or from the
    /// discretization step (e.g. singular matrix in Bilinear).
    pub fn new(config: S4Config) -> Result<Self> {
        let n = config.state_dim;
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "state_dim must be > 0".into(),
            ));
        }
        if config.dt <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "dt".into(),
                message: "discretization step must be > 0".into(),
            });
        }

        let (a_cont, b_cont) = hippo_matrix(&config.hippo_variant, n)?;

        let (a_bar, b_bar) = match &config.discretization {
            DiscretizationMethod::ZOH | DiscretizationMethod::Euler => {
                discretize_euler(&a_cont, &b_cont, config.dt)
            }
            DiscretizationMethod::Bilinear => discretize_bilinear(&a_cont, &b_cont, config.dt)?,
        };

        let c = vec![1.0_f64; n];
        let d = 0.0;

        Ok(Self {
            a_bar,
            b_bar,
            c,
            d,
            config,
        })
    }

    /// Create an S4 layer with custom C and D parameters.
    pub fn with_output(mut self, c: Vec<f64>, d: f64) -> Result<Self> {
        let n = self.config.state_dim;
        if c.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: c.len(),
            });
        }
        self.c = c;
        self.d = d;
        Ok(self)
    }

    /// Recurrent mode: process a full input sequence step-by-step.
    ///
    /// ```text
    /// x_{k+1} = A_bar * x_k + B_bar * u_k
    /// y_k     = C * x_k     + D * u_k
    /// ```
    ///
    /// Returns the output sequence of the same length as `input`.
    pub fn forward_recurrent(&self, input: &[f64]) -> Result<Vec<f64>> {
        let n = self.config.state_dim;
        let len = input.len();
        let mut x = vec![0.0_f64; n];
        let mut output = Vec::with_capacity(len);

        for t in 0..len {
            // x_t = A_bar * x_{t-1} + B_bar * u_t  (state update first)
            let x_new = mat_vec_add_scaled(&self.a_bar, &x, &self.b_bar, input[t]);
            x = x_new;

            // y_t = C . x_t + D * u_t
            let y_t: f64 = dot(&self.c, &x) + self.d * input[t];
            output.push(y_t);
        }

        Ok(output)
    }

    /// Convolutional mode: pre-compute kernel K then convolve.
    ///
    /// ```text
    /// K[k] = C * A_bar^k * B_bar   for k = 0 .. L-1
    /// y = conv(K, u)
    /// ```
    ///
    /// Returns the output sequence of the same length as `input`.
    pub fn forward_convolutional(&self, input: &[f64]) -> Result<Vec<f64>> {
        let len = input.len();
        if len == 0 {
            return Ok(vec![]);
        }

        let kernel = self.compute_kernel(len)?;

        // Causal convolution: y[t] = sum_{k=0}^{t} K[k] * u[t-k]  + D*u[t]
        let mut output = Vec::with_capacity(len);
        for t in 0..len {
            let mut y_t = self.d * input[t];
            let k_max = t + 1; // at most t+1 kernel elements
            for k in 0..k_max {
                y_t += kernel[k] * input[t - k];
            }
            output.push(y_t);
        }

        Ok(output)
    }

    /// Compute the convolution kernel of length `length`.
    ///
    /// K[k] = C^T A_bar^k B_bar
    fn compute_kernel(&self, length: usize) -> Result<Vec<f64>> {
        let n = self.config.state_dim;
        let mut kernel = Vec::with_capacity(length);

        // A_bar^0 * B_bar = B_bar
        let mut a_power_b = self.b_bar.clone();

        for _k in 0..length {
            // K[k] = C . (A_bar^k * B_bar)
            let val = dot(&self.c, &a_power_b);
            kernel.push(val);

            // A_bar^{k+1} * B_bar = A_bar * (A_bar^k * B_bar)
            let next = mat_vec(&self.a_bar, &a_power_b, n);
            a_power_b = next;
        }

        Ok(kernel)
    }

    /// Return the state dimension N.
    pub fn state_dim(&self) -> usize {
        self.config.state_dim
    }
}

// ---------------------------------------------------------------------------
// Discretization helpers
// ---------------------------------------------------------------------------

/// Forward Euler / simplified ZOH: A_bar = I + A*dt,  B_bar = B*dt
fn discretize_euler(a: &[Vec<f64>], b: &[f64], dt: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = a.len();
    let mut a_bar = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_bar[i][j] = a[i][j] * dt;
            if i == j {
                a_bar[i][j] += 1.0;
            }
        }
    }
    let b_bar: Vec<f64> = b.iter().map(|&bi| bi * dt).collect();
    (a_bar, b_bar)
}

/// Bilinear (Tustin) discretization:
///   A_bar = (I + A*dt/2) * inv(I - A*dt/2)
///   B_bar = dt * inv(I - A*dt/2) * B
fn discretize_bilinear(a: &[Vec<f64>], b: &[f64], dt: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let n = a.len();
    let half_dt = dt / 2.0;

    // Construct (I + A*dt/2) and (I - A*dt/2)
    let mut plus = vec![vec![0.0_f64; n]; n];
    let mut minus = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let a_dt_half = a[i][j] * half_dt;
            plus[i][j] = a_dt_half;
            minus[i][j] = -a_dt_half;
            if i == j {
                plus[i][j] += 1.0;
                minus[i][j] += 1.0;
            }
        }
    }

    // Invert (I - A*dt/2) using Gauss-Jordan
    let minus_inv = invert_matrix(&minus)?;

    // A_bar = plus * minus_inv
    let a_bar = mat_mat(&plus, &minus_inv, n);

    // B_bar = dt * minus_inv * B
    let inv_b = mat_vec(&minus_inv, b, n);
    let b_bar: Vec<f64> = inv_b.iter().map(|&v| v * dt).collect();

    Ok((a_bar, b_bar))
}

// ---------------------------------------------------------------------------
// Tiny linear algebra helpers
// ---------------------------------------------------------------------------

/// Dot product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Matrix-vector product: out = M * v
#[inline]
fn mat_vec(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += m[i][j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// M * x + scale * b (fused multiply-add for recurrence).
#[inline]
fn mat_vec_add_scaled(m: &[Vec<f64>], x: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i] * scale;
        for j in 0..n {
            s += m[i][j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Matrix-matrix product: C = A * B (both n x n).
fn mat_mat(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for l in 0..n {
                s += a[i][l] * b[l][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Gauss-Jordan matrix inversion for small (N x N) matrices.
///
/// Returns `Err` if the matrix is singular.
fn invert_matrix(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    // Augmented matrix [M | I]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(&m[i]);
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        aug.push(row);
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(TimeSeriesError::NumericalInstability(
                "singular matrix in bilinear discretization".into(),
            ));
        }
        if max_row != col {
            aug.swap(max_row, col);
        }

        // Scale pivot row
        let pivot = aug[col][col];
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * n) {
            aug[col][j] *= inv_pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract right half
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n: usize) -> S4Config {
        S4Config {
            state_dim: n,
            input_dim: 1,
            dt: 0.01,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Euler,
        }
    }

    #[test]
    fn test_s4_layer_creation() {
        let layer = S4Layer::new(default_config(4)).expect("should succeed");
        assert_eq!(layer.a_bar.len(), 4);
        assert_eq!(layer.b_bar.len(), 4);
        assert_eq!(layer.c.len(), 4);
    }

    #[test]
    fn test_recurrent_output_shape() {
        let layer = S4Layer::new(default_config(8)).expect("should succeed");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = layer.forward_recurrent(&input).expect("should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_convolutional_output_shape() {
        let layer = S4Layer::new(default_config(8)).expect("should succeed");
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let output = layer.forward_convolutional(&input).expect("should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_recurrent_convolutional_equivalence() {
        let layer = S4Layer::new(default_config(4)).expect("should succeed");
        let input = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6];

        let y_rec = layer.forward_recurrent(&input).expect("recurrent");
        let y_conv = layer.forward_convolutional(&input).expect("convolutional");

        assert_eq!(y_rec.len(), y_conv.len());
        for (i, (r, c)) in y_rec.iter().zip(y_conv.iter()).enumerate() {
            assert!(
                (r - c).abs() < 1e-8,
                "mismatch at index {}: rec={} conv={}",
                i,
                r,
                c
            );
        }
    }

    #[test]
    fn test_kernel_first_element() {
        // K[0] = C . B_bar
        let layer = S4Layer::new(default_config(4)).expect("should succeed");
        let kernel = layer.compute_kernel(5).expect("kernel");
        let expected = dot(&layer.c, &layer.b_bar);
        assert!(
            (kernel[0] - expected).abs() < 1e-12,
            "K[0]={} expected {}",
            kernel[0],
            expected
        );
    }

    #[test]
    fn test_zoh_small_dt_approximation() {
        // For very small dt, A_bar ≈ I + A*dt, which is exactly what Euler does
        let config = S4Config {
            state_dim: 4,
            input_dim: 1,
            dt: 1e-6,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::ZOH,
        };
        let layer = S4Layer::new(config).expect("should succeed");
        // Diagonal of A_bar should be close to 1 + A[i][i]*dt
        let (a_cont, _) = hippo_matrix(&HiPPOVariant::LegS, 4).expect("hippo");
        for i in 0..4 {
            let expected = 1.0 + a_cont[i][i] * 1e-6;
            assert!(
                (layer.a_bar[i][i] - expected).abs() < 1e-10,
                "A_bar[{}][{}]={} expected {}",
                i,
                i,
                layer.a_bar[i][i],
                expected
            );
        }
    }

    #[test]
    fn test_bilinear_discretization() {
        let config = S4Config {
            state_dim: 4,
            input_dim: 1,
            dt: 0.01,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Bilinear,
        };
        let layer = S4Layer::new(config).expect("should succeed");
        // Bilinear should produce a valid layer
        assert_eq!(layer.a_bar.len(), 4);
        assert_eq!(layer.b_bar.len(), 4);

        // Output should be finite
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = layer.forward_recurrent(&input).expect("should succeed");
        for y in &output {
            assert!(y.is_finite(), "output should be finite, got {}", y);
        }
    }

    #[test]
    fn test_bilinear_vs_euler_close_for_small_dt() {
        let dt = 1e-5;
        let n = 4;

        let euler_layer = S4Layer::new(S4Config {
            state_dim: n,
            input_dim: 1,
            dt,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Euler,
        })
        .expect("euler");

        let bilinear_layer = S4Layer::new(S4Config {
            state_dim: n,
            input_dim: 1,
            dt,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Bilinear,
        })
        .expect("bilinear");

        // For tiny dt both should produce very similar A_bar
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (euler_layer.a_bar[i][j] - bilinear_layer.a_bar[i][j]).abs() < 1e-6,
                    "A_bar[{}][{}] mismatch: euler={} bilinear={}",
                    i,
                    j,
                    euler_layer.a_bar[i][j],
                    bilinear_layer.a_bar[i][j]
                );
            }
        }
    }

    #[test]
    fn test_empty_input() {
        let layer = S4Layer::new(default_config(4)).expect("should succeed");
        let output = layer.forward_convolutional(&[]).expect("should succeed");
        assert!(output.is_empty());
        let output = layer.forward_recurrent(&[]).expect("should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_with_output() {
        let layer = S4Layer::new(default_config(4))
            .expect("should succeed")
            .with_output(vec![1.0, 0.0, 0.0, 0.0], 0.5)
            .expect("with_output");
        assert!((layer.c[0] - 1.0).abs() < 1e-15);
        assert!((layer.d - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_with_output_wrong_dim() {
        let result = S4Layer::new(default_config(4))
            .expect("should succeed")
            .with_output(vec![1.0, 0.0], 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config() {
        let result = S4Layer::new(S4Config {
            state_dim: 0,
            input_dim: 1,
            dt: 0.01,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Euler,
        });
        assert!(result.is_err());

        let result = S4Layer::new(S4Config {
            state_dim: 4,
            input_dim: 1,
            dt: -0.01,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Euler,
        });
        assert!(result.is_err());
    }
}
