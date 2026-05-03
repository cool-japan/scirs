//! S4D: Diagonal State Spaces
//! (Gu, Gupta, Goel, Re, 2022)
//!
//! Simplified S4 with a **diagonal** state matrix A, enabling O(N)
//! per-step computation and closed-form kernel evaluation.
//!
//! Because A is diagonal the recurrence becomes element-wise:
//!
//!   x_k[i] = a_bar[i] * x_{k-1}[i] + b_bar[i] * u_k
//!   y_k    = sum_i c[i] * x_k[i]    + d * u_k
//!
//! And the kernel has the closed form:
//!
//!   K[k] = sum_i c[i] * a_bar[i]^k * b_bar[i]

use crate::error::{Result, TimeSeriesError};
use crate::state_space::hippo::{hippo_matrix, HiPPOVariant};

// ---------------------------------------------------------------------------
// S4D initialisation method
// ---------------------------------------------------------------------------

/// Initialisation strategy for the diagonal of A.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum S4DInit {
    /// S4D-Lin: use the diagonal of HiPPO-LegS.
    ///
    /// A_diag[n] = -(n+1)
    Lin,

    /// S4D-Inv: inverse-spaced diagonal.
    ///
    /// A_diag[n] = -1 / (n+1)
    Inv,

    /// Custom diagonal entries (must have length == state_dim).
    Custom(Vec<f64>),
}

// ---------------------------------------------------------------------------
// S4DConfig
// ---------------------------------------------------------------------------

/// S4D layer configuration.
#[derive(Debug, Clone)]
pub struct S4DConfig {
    /// N: state dimension.
    pub state_dim: usize,
    /// H: input/output dimension (currently single-channel).
    pub input_dim: usize,
    /// Discretization time step.
    pub dt: f64,
    /// Initialisation method for the diagonal A.
    pub init_method: S4DInit,
}

// ---------------------------------------------------------------------------
// S4DLayer
// ---------------------------------------------------------------------------

/// S4D layer with diagonal state matrix.
///
/// Because A is diagonal, all operations are element-wise and
/// storage is O(N) instead of O(N^2).
#[derive(Debug, Clone)]
pub struct S4DLayer {
    /// Discrete diagonal of A: a_bar[i] = 1 + a_diag[i] * dt  (Euler).
    pub a_diag_bar: Vec<f64>,
    /// Discrete B vector: b_bar[i] = b[i] * dt.
    pub b_bar: Vec<f64>,
    /// Output vector C (N).
    pub c: Vec<f64>,
    /// Skip-connection scalar D.
    pub d: f64,
    /// Discretization step (stored for reference).
    pub dt: f64,
    /// State dimension.
    state_dim: usize,
}

impl S4DLayer {
    /// Create a new S4D layer.
    ///
    /// # Errors
    ///
    /// Returns an error if state_dim is 0, dt <= 0, or a custom
    /// diagonal has the wrong length.
    pub fn new(config: S4DConfig) -> Result<Self> {
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

        // Build continuous-time diagonal of A
        let a_diag = match &config.init_method {
            S4DInit::Lin => {
                // Diagonal of HiPPO-LegS: A[n][n] = -(n+1)
                (0..n).map(|i| -((i + 1) as f64)).collect::<Vec<_>>()
            }
            S4DInit::Inv => {
                // Inverse spacing: -1/(n+1)
                (0..n).map(|i| -1.0 / ((i + 1) as f64)).collect::<Vec<_>>()
            }
            S4DInit::Custom(diag) => {
                if diag.len() != n {
                    return Err(TimeSeriesError::DimensionMismatch {
                        expected: n,
                        actual: diag.len(),
                    });
                }
                diag.clone()
            }
        };

        // Get B from HiPPO-LegS (B[n] = sqrt(2n+1))
        let b_cont: Vec<f64> = (0..n).map(|i| ((2 * i + 1) as f64).sqrt()).collect();

        // Euler discretization (element-wise for diagonal A)
        let dt = config.dt;
        let a_diag_bar: Vec<f64> = a_diag.iter().map(|&ai| 1.0 + ai * dt).collect();
        let b_bar: Vec<f64> = b_cont.iter().map(|&bi| bi * dt).collect();

        // Default C = 1/N (normalised), D = 0
        let c = vec![1.0 / (n as f64); n];
        let d = 0.0;

        Ok(Self {
            a_diag_bar,
            b_bar,
            c,
            d,
            dt,
            state_dim: n,
        })
    }

    /// Set custom output parameters C and D.
    pub fn with_output(mut self, c: Vec<f64>, d: f64) -> Result<Self> {
        if c.len() != self.state_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.state_dim,
                actual: c.len(),
            });
        }
        self.c = c;
        self.d = d;
        Ok(self)
    }

    /// Forward pass (recurrent mode, element-wise).
    ///
    /// For each time step t:
    ///   x[i] <- a_bar[i] * x[i] + b_bar[i] * u[t]
    ///   y[t] = sum_i c[i] * x[i] + d * u[t]
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let n = self.state_dim;
        let len = input.len();
        let mut x = vec![0.0_f64; n];
        let mut output = Vec::with_capacity(len);

        for t in 0..len {
            // Element-wise state update first
            for i in 0..n {
                x[i] = self.a_diag_bar[i] * x[i] + self.b_bar[i] * input[t];
            }

            // Compute output from updated state
            let y_t: f64 = self
                .c
                .iter()
                .zip(x.iter())
                .map(|(ci, xi)| ci * xi)
                .sum::<f64>()
                + self.d * input[t];
            output.push(y_t);
        }

        Ok(output)
    }

    /// Compute the convolution kernel of length `length`.
    ///
    /// K[k] = sum_i c[i] * a_bar[i]^k * b_bar[i]
    pub fn compute_kernel(&self, length: usize) -> Vec<f64> {
        let n = self.state_dim;
        let mut kernel = Vec::with_capacity(length);

        // Powers of a_bar[i], starting at a_bar^0 = 1
        let mut powers = vec![1.0_f64; n];

        for _k in 0..length {
            let val: f64 = (0..n).map(|i| self.c[i] * powers[i] * self.b_bar[i]).sum();
            kernel.push(val);

            // Advance powers
            for i in 0..n {
                powers[i] *= self.a_diag_bar[i];
            }
        }

        kernel
    }

    /// Convolutional forward pass using the pre-computed kernel.
    pub fn forward_convolutional(&self, input: &[f64]) -> Result<Vec<f64>> {
        let len = input.len();
        if len == 0 {
            return Ok(vec![]);
        }

        let kernel = self.compute_kernel(len);

        let mut output = Vec::with_capacity(len);
        for t in 0..len {
            let mut y_t = self.d * input[t];
            for k in 0..=t {
                y_t += kernel[k] * input[t - k];
            }
            output.push(y_t);
        }

        Ok(output)
    }

    /// Return the state dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n: usize) -> S4DConfig {
        S4DConfig {
            state_dim: n,
            input_dim: 1,
            dt: 0.01,
            init_method: S4DInit::Lin,
        }
    }

    #[test]
    fn test_s4d_creation() {
        let layer = S4DLayer::new(default_config(8)).expect("should succeed");
        assert_eq!(layer.state_dim(), 8);
        assert_eq!(layer.a_diag_bar.len(), 8);
        assert_eq!(layer.b_bar.len(), 8);
        assert_eq!(layer.c.len(), 8);
    }

    #[test]
    fn test_s4d_forward_output_length() {
        let layer = S4DLayer::new(default_config(4)).expect("should succeed");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = layer.forward(&input).expect("should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_s4d_kernel_formula() {
        // K[0] = sum_i c[i] * 1 * b_bar[i] = sum(c .* b_bar)
        let layer = S4DLayer::new(default_config(4)).expect("should succeed");
        let kernel = layer.compute_kernel(3);

        let expected_k0: f64 = layer
            .c
            .iter()
            .zip(layer.b_bar.iter())
            .map(|(ci, bi)| ci * bi)
            .sum();
        assert!(
            (kernel[0] - expected_k0).abs() < 1e-12,
            "K[0]={} expected {}",
            kernel[0],
            expected_k0
        );

        // K[1] = sum_i c[i] * a_bar[i] * b_bar[i]
        let expected_k1: f64 = (0..4)
            .map(|i| layer.c[i] * layer.a_diag_bar[i] * layer.b_bar[i])
            .sum();
        assert!(
            (kernel[1] - expected_k1).abs() < 1e-12,
            "K[1]={} expected {}",
            kernel[1],
            expected_k1
        );
    }

    #[test]
    fn test_s4d_recurrent_conv_equivalence() {
        let layer = S4DLayer::new(default_config(4)).expect("should succeed");
        let input = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4];

        let y_rec = layer.forward(&input).expect("recurrent");
        let y_conv = layer.forward_convolutional(&input).expect("convolutional");

        assert_eq!(y_rec.len(), y_conv.len());
        for (i, (r, c)) in y_rec.iter().zip(y_conv.iter()).enumerate() {
            assert!(
                (r - c).abs() < 1e-10,
                "mismatch at {}: rec={} conv={}",
                i,
                r,
                c
            );
        }
    }

    #[test]
    fn test_s4d_lin_diagonal() {
        let layer = S4DLayer::new(default_config(4)).expect("should succeed");
        // Continuous diagonal should be -(n+1), so discrete = 1 + (-(n+1))*dt
        let dt = 0.01;
        for i in 0..4 {
            let expected = 1.0 - ((i + 1) as f64) * dt;
            assert!(
                (layer.a_diag_bar[i] - expected).abs() < 1e-12,
                "a_bar[{}]={} expected {}",
                i,
                layer.a_diag_bar[i],
                expected
            );
        }
    }

    #[test]
    fn test_s4d_inv_init() {
        let config = S4DConfig {
            state_dim: 4,
            input_dim: 1,
            dt: 0.01,
            init_method: S4DInit::Inv,
        };
        let layer = S4DLayer::new(config).expect("should succeed");
        // A_diag[0] = -1/1 = -1, discrete = 1 + (-1)*0.01 = 0.99
        assert!((layer.a_diag_bar[0] - 0.99).abs() < 1e-12);
        // A_diag[1] = -1/2, discrete = 1 + (-0.5)*0.01 = 0.995
        assert!((layer.a_diag_bar[1] - 0.995).abs() < 1e-12);
    }

    #[test]
    fn test_s4d_custom_init() {
        let config = S4DConfig {
            state_dim: 3,
            input_dim: 1,
            dt: 0.1,
            init_method: S4DInit::Custom(vec![-2.0, -4.0, -6.0]),
        };
        let layer = S4DLayer::new(config).expect("should succeed");
        assert!((layer.a_diag_bar[0] - 0.8).abs() < 1e-12);
        assert!((layer.a_diag_bar[1] - 0.6).abs() < 1e-12);
        assert!((layer.a_diag_bar[2] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_s4d_custom_wrong_length() {
        let config = S4DConfig {
            state_dim: 3,
            input_dim: 1,
            dt: 0.1,
            init_method: S4DInit::Custom(vec![-1.0, -2.0]),
        };
        assert!(S4DLayer::new(config).is_err());
    }

    #[test]
    fn test_s4d_invalid_config() {
        assert!(S4DLayer::new(S4DConfig {
            state_dim: 0,
            input_dim: 1,
            dt: 0.01,
            init_method: S4DInit::Lin,
        })
        .is_err());

        assert!(S4DLayer::new(S4DConfig {
            state_dim: 4,
            input_dim: 1,
            dt: 0.0,
            init_method: S4DInit::Lin,
        })
        .is_err());
    }

    #[test]
    fn test_s4d_empty_input() {
        let layer = S4DLayer::new(default_config(4)).expect("should succeed");
        let output = layer.forward(&[]).expect("should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_s4d_with_output() {
        let layer = S4DLayer::new(default_config(4))
            .expect("should succeed")
            .with_output(vec![1.0, 0.0, 0.0, 0.0], 0.5)
            .expect("with_output");
        assert!((layer.c[0] - 1.0).abs() < 1e-15);
        assert!((layer.d - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_s4d_compare_with_s4_diagonal() {
        // S4D-Lin should approximate the diagonal part of S4 with LegS
        use crate::state_space::s4::{DiscretizationMethod, S4Config, S4Layer};

        let n = 4;
        let dt = 0.001; // small dt for better agreement

        let s4_layer = S4Layer::new(S4Config {
            state_dim: n,
            input_dim: 1,
            dt,
            hippo_variant: HiPPOVariant::LegS,
            discretization: DiscretizationMethod::Euler,
        })
        .expect("s4");

        let s4d_layer = S4DLayer::new(S4DConfig {
            state_dim: n,
            input_dim: 1,
            dt,
            init_method: S4DInit::Lin,
        })
        .expect("s4d");

        // The diagonal of S4's A_bar should match S4D's a_diag_bar
        for i in 0..n {
            assert!(
                (s4_layer.a_bar[i][i] - s4d_layer.a_diag_bar[i]).abs() < 1e-10,
                "diagonal mismatch at {}: s4={} s4d={}",
                i,
                s4_layer.a_bar[i][i],
                s4d_layer.a_diag_bar[i]
            );
        }
    }
}
