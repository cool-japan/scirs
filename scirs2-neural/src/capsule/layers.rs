//! Capsule Network Layers
//!
//! Provides `PrimaryCaps` (input→capsule conversion) and `DigitCaps`
//! (class-level capsules with transformation matrices).

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Squash activation (module-private + public re-export)
// ---------------------------------------------------------------------------

/// Squash non-linearity that maps a vector v to an output vector with
/// magnitude in [0, 1) while preserving its orientation:
///
/// ```text
/// squash(v) = (||v||² / (1 + ||v||²)) × (v / ||v||)
/// ```
pub fn squash(v: &[f32]) -> Vec<f32> {
    let sq_norm: f32 = v.iter().map(|x| x * x).sum();
    if sq_norm < 1e-12 {
        return vec![0.0_f32; v.len()];
    }
    let norm = sq_norm.sqrt();
    let scale = sq_norm / (1.0 + sq_norm);
    v.iter().map(|&x| scale * x / norm).collect()
}

/// Compute the L2 norm of a vector slice.
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// PrimaryCaps
// ---------------------------------------------------------------------------

/// Primary capsule layer.
///
/// Converts a flat feature vector into `n_capsules` capsule vectors, each of
/// dimension `cap_dim`, using a learned linear projection followed by the
/// squash non-linearity.
///
/// Weight layout: `weights[c][i]` is the weight from input feature `i` to the
/// c-th output element (where c = capsule_index * cap_dim + dim_index).
#[derive(Debug, Clone)]
pub struct PrimaryCaps {
    /// Number of capsule groups in the output
    pub n_capsules: usize,
    /// Dimensionality of each capsule vector
    pub cap_dim: usize,
    /// Linear projection weights: shape [n_capsules * cap_dim][input_features]
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: length n_capsules * cap_dim
    pub bias: Vec<f32>,
}

impl PrimaryCaps {
    /// Create a new `PrimaryCaps` layer with Xavier-style uniform initialisation.
    ///
    /// # Arguments
    /// * `input_size`  — number of input features
    /// * `n_capsules`  — number of output capsule groups
    /// * `cap_dim`     — dimension of each capsule vector
    ///
    /// # Errors
    /// Returns an error if any dimension is zero.
    pub fn new(input_size: usize, n_capsules: usize, cap_dim: usize) -> Result<Self> {
        if input_size == 0 || n_capsules == 0 || cap_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "PrimaryCaps: all dimensions must be > 0".into(),
            ));
        }
        let out_size = n_capsules * cap_dim;
        // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        let limit = (6.0_f32 / (input_size + out_size) as f32).sqrt();
        // Deterministic initialisation using simple pattern (avoids rand dependency)
        let weights: Vec<Vec<f32>> = (0..out_size)
            .map(|c| {
                (0..input_size)
                    .map(|i| {
                        let v = ((c * input_size + i) as f32 * std::f32::consts::E).sin();
                        v * limit
                    })
                    .collect()
            })
            .collect();
        let bias = vec![0.0_f32; out_size];
        Ok(Self {
            n_capsules,
            cap_dim,
            weights,
            bias,
        })
    }

    /// Forward pass: project input to capsule space and apply squash.
    ///
    /// # Arguments
    /// * `input` — feature vector of length `input_size`
    ///
    /// # Returns
    /// Flat capsule output: length `n_capsules * cap_dim`, then interpreted as
    /// n_capsules squashed vectors concatenated.
    ///
    /// # Errors
    /// Returns an error if input length mismatches.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<Vec<f32>>> {
        let n_out = self.n_capsules * self.cap_dim;
        if self.weights.is_empty() || self.weights[0].len() != input.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "PrimaryCaps: input size {} != expected {}",
                input.len(),
                self.weights.first().map(|r| r.len()).unwrap_or(0)
            )));
        }

        // Linear projection
        let mut pre_squash = vec![0.0_f32; n_out];
        for (c, (row, &b)) in self.weights.iter().zip(self.bias.iter()).enumerate() {
            pre_squash[c] = b + row
                .iter()
                .zip(input.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f32>();
        }

        // Split into n_capsules vectors and squash each
        let caps: Vec<Vec<f32>> = (0..self.n_capsules)
            .map(|j| {
                let start = j * self.cap_dim;
                let end = start + self.cap_dim;
                squash(&pre_squash[start..end])
            })
            .collect();

        Ok(caps)
    }
}

// ---------------------------------------------------------------------------
// DigitCaps
// ---------------------------------------------------------------------------

/// Class-level (digit/output) capsule layer.
///
/// Each primary capsule `j` is transformed to a "prediction" `û_{j|i}` for
/// each class capsule `i` via a learned matrix W_{ij}:
///
/// ```text
/// û_{j|i} = W_{ij} × u_j
/// ```
///
/// The class capsules are then computed via dynamic routing (see
/// [`super::dynamic_routing::DynamicRouting`]).
///
/// Weight layout: `W[i][j]` is a `(digit_dim × primary_dim)` matrix stored
/// as a flat vector of length `digit_dim * primary_dim`.
#[derive(Debug, Clone)]
pub struct DigitCaps {
    /// Number of class capsules (e.g. 10 for digit recognition)
    pub n_classes: usize,
    /// Dimension of each class capsule vector
    pub cap_dim: usize,
    /// Number of primary capsules (input capsules)
    pub n_primary: usize,
    /// Input capsule dimension
    pub primary_dim: usize,
    /// Transformation matrices: W[i][j] has shape [cap_dim * primary_dim]
    /// representing the matrix W_{ij} flattened row-major
    pub w: Vec<Vec<Vec<f32>>>,
}

impl DigitCaps {
    /// Create a new `DigitCaps` layer with small random initialisation.
    ///
    /// # Arguments
    /// * `n_classes`   — number of output capsule classes
    /// * `cap_dim`     — dimension of each class capsule vector
    /// * `n_primary`   — number of input primary capsules
    /// * `primary_dim` — dimension of each primary capsule
    ///
    /// # Errors
    /// Returns an error if any dimension is zero.
    pub fn new(
        n_classes: usize,
        cap_dim: usize,
        n_primary: usize,
        primary_dim: usize,
    ) -> Result<Self> {
        if n_classes == 0 || cap_dim == 0 || n_primary == 0 || primary_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "DigitCaps: all dimensions must be > 0".into(),
            ));
        }
        let mat_size = cap_dim * primary_dim;
        let scale = 0.01_f32;
        // Deterministic small weights
        let w: Vec<Vec<Vec<f32>>> = (0..n_classes)
            .map(|i| {
                (0..n_primary)
                    .map(|j| {
                        (0..mat_size)
                            .map(|k| {
                                let v = ((i * n_primary * mat_size + j * mat_size + k) as f32
                                    * 1.6180339)
                                    .sin();
                                v * scale
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            n_classes,
            cap_dim,
            n_primary,
            primary_dim,
            w,
        })
    }

    /// Compute all predictions û_{j|i} = W_{ij} × u_j.
    ///
    /// # Arguments
    /// * `primary_caps` — list of `n_primary` capsule vectors each of length `primary_dim`
    ///
    /// # Returns
    /// Predictions as `u_hat[i][j]` (class i, primary j), each of length `cap_dim`.
    ///
    /// # Errors
    /// Returns an error if lengths mismatch.
    pub fn compute_predictions(&self, primary_caps: &[Vec<f32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        if primary_caps.len() != self.n_primary {
            return Err(NeuralError::DimensionMismatch(format!(
                "DigitCaps: expected {} primary capsules, got {}",
                self.n_primary,
                primary_caps.len()
            )));
        }
        for (j, cap) in primary_caps.iter().enumerate() {
            if cap.len() != self.primary_dim {
                return Err(NeuralError::DimensionMismatch(format!(
                    "DigitCaps: primary capsule {j} has dim {}, expected {}",
                    cap.len(),
                    self.primary_dim
                )));
            }
        }

        // u_hat[i][j] = W[i][j] @ u[j]
        let mut u_hat: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0_f32; self.cap_dim]; self.n_primary]; self.n_classes];

        for (i, u_hat_i) in u_hat.iter_mut().enumerate().take(self.n_classes) {
            for j in 0..self.n_primary {
                let mat = &self.w[i][j]; // [cap_dim * primary_dim]
                let u_j = &primary_caps[j]; // [primary_dim]
                for (d, u_hat_ij_d) in u_hat_i[j].iter_mut().enumerate().take(self.cap_dim) {
                    let row_start = d * self.primary_dim;
                    *u_hat_ij_d = mat[row_start..row_start + self.primary_dim]
                        .iter()
                        .zip(u_j.iter())
                        .map(|(&w, &u)| w * u)
                        .sum();
                }
            }
        }

        Ok(u_hat)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_caps_output_shape() {
        let pc = PrimaryCaps::new(16, 8, 4).expect("operation should succeed");
        let input = vec![0.5_f32; 16];
        let out = pc.forward(&input).expect("operation should succeed");
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn primary_caps_squash_magnitude() {
        let pc = PrimaryCaps::new(16, 4, 8).expect("operation should succeed");
        let input = vec![1.0_f32; 16];
        let out = pc.forward(&input).expect("operation should succeed");
        for cap in &out {
            let mag = l2_norm(cap);
            assert!(mag < 1.0 + 1e-5, "squash output must have magnitude < 1");
        }
    }

    #[test]
    fn primary_caps_rejects_zero_dim() {
        assert!(PrimaryCaps::new(0, 8, 4).is_err());
        assert!(PrimaryCaps::new(16, 0, 4).is_err());
        assert!(PrimaryCaps::new(16, 8, 0).is_err());
    }

    #[test]
    fn squash_zero_vector() {
        let v = vec![0.0_f32; 4];
        let out = squash(&v);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn squash_unit_vector_magnitude() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let out = squash(&v);
        let mag = l2_norm(&out);
        // squash(e1) = 0.5 * e1
        assert!((mag - 0.5).abs() < 1e-5);
    }

    #[test]
    fn digit_caps_prediction_shape() {
        let dc = DigitCaps::new(10, 16, 8, 4).expect("operation should succeed");
        let primary: Vec<Vec<f32>> = (0..8).map(|_| vec![0.1_f32; 4]).collect();
        let u_hat = dc
            .compute_predictions(&primary)
            .expect("operation should succeed");
        assert_eq!(u_hat.len(), 10);
        assert_eq!(u_hat[0].len(), 8);
        assert_eq!(u_hat[0][0].len(), 16);
    }

    #[test]
    fn digit_caps_rejects_mismatched_input() {
        let dc = DigitCaps::new(10, 16, 8, 4).expect("operation should succeed");
        // Only 5 primary caps instead of 8
        let primary: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1_f32; 4]).collect();
        assert!(dc.compute_predictions(&primary).is_err());
    }
}
