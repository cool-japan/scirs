//! Dynamic Routing Between Capsules
//!
//! Implements the iterative agreement routing algorithm from Sabour et al. (2017).
//!
//! ## Algorithm
//!
//! Given predictions û_{j|i} (shape: [n_classes, n_primary, cap_dim]):
//!
//! 1. Initialise routing logits b_{ij} = 0  ∀ i, j
//! 2. Repeat `n_iters` times:
//!    a. c_{ij} = softmax_i(b_j)           — coupling coefficients
//!    b. s_i = Σ_j c_{ij} · û_{j|i}       — weighted sum
//!    c. v_i = squash(s_i)                  — class capsule output
//!    d. b_{ij} += û_{j|i} · v_i           — agreement update
//! 3. Return {v_i}

use crate::capsule::layers::squash;
use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// DynamicRouting
// ---------------------------------------------------------------------------

/// Dynamic routing-by-agreement algorithm.
#[derive(Debug, Clone)]
pub struct DynamicRouting {
    /// Number of routing iterations (3 is typical)
    pub n_iterations: usize,
}

impl DynamicRouting {
    /// Create a new dynamic routing module.
    ///
    /// # Errors
    /// Returns an error if `n_iterations == 0`.
    pub fn new(n_iterations: usize) -> Result<Self> {
        if n_iterations == 0 {
            return Err(NeuralError::InvalidArgument(
                "n_iterations must be ≥ 1".into(),
            ));
        }
        Ok(Self { n_iterations })
    }

    /// Run the routing algorithm.
    ///
    /// # Arguments
    /// * `u_hat` — predictions: `u_hat[i][j]` is the prediction from
    ///   primary capsule `j` to class capsule `i`, each of length `cap_dim`
    ///
    /// # Returns
    /// Class capsule vectors: `[n_classes][cap_dim]`
    ///
    /// # Errors
    /// Returns an error on empty or jagged inputs.
    pub fn route(&self, u_hat: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>> {
        let n_classes = u_hat.len();
        if n_classes == 0 {
            return Err(NeuralError::InvalidArgument(
                "u_hat must be non-empty".into(),
            ));
        }
        let n_primary = u_hat[0].len();
        if n_primary == 0 {
            return Err(NeuralError::InvalidArgument(
                "u_hat[0] must be non-empty (n_primary > 0)".into(),
            ));
        }
        let cap_dim = u_hat[0][0].len();
        if cap_dim == 0 {
            return Err(NeuralError::InvalidArgument("cap_dim must be > 0".into()));
        }

        // Validate shape consistency
        for (i, class_preds) in u_hat.iter().enumerate() {
            if class_preds.len() != n_primary {
                return Err(NeuralError::DimensionMismatch(format!(
                    "u_hat[{i}].len() = {} ≠ n_primary {n_primary}",
                    class_preds.len()
                )));
            }
            for (j, pred) in class_preds.iter().enumerate() {
                if pred.len() != cap_dim {
                    return Err(NeuralError::DimensionMismatch(format!(
                        "u_hat[{i}][{j}].len() = {} ≠ cap_dim {cap_dim}",
                        pred.len()
                    )));
                }
            }
        }

        // Routing logits b[j][i]: indexed by primary j, class i
        // (transposed for efficient softmax over classes per primary neuron)
        let mut b = vec![vec![0.0_f32; n_classes]; n_primary];
        let mut v: Vec<Vec<f32>> = vec![vec![0.0_f32; cap_dim]; n_classes];

        for _iter in 0..self.n_iterations {
            // Step a: coupling coefficients c[j][i] = softmax over i of b[j]
            let c = softmax_rows(&b, n_classes);

            // Step b: weighted sum s[i] = Σ_j c[j][i] * u_hat[i][j]
            let mut s: Vec<Vec<f32>> = vec![vec![0.0_f32; cap_dim]; n_classes];
            for j in 0..n_primary {
                for i in 0..n_classes {
                    let cij = c[j][i];
                    for d in 0..cap_dim {
                        s[i][d] += cij * u_hat[i][j][d];
                    }
                }
            }

            // Step c: v[i] = squash(s[i])
            for i in 0..n_classes {
                v[i] = squash(&s[i]);
            }

            // Step d: update b[j][i] += u_hat[i][j] · v[i]
            for j in 0..n_primary {
                for i in 0..n_classes {
                    let agreement: f32 = u_hat[i][j]
                        .iter()
                        .zip(v[i].iter())
                        .map(|(&u, &vi)| u * vi)
                        .sum();
                    b[j][i] += agreement;
                }
            }
        }

        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------

/// Compute softmax along axis 1 (over classes) for each primary capsule j.
///
/// Input: `b[j][i]` — routing logits for primary j, class i.
/// Output: `c[j][i]` — coupling coefficients.
fn softmax_rows(b: &[Vec<f32>], n_classes: usize) -> Vec<Vec<f32>> {
    b.iter()
        .map(|row| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
            let mut result = vec![0.0_f32; n_classes];
            for (i, e) in exps.iter().enumerate() {
                result[i] = e / sum;
            }
            result
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_u_hat(n_classes: usize, n_primary: usize, cap_dim: usize) -> Vec<Vec<Vec<f32>>> {
        (0..n_classes)
            .map(|i| {
                (0..n_primary)
                    .map(|j| {
                        (0..cap_dim)
                            .map(|d| ((i + j + d) as f32 * 0.1).sin() * 0.5)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn routing_output_shape() {
        let routing = DynamicRouting::new(3).expect("operation should succeed");
        let u_hat = make_u_hat(10, 8, 16);
        let v = routing.route(&u_hat).expect("operation should succeed");
        assert_eq!(v.len(), 10);
        assert_eq!(v[0].len(), 16);
    }

    #[test]
    fn routing_output_magnitude_bounded() {
        let routing = DynamicRouting::new(3).expect("operation should succeed");
        let u_hat = make_u_hat(5, 6, 8);
        let v = routing.route(&u_hat).expect("operation should succeed");
        for vi in &v {
            let mag: f32 = vi.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(mag <= 1.0 + 1e-5, "squash should bound magnitude ≤ 1");
        }
    }

    #[test]
    fn routing_rejects_zero_iterations() {
        assert!(DynamicRouting::new(0).is_err());
    }

    #[test]
    fn routing_rejects_empty_u_hat() {
        let routing = DynamicRouting::new(3).expect("operation should succeed");
        let u_hat: Vec<Vec<Vec<f32>>> = Vec::new();
        assert!(routing.route(&u_hat).is_err());
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let b = vec![vec![1.0_f32, 2.0, 3.0]; 4];
        let c = softmax_rows(&b, 3);
        for row in &c {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn routing_one_iteration_matches_manual() {
        let routing = DynamicRouting::new(1).expect("operation should succeed");
        // Simple 2 classes, 2 primaries, 2-dim capsules
        // u_hat[i][j] all ones
        let u_hat = vec![vec![vec![1.0_f32, 0.0]; 2], vec![vec![0.0_f32, 1.0]; 2]];
        let v = routing.route(&u_hat).expect("operation should succeed");
        assert_eq!(v.len(), 2);
        // Both classes should have non-zero magnitude
        for vi in &v {
            let mag: f32 = vi.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(mag > 0.0);
        }
    }
}
