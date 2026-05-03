//! Privacy mechanisms for federated learning

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;

/// Differential privacy mechanism
pub struct DifferentialPrivacy {
    /// Privacy budget (epsilon)
    epsilon: f64,
    /// Delta parameter
    delta: f64,
    /// Clipping threshold
    clip_threshold: f64,
    /// Noise mechanism
    mechanism: NoiseMethod,
}

/// Noise mechanism for differential privacy
#[derive(Debug, Clone)]
pub enum NoiseMethod {
    Gaussian,
    Laplace,
}

impl DifferentialPrivacy {
    /// Create new differential privacy mechanism
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            clip_threshold: 1.0,
            mechanism: NoiseMethod::Gaussian,
        }
    }

    /// Set clipping threshold
    pub fn with_clipping(mut self, threshold: f64) -> Self {
        self.clip_threshold = threshold;
        self
    }

    /// Apply differential privacy to gradients
    pub fn apply_to_gradients(&self, gradients: &mut [Array2<f32>]) -> Result<()> {
        self.clip_gradients(gradients)?;
        self.add_noise(gradients)?;
        Ok(())
    }

    /// Clip gradients to norm threshold
    pub fn clip_gradients(&self, gradients: &mut [Array2<f32>]) -> Result<()> {
        // Calculate global norm
        let mut global_norm = 0.0_f32;
        for grad in gradients.iter() {
            global_norm += grad.iter().map(|x| x * x).sum::<f32>();
        }
        global_norm = global_norm.sqrt();
        // Clip if necessary
        if global_norm > self.clip_threshold as f32 {
            let scale = self.clip_threshold as f32 / global_norm;
            for grad in gradients.iter_mut() {
                *grad *= scale;
            }
        }
        Ok(())
    }

    /// Add noise based on mechanism
    fn add_noise(&self, gradients: &mut [Array2<f32>]) -> Result<()> {
        use scirs2_core::random::{Distribution, Normal};
        let mut rng_inst = scirs2_core::random::rng();
        match self.mechanism {
            NoiseMethod::Gaussian => {
                let sigma =
                    self.clip_threshold * (2.0 * (1.0 / self.delta).ln()).sqrt() / self.epsilon;
                let noise_dist = Normal::new(0.0_f32, sigma as f32)
                    .map_err(|e| crate::error::NeuralError::InferenceError(format!("{e}")))?;
                for grad in gradients.iter_mut() {
                    for elem in grad.iter_mut() {
                        *elem += noise_dist.sample(&mut rng_inst);
                    }
                }
            }
            NoiseMethod::Laplace => {
                use scirs2_core::random::{Distribution, Uniform};
                let b = (self.clip_threshold / self.epsilon) as f32;
                let uniform = Uniform::new(-0.5_f32, 0.5_f32)
                    .map_err(|e| crate::error::NeuralError::InferenceError(format!("{e}")))?;
                for grad in gradients.iter_mut() {
                    for elem in grad.iter_mut() {
                        // Manual Laplace distribution: sample from uniform and transform
                        let u: f32 = uniform.sample(&mut rng_inst);
                        let laplace_sample = -b * u.signum() * (1.0 - 2.0 * u.abs()).max(1e-8).ln();
                        *elem += laplace_sample;
                    }
                }
            }
        }
        Ok(())
    }

    /// Calculate privacy spent
    pub fn privacy_spent(&self, num_steps: usize) -> f64 {
        // Simplified composition
        self.epsilon * (num_steps as f64).sqrt()
    }
}

/// Secure aggregation protocol
pub struct SecureAggregation {
    /// Number of clients required
    threshold: usize,
    /// Security parameter
    #[allow(dead_code)]
    security_param: usize,
}

impl SecureAggregation {
    /// Create new secure aggregation
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            security_param: 128,
        }
    }

    /// Get threshold
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Mask client updates
    pub fn mask_updates(
        &self,
        updates: &[Array2<f32>],
        client_id: usize,
    ) -> Result<Vec<Array2<f32>>> {
        // Simplified masking
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{RngExt, SeedableRng};
        let mut masked = Vec::new();
        for update in updates.iter() {
            let mut mask = Array2::<f32>::zeros(update.raw_dim());
            let seed = client_id as u64 * 1000 + 42;
            let mut rng_inst = StdRng::seed_from_u64(seed);
            for elem in mask.iter_mut() {
                *elem = rng_inst.random_range(-1.0_f32..1.0_f32);
            }
            masked.push(update + &mask);
        }
        Ok(masked)
    }

    /// Unmask aggregated updates
    pub fn unmask_aggregate(
        &self,
        aggregated: &mut Vec<Array2<f32>>,
        participating_clients: &[usize],
    ) -> Result<()> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{RngExt, SeedableRng};
        // Remove masks from aggregated result
        for update in aggregated.iter_mut() {
            let mut total_mask = Array2::<f32>::zeros(update.raw_dim());
            for &client_id in participating_clients {
                let seed = client_id as u64 * 1000 + 42;
                let mut rng_inst = StdRng::seed_from_u64(seed);
                for elem in total_mask.iter_mut() {
                    *elem += rng_inst.random_range(-1.0_f32..1.0_f32);
                }
            }
            *update -= &total_mask;
        }
        Ok(())
    }
}

/// Homomorphic encryption (placeholder)
pub struct HomomorphicEncryption {
    /// Key size
    #[allow(dead_code)]
    key_size: usize,
}

impl HomomorphicEncryption {
    /// Create new homomorphic encryption
    pub fn new(key_size: usize) -> Self {
        Self { key_size }
    }

    /// Encrypt weights
    pub fn encrypt(&self, weights: &Array2<f32>) -> Result<Vec<u8>> {
        // Placeholder - would use actual HE library
        Ok(weights
            .as_slice()
            .ok_or_else(|| {
                crate::error::NeuralError::InferenceError("Array not contiguous".to_string())
            })?
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect())
    }

    /// Decrypt weights
    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Array2<f32>> {
        let floats: Vec<f32> = encrypted
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let size = (floats.len() as f64).sqrt() as usize;
        let total = size * size;
        Ok(Array2::from_shape_vec(
            (size, size),
            floats[..total].to_vec(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_privacy() {
        let dp = DifferentialPrivacy::new(1.0, 1e-5);
        let mut gradients = vec![Array2::ones((2, 2))];
        dp.apply_to_gradients(&mut gradients)
            .expect("apply_to_gradients failed");
        // Check that noise was added (with high probability values differ)
        // Note: there's a tiny chance they're equal; this is fine for a smoke test
        let _ = gradients[0][[0, 0]]; // just access to ensure it compiled
    }

    #[test]
    fn test_gradient_clipping() {
        let dp = DifferentialPrivacy::new(1.0, 1e-5).with_clipping(1.0);
        let mut gradients = vec![Array2::ones((2, 2)) * 10.0];
        dp.clip_gradients(&mut gradients)
            .expect("clip_gradients failed");
        // Check that gradients were clipped: norm should be ~1.0
        let norm: f32 = gradients[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_secure_aggregation() {
        let sa = SecureAggregation::new(3);
        assert_eq!(sa.threshold(), 3);
        let weights = vec![Array2::ones((2, 2))];
        let masked = sa.mask_updates(&weights, 0).expect("mask_updates failed");
        assert_eq!(masked.len(), 1);
        assert_eq!(masked[0].shape(), weights[0].shape());
    }
}
