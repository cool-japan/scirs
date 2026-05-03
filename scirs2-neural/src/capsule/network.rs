//! Complete CapsNet Model
//!
//! Composes `PrimaryCaps`, `DigitCaps`, and `DynamicRouting` into a complete
//! forward-pass capsule network.

use crate::capsule::dynamic_routing::DynamicRouting;
use crate::capsule::layers::{l2_norm, DigitCaps, PrimaryCaps};
use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Capsule value type
// ---------------------------------------------------------------------------

/// A single capsule: an activity vector plus its activation scalar.
///
/// `activation = ||pose||` (L2 norm of the pose vector).
#[derive(Debug, Clone)]
pub struct Capsule {
    /// Pose vector encoding instantiation parameters
    pub pose: Vec<f32>,
    /// Activation probability: ||pose|| ∈ [0, 1)
    pub activation: f32,
}

impl Capsule {
    /// Create a capsule from a raw pose vector.
    pub fn from_pose(pose: Vec<f32>) -> Self {
        let activation = l2_norm(&pose);
        Self { pose, activation }
    }

    /// Return whether the capsule is considered "active" (activation > threshold).
    pub fn is_active(&self, threshold: f32) -> bool {
        self.activation > threshold
    }
}

// ---------------------------------------------------------------------------
// CapsNetConfig
// ---------------------------------------------------------------------------

/// Configuration for a complete capsule network.
#[derive(Debug, Clone)]
pub struct CapsNetConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of primary capsule groups
    pub primary_caps: usize,
    /// Dimension of each primary capsule vector
    pub primary_dim: usize,
    /// Number of class/output capsules (e.g. 10 for MNIST digits)
    pub digit_caps: usize,
    /// Dimension of each output capsule vector
    pub digit_dim: usize,
    /// Number of dynamic routing iterations
    pub routing_iters: usize,
}

impl Default for CapsNetConfig {
    /// Default configuration matching the MNIST experiment from Sabour et al.
    fn default() -> Self {
        Self {
            input_size: 1152, // 32 × 6 × 6 (after two conv layers)
            primary_caps: 32,
            primary_dim: 8,
            digit_caps: 10,
            digit_dim: 16,
            routing_iters: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// CapsNet
// ---------------------------------------------------------------------------

/// Full capsule network (forward-pass only).
///
/// The network is composed of:
/// 1. `PrimaryCaps` — converts flat input features to capsule vectors
/// 2. `DigitCaps`   — transforms primary capsules to class capsules via W_{ij}
/// 3. `DynamicRouting` — computes class capsule outputs via routing-by-agreement
pub struct CapsNet {
    primary: PrimaryCaps,
    digit: DigitCaps,
    routing: DynamicRouting,
    config: CapsNetConfig,
}

impl std::fmt::Debug for CapsNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapsNet")
            .field("config", &self.config)
            .finish()
    }
}

impl CapsNet {
    /// Create a new capsule network from a configuration.
    ///
    /// # Errors
    /// Returns an error if any layer cannot be constructed with the given config.
    pub fn new(config: CapsNetConfig) -> Result<Self> {
        if config.input_size == 0 {
            return Err(NeuralError::InvalidArgument(
                "input_size must be > 0".into(),
            ));
        }
        let primary = PrimaryCaps::new(config.input_size, config.primary_caps, config.primary_dim)?;
        let digit = DigitCaps::new(
            config.digit_caps,
            config.digit_dim,
            config.primary_caps,
            config.primary_dim,
        )?;
        let routing = DynamicRouting::new(config.routing_iters)?;
        Ok(Self {
            primary,
            digit,
            routing,
            config,
        })
    }

    /// Forward pass: input features → class capsules.
    ///
    /// # Arguments
    /// * `x` — flat feature vector of length `input_size`
    ///
    /// # Returns
    /// Vector of `digit_caps` capsules, one per class.
    ///
    /// # Errors
    /// Returns an error if `x.len() != input_size`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<Capsule>> {
        if x.len() != self.config.input_size {
            return Err(NeuralError::DimensionMismatch(format!(
                "CapsNet: input size {} != config.input_size {}",
                x.len(),
                self.config.input_size
            )));
        }

        // Step 1: PrimaryCaps → n_primary capsule vectors each of dim primary_dim
        let primary_out = self.primary.forward(x)?;

        // Step 2: DigitCaps predictions û_{j|i}
        let u_hat = self.digit.compute_predictions(&primary_out)?;

        // Step 3: Dynamic routing → class capsule vectors
        let v = self.routing.route(&u_hat)?;

        // Build Capsule structs
        let capsules = v.into_iter().map(Capsule::from_pose).collect();
        Ok(capsules)
    }

    /// Predict the most likely class (argmax of capsule activations).
    ///
    /// # Errors
    /// Returns an error if `x` has wrong length.
    pub fn predict(&self, x: &[f32]) -> Result<usize> {
        let capsules = self.forward(x)?;
        capsules
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| NeuralError::ComputationError("no capsules produced".into()))
    }

    /// Return the network configuration.
    pub fn config(&self) -> &CapsNetConfig {
        &self.config
    }

    /// Return the number of trainable parameters (weights + biases).
    pub fn n_params(&self) -> usize {
        // PrimaryCaps: weights + bias
        let primary_params =
            self.config.input_size * self.config.primary_caps * self.config.primary_dim
                + self.config.primary_caps * self.config.primary_dim;
        // DigitCaps: n_classes × n_primary × (cap_dim × primary_dim)
        let digit_params = self.config.digit_caps
            * self.config.primary_caps
            * self.config.digit_dim
            * self.config.primary_dim;
        primary_params + digit_params
    }
}

// ---------------------------------------------------------------------------
// CapsNetBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for `CapsNet`.
#[derive(Debug, Default)]
pub struct CapsNetBuilder {
    config: CapsNetConfig,
}

impl CapsNetBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: CapsNetConfig::default(),
        }
    }

    /// Set the input feature size.
    pub fn input_size(mut self, n: usize) -> Self {
        self.config.input_size = n;
        self
    }

    /// Set the number of primary capsule groups.
    pub fn primary_caps(mut self, n: usize) -> Self {
        self.config.primary_caps = n;
        self
    }

    /// Set the primary capsule dimension.
    pub fn primary_dim(mut self, d: usize) -> Self {
        self.config.primary_dim = d;
        self
    }

    /// Set the number of output (digit/class) capsules.
    pub fn digit_caps(mut self, n: usize) -> Self {
        self.config.digit_caps = n;
        self
    }

    /// Set the output capsule dimension.
    pub fn digit_dim(mut self, d: usize) -> Self {
        self.config.digit_dim = d;
        self
    }

    /// Set the number of routing iterations.
    pub fn routing_iters(mut self, n: usize) -> Self {
        self.config.routing_iters = n;
        self
    }

    /// Build the capsule network.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> Result<CapsNet> {
        CapsNet::new(self.config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> CapsNetConfig {
        CapsNetConfig {
            input_size: 16,
            primary_caps: 4,
            primary_dim: 4,
            digit_caps: 5,
            digit_dim: 8,
            routing_iters: 2,
        }
    }

    #[test]
    fn capsnet_forward_shape() {
        let net = CapsNet::new(small_config()).expect("operation should succeed");
        let input = vec![0.1_f32; 16];
        let out = net.forward(&input).expect("operation should succeed");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].pose.len(), 8);
    }

    #[test]
    fn capsnet_activations_bounded() {
        let net = CapsNet::new(small_config()).expect("operation should succeed");
        let input = vec![1.0_f32; 16];
        let out = net.forward(&input).expect("operation should succeed");
        for cap in &out {
            assert!(
                cap.activation <= 1.0 + 1e-5,
                "activation {} should be ≤ 1",
                cap.activation
            );
            assert!(cap.activation >= 0.0);
        }
    }

    #[test]
    fn capsnet_predict_returns_valid_class() {
        let net = CapsNet::new(small_config()).expect("operation should succeed");
        let input = vec![0.5_f32; 16];
        let pred = net.predict(&input).expect("operation should succeed");
        assert!(pred < 5);
    }

    #[test]
    fn capsnet_wrong_input_size() {
        let net = CapsNet::new(small_config()).expect("operation should succeed");
        let bad_input = vec![0.1_f32; 8]; // wrong size
        assert!(net.forward(&bad_input).is_err());
    }

    #[test]
    fn capsnet_builder() {
        let net = CapsNetBuilder::new()
            .input_size(16)
            .primary_caps(4)
            .primary_dim(4)
            .digit_caps(5)
            .digit_dim(8)
            .routing_iters(2)
            .build()
            .expect("operation should succeed");
        assert_eq!(net.config().digit_caps, 5);
    }

    #[test]
    fn capsule_from_pose() {
        let cap = Capsule::from_pose(vec![0.6, 0.8]);
        // L2 norm = 1.0 exactly
        assert!((cap.activation - 1.0).abs() < 1e-5);
        assert!(cap.is_active(0.5));
    }

    #[test]
    fn capsnet_n_params_is_positive() {
        let net = CapsNet::new(small_config()).expect("operation should succeed");
        assert!(net.n_params() > 0);
    }
}
