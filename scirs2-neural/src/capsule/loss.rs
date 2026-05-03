//! Capsule Network Loss Functions
//!
//! Implements the margin loss from Sabour et al. (2017):
//!
//! ```text
//! L_k = T_k · max(0, m⁺ - ||v_k||)²
//!       + λ · (1 - T_k) · max(0, ||v_k|| - m⁻)²
//! ```
//!
//! and a combined loss that sums over all capsule classes.

use crate::capsule::network::Capsule;
use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// MarginLoss
// ---------------------------------------------------------------------------

/// Margin loss for capsule network training.
///
/// Hyperparameters follow the original paper:
/// - m⁺ = 0.9 (presence margin)
/// - m⁻ = 0.1 (absence margin)
/// - λ  = 0.5 (down-weighting factor for absent classes)
#[derive(Debug, Clone)]
pub struct MarginLoss {
    /// Presence margin (default 0.9)
    pub m_plus: f32,
    /// Absence margin (default 0.1)
    pub m_minus: f32,
    /// Down-weighting for absent classes (default 0.5)
    pub lambda: f32,
}

impl Default for MarginLoss {
    fn default() -> Self {
        Self {
            m_plus: 0.9,
            m_minus: 0.1,
            lambda: 0.5,
        }
    }
}

impl MarginLoss {
    /// Create a `MarginLoss` with default paper hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `MarginLoss` with custom hyperparameters.
    ///
    /// # Errors
    /// Returns an error if m_plus ≤ m_minus, lambda < 0, or margins out of (0,1).
    pub fn with_params(m_plus: f32, m_minus: f32, lambda: f32) -> Result<Self> {
        if m_plus <= m_minus {
            return Err(NeuralError::InvalidArgument(
                "m_plus must be > m_minus".into(),
            ));
        }
        if lambda < 0.0 {
            return Err(NeuralError::InvalidArgument("lambda must be ≥ 0".into()));
        }
        if !(0.0..=1.0).contains(&m_plus) || !(0.0..=1.0).contains(&m_minus) {
            return Err(NeuralError::InvalidArgument(
                "margins must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            m_plus,
            m_minus,
            lambda,
        })
    }

    /// Compute the margin loss for a single capsule class.
    ///
    /// # Arguments
    /// * `norm`    — L2 norm of the class capsule vector ||v_k||
    /// * `present` — whether the class is present in the target
    ///
    /// # Returns
    /// Per-class loss value L_k.
    pub fn per_class_loss(&self, norm: f32, present: bool) -> f32 {
        if present {
            // T_k = 1: penalise if norm < m+
            let diff = (self.m_plus - norm).max(0.0);
            diff * diff
        } else {
            // T_k = 0: penalise (with lambda) if norm > m-
            let diff = (norm - self.m_minus).max(0.0);
            self.lambda * diff * diff
        }
    }

    /// Compute the total margin loss over all capsule classes.
    ///
    /// # Arguments
    /// * `capsules` — output capsule vectors (one per class)
    /// * `labels`   — presence flags (same length as capsules)
    ///
    /// # Returns
    /// Scalar total loss.
    ///
    /// # Errors
    /// Returns an error if lengths mismatch.
    pub fn compute(capsules: &[Capsule], labels: &[bool]) -> f32 {
        let loss_fn = Self::default();
        loss_fn.total_loss(capsules, labels).unwrap_or(0.0)
    }

    /// Compute the total margin loss with custom hyperparameters.
    ///
    /// # Errors
    /// Returns an error if `capsules.len() != labels.len()`.
    pub fn total_loss(&self, capsules: &[Capsule], labels: &[bool]) -> Result<f32> {
        if capsules.len() != labels.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "capsules length {} != labels length {}",
                capsules.len(),
                labels.len()
            )));
        }
        let total = capsules
            .iter()
            .zip(labels.iter())
            .map(|(cap, &present)| self.per_class_loss(cap.activation, present))
            .sum();
        Ok(total)
    }

    /// Compute per-class losses (useful for debugging / analysis).
    ///
    /// # Errors
    /// Returns an error on length mismatch.
    pub fn per_class_losses(&self, capsules: &[Capsule], labels: &[bool]) -> Result<Vec<f32>> {
        if capsules.len() != labels.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "capsules length {} != labels length {}",
                capsules.len(),
                labels.len()
            )));
        }
        Ok(capsules
            .iter()
            .zip(labels.iter())
            .map(|(cap, &present)| self.per_class_loss(cap.activation, present))
            .collect())
    }

    /// Predict the class with the highest capsule activation.
    ///
    /// # Errors
    /// Returns an error if `capsules` is empty.
    pub fn predict_class(capsules: &[Capsule]) -> Result<usize> {
        capsules
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| NeuralError::InvalidArgument("capsules is empty".into()))
    }
}

// ---------------------------------------------------------------------------
// SpreadLoss
// ---------------------------------------------------------------------------

/// Spread loss (alternative to margin loss; Hinton et al. 2018).
///
/// Encourages the target capsule activation to be far from non-target
/// activations:
///
/// ```text
/// L = Σ_{k ≠ t} max(0, m - (a_t - a_k))²
/// ```
///
/// The margin `m` grows linearly from `m_min` to `m_max` during training.
#[derive(Debug, Clone)]
pub struct SpreadLoss {
    /// Current margin value
    pub m: f32,
    /// Minimum margin
    pub m_min: f32,
    /// Maximum margin
    pub m_max: f32,
}

impl SpreadLoss {
    /// Create a spread loss with paper defaults (m grows from 0.2 to 0.9).
    pub fn new() -> Self {
        Self {
            m: 0.2,
            m_min: 0.2,
            m_max: 0.9,
        }
    }

    /// Update margin based on training progress.
    ///
    /// # Arguments
    /// * `epoch`       — current epoch (0-indexed)
    /// * `total_epochs` — total training epochs
    pub fn update_margin(&mut self, epoch: usize, total_epochs: usize) {
        let t = epoch as f32 / total_epochs.max(1) as f32;
        self.m = self.m_min + (self.m_max - self.m_min) * t;
    }

    /// Compute spread loss.
    ///
    /// # Arguments
    /// * `activations` — capsule activation norms
    /// * `target`      — index of the target class
    ///
    /// # Errors
    /// Returns an error if target is out of bounds or activations is empty.
    pub fn compute(&self, activations: &[f32], target: usize) -> Result<f32> {
        if activations.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "activations must be non-empty".into(),
            ));
        }
        if target >= activations.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "target {target} out of bounds ({})",
                activations.len()
            )));
        }
        let a_t = activations[target];
        let loss: f32 = activations
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != target)
            .map(|(_, &a_k)| {
                let diff = (self.m - (a_t - a_k)).max(0.0);
                diff * diff
            })
            .sum();
        Ok(loss)
    }
}

impl Default for SpreadLoss {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsule::network::Capsule;

    fn make_caps(norms: &[f32]) -> Vec<Capsule> {
        norms
            .iter()
            .map(|&n| Capsule {
                pose: vec![n, 0.0],
                activation: n,
            })
            .collect()
    }

    #[test]
    fn margin_loss_present_correct() {
        let loss = MarginLoss::new();
        // norm = 0.95 ≥ m+ = 0.9 → loss = 0
        assert_eq!(loss.per_class_loss(0.95, true), 0.0);
    }

    #[test]
    fn margin_loss_absent_correct() {
        let loss = MarginLoss::new();
        // norm = 0.05 ≤ m- = 0.1 → loss = 0
        assert_eq!(loss.per_class_loss(0.05, false), 0.0);
    }

    #[test]
    fn margin_loss_present_wrong() {
        let loss = MarginLoss::new();
        // norm = 0.5 < m+ = 0.9 → loss = (0.9-0.5)^2 = 0.16
        let l = loss.per_class_loss(0.5, true);
        assert!((l - 0.16).abs() < 1e-5, "expected 0.16, got {l}");
    }

    #[test]
    fn margin_loss_total() {
        let caps = make_caps(&[0.95, 0.05, 0.05]);
        let labels = [true, false, false];
        let l = MarginLoss::compute(&caps, &labels);
        assert!(l < 0.01, "All correct → near-zero loss, got {l}");
    }

    #[test]
    fn margin_loss_mismatch() {
        let loss = MarginLoss::new();
        let caps = make_caps(&[0.9, 0.1]);
        assert!(loss.total_loss(&caps, &[true]).is_err());
    }

    #[test]
    fn predict_class_returns_highest() {
        let caps = make_caps(&[0.2, 0.8, 0.5]);
        let pred = MarginLoss::predict_class(&caps).expect("operation should succeed");
        assert_eq!(pred, 1);
    }

    #[test]
    fn spread_loss_zero_for_large_margin() {
        let loss = SpreadLoss::new();
        // target class has activation 0.9, others have 0.1
        let acts = vec![0.1, 0.9, 0.1];
        let l = loss.compute(&acts, 1).expect("operation should succeed");
        // m=0.2, a_t=0.9, a_k=0.1 → m - (0.9-0.1) = 0.2-0.8 < 0 → clamped to 0
        assert_eq!(l, 0.0);
    }

    #[test]
    fn spread_loss_nonzero_when_close() {
        let loss = SpreadLoss::new();
        // target=1 with a_t=0.3, others have a_k=0.25 → m - (0.3-0.25) = 0.2-0.05=0.15
        let acts = vec![0.25, 0.3, 0.25];
        let l = loss.compute(&acts, 1).expect("operation should succeed");
        assert!(
            l > 0.0,
            "Should have positive loss when activations are close"
        );
    }
}
