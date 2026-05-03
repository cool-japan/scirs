//! Sparse training for reduced model size and computation

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::ndarray::ArrayViewMut2;
use std::collections::HashSet;

/// Sparsity schedule — controls how sparsity increases during training
#[derive(Debug, Clone)]
pub enum SparsitySchedule {
    /// Constant sparsity throughout training
    Constant,
    /// Linearly ramp from 0 → target between `start_step` and `end_step`
    Linear { start_step: usize, end_step: usize },
    /// Polynomial ramp: target × progress^power
    Polynomial {
        start_step: usize,
        end_step: usize,
        power: f32,
    },
    /// Exponential ramp: target × (1 − e^(−5 × progress))
    Exponential { start_step: usize, end_step: usize },
}

impl SparsitySchedule {
    /// Current sparsity level for the given training `step`
    pub fn get_sparsity(&self, step: usize, target: f32) -> f32 {
        match self {
            SparsitySchedule::Constant => target,
            SparsitySchedule::Linear {
                start_step,
                end_step,
            } => {
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    target
                } else {
                    let progress =
                        (step - start_step) as f32 / (end_step - start_step).max(1) as f32;
                    target * progress
                }
            }
            SparsitySchedule::Polynomial {
                start_step,
                end_step,
                power,
            } => {
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    target
                } else {
                    let progress =
                        (step - start_step) as f32 / (end_step - start_step).max(1) as f32;
                    target * progress.powf(*power)
                }
            }
            SparsitySchedule::Exponential {
                start_step,
                end_step,
            } => {
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    target
                } else {
                    let progress =
                        (step - start_step) as f32 / (end_step - start_step).max(1) as f32;
                    target * (1.0 - (-5.0 * progress).exp())
                }
            }
        }
    }
}

/// Pruning method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMethod {
    /// Remove weights with smallest absolute value
    Magnitude,
    /// Gradient-based importance (simplified: falls back to magnitude)
    Gradient,
    /// Random pruning
    Random,
    /// Structured channel pruning
    Structured,
}

/// Statistics returned by each pruning call
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of zeroed-out parameters
    pub pruned_params: usize,
    /// Actual sparsity ratio (`pruned / total`)
    pub sparsity: f32,
    /// Whether structured sparsity was applied
    pub structured: bool,
}

/// Sparse training manager
pub struct SparseTrainer {
    target_sparsity: f32,
    schedule: SparsitySchedule,
    /// Pruning method to apply
    pub pruning_method: PruningMethod,
    /// Enable structured (channel-level) sparsity
    pub structured: bool,
    /// Granularity block size for structured pruning
    pub granularity: usize,
}

impl SparseTrainer {
    /// Create a new sparse trainer
    pub fn new(target_sparsity: f32, schedule: SparsitySchedule) -> Self {
        Self {
            target_sparsity,
            schedule,
            pruning_method: PruningMethod::Magnitude,
            structured: false,
            granularity: 1,
        }
    }

    /// Apply sparsity to the mutable weight tensor at the given training `step`
    pub fn apply_sparsity(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        step: usize,
        _layer_name: &str,
    ) -> Result<SparsityStats> {
        let current_sparsity = self.schedule.get_sparsity(step, self.target_sparsity);
        match self.pruning_method {
            PruningMethod::Magnitude => self.magnitude_pruning(weights, current_sparsity),
            PruningMethod::Gradient => self.magnitude_pruning(weights, current_sparsity),
            PruningMethod::Random => self.random_pruning(weights, current_sparsity),
            PruningMethod::Structured => self.structured_pruning(weights, current_sparsity),
        }
    }

    /// Magnitude-based pruning: zero out the `sparsity` fraction of smallest |w|
    pub fn magnitude_pruning(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        sparsity: f32,
    ) -> Result<SparsityStats> {
        let total_params = weights.len();
        let params_to_prune = (total_params as f32 * sparsity) as usize;

        let mut weight_magnitudes: Vec<(f32, (usize, usize))> = weights
            .indexed_iter()
            .map(|((i, j), &w)| (w.abs(), (i, j)))
            .collect();
        weight_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN"));

        let mut pruned_count = 0;
        for (_, (row, col)) in weight_magnitudes.iter().take(params_to_prune) {
            weights[[*row, *col]] = 0.0;
            pruned_count += 1;
        }
        Ok(SparsityStats {
            total_params,
            pruned_params: pruned_count,
            sparsity: pruned_count as f32 / total_params.max(1) as f32,
            structured: false,
        })
    }

    /// Random pruning: zero out a random subset of weights
    pub fn random_pruning(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        sparsity: f32,
    ) -> Result<SparsityStats> {
        let total_params = weights.len();
        let params_to_prune = (total_params as f32 * sparsity) as usize;
        let mut indices: Vec<(usize, usize)> =
            weights.indexed_iter().map(|((i, j), _)| (i, j)).collect();
        // Deterministic shuffle using a simple xorshift for reproducibility
        let mut rng_state: u64 = 0xdeadbeef_cafebabe;
        for i in (1..indices.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
        let mut pruned_params = 0;
        for &(row, col) in indices.iter().take(params_to_prune) {
            weights[[row, col]] = 0.0;
            pruned_params += 1;
        }
        Ok(SparsityStats {
            total_params,
            pruned_params,
            sparsity: pruned_params as f32 / total_params.max(1) as f32,
            structured: false,
        })
    }

    /// Structured pruning: zero out entire columns (channels) with lowest L2 norm
    pub fn structured_pruning(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        sparsity: f32,
    ) -> Result<SparsityStats> {
        let (rows, cols) = weights.dim();
        let channels_to_prune = (cols as f32 * sparsity) as usize;

        let mut channel_importance: Vec<(f32, usize)> = (0..cols)
            .map(|c| {
                let norm = weights.column(c).iter().map(|x| x * x).sum::<f32>().sqrt();
                (norm, c)
            })
            .collect();
        channel_importance.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN"));

        let mut pruned_channels = 0;
        for (_, channel) in channel_importance.iter().take(channels_to_prune) {
            for r in 0..rows {
                weights[[r, *channel]] = 0.0;
            }
            pruned_channels += 1;
        }
        Ok(SparsityStats {
            total_params: weights.len(),
            pruned_params: pruned_channels * rows,
            sparsity: (pruned_channels * rows) as f32 / weights.len().max(1) as f32,
            structured: true,
        })
    }

    /// Compute a binary mask: `true` where `w ≠ 0`
    pub fn get_mask(&self, weights: &ArrayView2<f32>) -> Array2<bool> {
        weights.mapv(|w| w != 0.0)
    }

    /// Zero out gradient entries where the corresponding mask is `false`
    pub fn mask_gradients(
        gradients: &mut ArrayViewMut2<f32>,
        mask: &ArrayView2<bool>,
    ) -> Result<()> {
        gradients.zip_mut_with(mask, |g, &m| {
            if !m {
                *g = 0.0;
            }
        });
        Ok(())
    }
}

// ── Dynamic sparse network ────────────────────────────────────────────────────

/// Growth method for new connections in a dynamic sparse network
#[derive(Debug, Clone, Copy, PartialEq)]
enum GrowthMethod {
    /// Grow connections at random positions
    Random,
    /// Grow connections with the highest gradient magnitude
    Gradient,
}

/// Connection history tracker
struct ConnectionHistory {
    history: Vec<HashSet<(usize, usize)>>,
    max_history: usize,
}

impl ConnectionHistory {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
        }
    }

    fn update(&mut self, weights: &ArrayView2<f32>, _step: usize) {
        let active: HashSet<(usize, usize)> = weights
            .indexed_iter()
            .filter(|(_, &w)| w != 0.0)
            .map(|((i, j), _)| (i, j))
            .collect();
        self.history.push(active);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
}

/// Dynamic sparse network: prunes and regrows connections each update step
pub struct DynamicSparseNetwork {
    prune_grow_ratio: f32,
    growth_method: GrowthMethod,
    connection_history: ConnectionHistory,
}

impl DynamicSparseNetwork {
    /// Create a new dynamic sparse network
    pub fn new(prune_grow_ratio: f32) -> Self {
        Self {
            prune_grow_ratio,
            growth_method: GrowthMethod::Gradient,
            connection_history: ConnectionHistory::new(),
        }
    }

    /// Update connections: prune the weakest and grow new ones
    pub fn update_connections(
        &mut self,
        weights: &mut ArrayViewMut2<f32>,
        gradients: &ArrayView2<f32>,
        step: usize,
    ) -> Result<()> {
        let num_connections = weights.iter().filter(|&&w| w != 0.0).count();
        let num_to_update = (num_connections as f32 * self.prune_grow_ratio) as usize;
        self.prune_connections(weights, num_to_update)?;
        self.grow_connections(weights, gradients, num_to_update)?;
        self.connection_history.update(&weights.view(), step);
        Ok(())
    }

    fn prune_connections(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        num_to_prune: usize,
    ) -> Result<()> {
        let mut active_weights: Vec<(f32, (usize, usize))> = weights
            .indexed_iter()
            .filter(|(_, &w)| w != 0.0)
            .map(|((i, j), &w)| (w.abs(), (i, j)))
            .collect();
        active_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN"));
        for (_, (row, col)) in active_weights.iter().take(num_to_prune) {
            weights[[*row, *col]] = 0.0;
        }
        Ok(())
    }

    fn grow_connections(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        gradients: &ArrayView2<f32>,
        num_to_grow: usize,
    ) -> Result<()> {
        match self.growth_method {
            GrowthMethod::Random => self.random_growth(weights, num_to_grow),
            GrowthMethod::Gradient => self.gradient_based_growth(weights, gradients, num_to_grow),
        }
    }

    fn random_growth(&self, weights: &mut ArrayViewMut2<f32>, num_to_grow: usize) -> Result<()> {
        let mut zero_indices: Vec<(usize, usize)> = weights
            .indexed_iter()
            .filter(|(_, &w)| w == 0.0)
            .map(|((i, j), _)| (i, j))
            .collect();
        // Deterministic shuffle
        let mut rng_state: u64 = 0xcafe_babe_dead_beef;
        for i in (1..zero_indices.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            zero_indices.swap(i, j);
        }
        for &(row, col) in zero_indices.iter().take(num_to_grow) {
            // Small random initialisation using LCG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let small = (rng_state >> 33) as f32 / u32::MAX as f32 * 0.001;
            weights[[row, col]] = small;
        }
        Ok(())
    }

    fn gradient_based_growth(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        gradients: &ArrayView2<f32>,
        num_to_grow: usize,
    ) -> Result<()> {
        let mut gradient_magnitudes: Vec<(f32, (usize, usize))> = weights
            .indexed_iter()
            .filter(|(_, &w)| w == 0.0)
            .map(|((i, j), _)| (gradients[[i, j]].abs(), (i, j)))
            .collect();
        gradient_magnitudes.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("non-NaN"));
        for (_, (row, col)) in gradient_magnitudes.iter().take(num_to_grow) {
            weights[[*row, *col]] = 0.001 * gradients[[*row, *col]].signum();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude_pruning() {
        let trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        let mut weights = Array2::from_shape_vec((2, 3), vec![0.1, -0.5, 0.2, -0.3, 0.4, -0.6])
            .expect("shape ok");
        let stats = trainer
            .magnitude_pruning(&mut weights.view_mut(), 0.5)
            .expect("pruning ok");
        assert_eq!(stats.pruned_params, 3);
        assert!((stats.sparsity - 0.5).abs() < 0.01);
        // Smallest magnitude values should be zeroed
        assert_eq!(weights[[0, 0]], 0.0); // |0.1|
        assert_eq!(weights[[0, 2]], 0.0); // |0.2|
        assert_eq!(weights[[1, 0]], 0.0); // |-0.3|
    }

    #[test]
    fn test_sparsity_schedule_linear() {
        let linear = SparsitySchedule::Linear {
            start_step: 0,
            end_step: 100,
        };
        assert_eq!(linear.get_sparsity(0, 0.9), 0.0);
        assert!((linear.get_sparsity(50, 0.9) - 0.45).abs() < 1e-5);
        assert_eq!(linear.get_sparsity(100, 0.9), 0.9);
        assert_eq!(linear.get_sparsity(150, 0.9), 0.9);
    }

    #[test]
    fn test_sparsity_schedule_constant() {
        let schedule = SparsitySchedule::Constant;
        assert_eq!(schedule.get_sparsity(0, 0.5), 0.5);
        assert_eq!(schedule.get_sparsity(1000, 0.5), 0.5);
    }

    #[test]
    fn test_sparsity_schedule_polynomial() {
        let poly = SparsitySchedule::Polynomial {
            start_step: 0,
            end_step: 100,
            power: 2.0,
        };
        assert_eq!(poly.get_sparsity(0, 1.0), 0.0);
        // At step 50: progress = 0.5, sparsity = 1.0 * 0.5^2 = 0.25
        assert!((poly.get_sparsity(50, 1.0) - 0.25).abs() < 1e-4);
        assert_eq!(poly.get_sparsity(100, 1.0), 1.0);
    }

    #[test]
    fn test_structured_pruning() {
        let trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        let mut weights = Array2::from_shape_vec(
            (3, 4),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        )
        .expect("shape ok");
        let stats = trainer
            .structured_pruning(&mut weights.view_mut(), 0.5)
            .expect("pruning ok");
        assert!(stats.structured);
        // Two channels (cols 0 and 1) should be zeroed
        assert_eq!(weights.column(0).sum(), 0.0);
        assert_eq!(weights.column(1).sum(), 0.0);
    }

    #[test]
    fn test_get_mask() {
        let trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        let weights =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, -1.0, 0.0, 2.0]).expect("shape ok");
        let mask = trainer.get_mask(&weights.view());
        assert!(!mask[[0, 0]]);
        assert!(mask[[0, 1]]);
        assert!(mask[[1, 3 % 3]]); // [[1,0]] = -1.0 → true
    }

    #[test]
    fn test_mask_gradients() {
        let weights =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, -1.0, 0.0, 2.0]).expect("shape ok");
        let trainer = SparseTrainer::new(0.0, SparsitySchedule::Constant);
        let mask = trainer.get_mask(&weights.view());
        let mut grads = Array2::ones((2, 3));
        SparseTrainer::mask_gradients(&mut grads.view_mut(), &mask.view()).expect("mask ok");
        assert_eq!(grads[[0, 0]], 0.0); // was zero → masked
        assert_eq!(grads[[0, 1]], 1.0); // was non-zero → kept
    }

    #[test]
    fn test_dynamic_sparse_network() {
        let mut dsn = DynamicSparseNetwork::new(0.1);
        // Initialize with some sparse weights
        let mut weights = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 0.0, 0.4,
            ],
        )
        .expect("shape ok");
        let gradients = Array2::ones((4, 4)) * 0.1f32;
        dsn.update_connections(&mut weights.view_mut(), &gradients.view(), 0)
            .expect("update ok");
        // After update, some zero weights should be grown
        let non_zero_count = weights.iter().filter(|&&w| w != 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_random_pruning() {
        let trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        let mut weights = Array2::ones((4, 4));
        let stats = trainer
            .random_pruning(&mut weights.view_mut(), 0.5)
            .expect("random pruning ok");
        assert_eq!(stats.pruned_params, 8);
        assert!((stats.sparsity - 0.5).abs() < 1e-5);
    }
}
