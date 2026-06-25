//! Continual / lifelong learning for neural networks
//!
//! This module provides algorithms that allow a neural network to learn new
//! tasks sequentially without catastrophically forgetting previous ones.
//!
//! ## Algorithms
//!
//! | Name | Strategy | Paper |
//! |------|----------|-------|
//! | `ElasticWeightConsolidation` | Regularisation | Kirkpatrick et al. 2017 |
//! | `PackNet` | Pruning/packing | Mallya & Lazebnik 2018 |
//! | `ProgressiveNeuralNetwork` | Growing architecture | Rusu et al. 2016 |
//! | `RehearsalBuffer` | Experience replay | various |
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::training::continual::{EwcConfig, ewc_penalty};
//! use std::collections::HashMap;
//!
//! let params: HashMap<String, Vec<f64>> = HashMap::from([
//!     ("w1".to_string(), vec![0.5_f64, -0.3, 0.1]),
//! ]);
//! let anchors: HashMap<String, Vec<f64>> = HashMap::from([
//!     ("w1".to_string(), vec![0.5_f64, -0.3, 0.1]),
//! ]);
//! let fisher: HashMap<String, Vec<f64>> = HashMap::from([
//!     ("w1".to_string(), vec![1.0_f64, 1.0, 1.0]),
//! ]);
//! let cfg = EwcConfig { lambda: 100.0 };
//! let penalty = ewc_penalty(&params, &anchors, &fisher, &cfg)
//!     .expect("ewc penalty");
//! assert!(penalty.abs() < 1e-10, "identical params → zero penalty");
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────────────────────
// EWC — Elastic Weight Consolidation (Kirkpatrick et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Elastic Weight Consolidation.
#[derive(Debug, Clone)]
pub struct EwcConfig {
    /// Regularisation strength `λ`.  Larger values = stronger protection of
    /// old task knowledge.  Typical range: 100–10 000.
    pub lambda: f64,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self { lambda: 5000.0 }
    }
}

impl EwcConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "EWC lambda must be non-negative and finite, got {}",
                self.lambda
            )));
        }
        Ok(())
    }
}

/// EWC regularisation penalty.
///
/// Computes:
/// ```text
/// L_EWC = (λ/2) * Σ_k  F_k * (θ_k - θ*_k)²
/// ```
/// where `F_k` is the Fisher importance of parameter `k` and `θ*_k` is the
/// anchor (post-task-A optimum).
///
/// # Parameters
/// - `params`: Current model parameters (name → flat values).
/// - `anchors`: Parameters frozen after training on the previous task.
/// - `fisher`: Diagonal Fisher information (name → per-element values).
/// - `config`: EWC hyperparameters.
pub fn ewc_penalty(
    params: &HashMap<String, Vec<f64>>,
    anchors: &HashMap<String, Vec<f64>>,
    fisher: &HashMap<String, Vec<f64>>,
    config: &EwcConfig,
) -> Result<f64> {
    config.validate()?;

    let mut penalty = 0.0_f64;

    for (name, current) in params {
        let anchor = anchors.get(name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "anchor not found for parameter '{name}'"
            ))
        })?;
        let f = fisher.get(name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "Fisher information not found for parameter '{name}'"
            ))
        })?;

        if current.len() != anchor.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "parameter '{name}': current length {} != anchor length {}",
                current.len(),
                anchor.len()
            )));
        }
        if current.len() != f.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "parameter '{name}': param length {} != Fisher length {}",
                current.len(),
                f.len()
            )));
        }

        for ((&theta, &theta_star), &fi) in
            current.iter().zip(anchor.iter()).zip(f.iter())
        {
            let diff = theta - theta_star;
            penalty += fi * diff * diff;
        }
    }

    Ok(0.5 * config.lambda * penalty)
}

/// Gradient of the EWC penalty w.r.t. the current parameters.
///
/// Returns the same structure as `params` with gradient values:
/// ```text
/// dL_EWC / dθ_k = λ * F_k * (θ_k - θ*_k)
/// ```
pub fn ewc_gradient(
    params: &HashMap<String, Vec<f64>>,
    anchors: &HashMap<String, Vec<f64>>,
    fisher: &HashMap<String, Vec<f64>>,
    config: &EwcConfig,
) -> Result<HashMap<String, Vec<f64>>> {
    config.validate()?;
    let mut grads: HashMap<String, Vec<f64>> = HashMap::new();

    for (name, current) in params {
        let anchor = anchors.get(name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "anchor not found for parameter '{name}'"
            ))
        })?;
        let f = fisher.get(name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "Fisher not found for parameter '{name}'"
            ))
        })?;

        if current.len() != anchor.len() || current.len() != f.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "parameter '{name}': shape mismatch (current={}, anchor={}, fisher={})",
                current.len(),
                anchor.len(),
                f.len()
            )));
        }

        let g: Vec<f64> = current
            .iter()
            .zip(anchor.iter())
            .zip(f.iter())
            .map(|((&theta, &theta_star), &fi)| config.lambda * fi * (theta - theta_star))
            .collect();
        grads.insert(name.clone(), g);
    }

    Ok(grads)
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Fisher information estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the diagonal of the Fisher information matrix via the empirical
/// (squared-gradient) approximation.
///
/// Given a set of per-sample squared gradients (one `HashMap<name, Vec<f64>>`
/// per sample), the diagonal Fisher estimate is:
/// ```text
/// F_k ≈ (1/N) * Σ_n  (∂ log p(y_n | x_n, θ) / ∂ θ_k)²
/// ```
///
/// # Parameters
/// - `squared_gradients`: One gradient-squared map per training sample.
///   Each entry maps a parameter name to its element-wise squared gradient.
///
/// # Returns
/// Averaged diagonal Fisher estimate.
pub fn compute_fisher_information(
    squared_gradients: &[HashMap<String, Vec<f64>>],
) -> Result<HashMap<String, Vec<f64>>> {
    if squared_gradients.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "squared_gradients must not be empty".to_string(),
        ));
    }

    let n = squared_gradients.len() as f64;
    let mut accumulator: HashMap<String, Vec<f64>> = HashMap::new();

    for sample_grads in squared_gradients {
        for (name, sq_g) in sample_grads {
            let acc = accumulator
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; sq_g.len()]);

            if acc.len() != sq_g.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Fisher accumulation: parameter '{name}' has inconsistent lengths"
                )));
            }
            for (a, &v) in acc.iter_mut().zip(sq_g.iter()) {
                *a += v;
            }
        }
    }

    // Divide by N
    for values in accumulator.values_mut() {
        for v in values.iter_mut() {
            *v /= n;
        }
    }

    Ok(accumulator)
}

// ─────────────────────────────────────────────────────────────────────────────
// ElasticWeightConsolidation struct
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful EWC manager that accumulates Fisher information over tasks.
///
/// After finishing a task, call [`ElasticWeightConsolidation::consolidate`] to
/// snapshot the current parameters and compute (or update) the Fisher diagonal.
/// During training on the next task, call [`ElasticWeightConsolidation::penalty`]
/// to obtain the regularisation loss.
#[derive(Debug, Clone)]
pub struct ElasticWeightConsolidation {
    /// EWC configuration (λ).
    pub config: EwcConfig,
    /// Cumulative Fisher diagonal (summed over all consolidated tasks).
    fisher: HashMap<String, Vec<f64>>,
    /// Consolidated parameter anchors (last consolidation point).
    anchors: HashMap<String, Vec<f64>>,
    /// Number of tasks consolidated so far.
    pub num_tasks: usize,
}

impl ElasticWeightConsolidation {
    /// Create a new EWC manager.
    pub fn new(config: EwcConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            fisher: HashMap::new(),
            anchors: HashMap::new(),
            num_tasks: 0,
        })
    }

    /// Consolidate after completing a task.
    ///
    /// - `params`: Current (converged) parameters.
    /// - `new_fisher`: Newly computed diagonal Fisher for this task.
    ///
    /// The Fisher is added cumulatively; the anchors are updated to the
    /// current parameter values.
    pub fn consolidate(
        &mut self,
        params: &HashMap<String, Vec<f64>>,
        new_fisher: &HashMap<String, Vec<f64>>,
    ) -> Result<()> {
        // Update anchors
        self.anchors = params.clone();

        // Accumulate Fisher (online EWC: sum over tasks)
        for (name, new_f) in new_fisher {
            let acc = self
                .fisher
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; new_f.len()]);
            if acc.len() != new_f.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Fisher accumulation: '{name}' shape mismatch"
                )));
            }
            for (a, &v) in acc.iter_mut().zip(new_f.iter()) {
                *a += v;
            }
        }

        self.num_tasks += 1;
        Ok(())
    }

    /// Compute the EWC penalty for the given current parameters.
    pub fn penalty(&self, params: &HashMap<String, Vec<f64>>) -> Result<f64> {
        if self.num_tasks == 0 {
            return Ok(0.0);
        }
        ewc_penalty(params, &self.anchors, &self.fisher, &self.config)
    }

    /// Compute the EWC gradient for the given current parameters.
    pub fn gradient(
        &self,
        params: &HashMap<String, Vec<f64>>,
    ) -> Result<HashMap<String, Vec<f64>>> {
        if self.num_tasks == 0 {
            // No tasks have been consolidated yet, so the EWC penalty is exactly
            // zero and therefore so is its gradient. This zero is the correct
            // analytic value, not a placeholder.
            return Ok(params
                .iter()
                .map(|(k, v)| (k.clone(), vec![0.0; v.len()]))
                .collect());
        }
        ewc_gradient(params, &self.anchors, &self.fisher, &self.config)
    }

    /// Whether any tasks have been consolidated.
    #[inline]
    pub fn has_consolidated(&self) -> bool {
        self.num_tasks > 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PackNet — Task Pruning / Capacity Allocation (Mallya & Lazebnik 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// PackNet: progressively prune and "pack" task-specific sub-networks.
///
/// ## Procedure per new task
/// 1. Train the *free* (unpruned) portion of the network on the new task.
/// 2. Prune the least-important free weights (by magnitude).  These become
///    the task mask for this task.
/// 3. Freeze the pruned weights; the remaining free weights are available for
///    future tasks.
///
/// This implementation manages the binary per-parameter allocation masks and
/// provides utilities for pruning and mask retrieval.
#[derive(Debug, Clone)]
pub struct PackNet {
    /// Number of tasks packed so far.
    pub num_tasks: usize,
    /// Per-parameter allocation: `None` = free, `Some(task_id)` = allocated to that task.
    allocation: HashMap<String, Vec<Option<usize>>>,
    /// Fraction of free parameters to prune after each task (e.g. 0.5 = 50%).
    pub prune_fraction: f64,
}

impl PackNet {
    /// Create a new PackNet manager.
    ///
    /// # Parameters
    /// - `prune_fraction`: Fraction of currently free parameters to prune after
    ///   each task.  Must be in `(0, 1)`.
    pub fn new(prune_fraction: f64) -> Result<Self> {
        if prune_fraction <= 0.0 || prune_fraction >= 1.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "prune_fraction must be in (0, 1), got {prune_fraction}"
            )));
        }
        Ok(Self {
            num_tasks: 0,
            allocation: HashMap::new(),
            prune_fraction,
        })
    }

    /// Initialise the allocation structure from model parameter shapes.
    ///
    /// # Parameters
    /// - `param_shapes`: Maps each parameter name to its total number of elements.
    pub fn init_params(&mut self, param_shapes: &HashMap<String, usize>) {
        for (name, &n) in param_shapes {
            self.allocation
                .entry(name.clone())
                .or_insert_with(|| vec![None; n]);
        }
    }

    /// Prune the top-`prune_fraction` free weights by magnitude, allocating
    /// them to the current task, then freeze them.
    ///
    /// # Parameters
    /// - `params`: Current parameter values (name → flat values).
    ///
    /// # Returns
    /// A mask per parameter: `true` = kept free (can be updated), `false` = frozen.
    pub fn prune_for_task(
        &mut self,
        params: &HashMap<String, Vec<f64>>,
    ) -> Result<HashMap<String, Vec<bool>>> {
        let task_id = self.num_tasks;

        // Collect all free (unallocated) weights with their absolute values
        struct WeightRef {
            name_idx: usize,
            elem_idx: usize,
            abs_val: f64,
        }
        let names: Vec<&String> = params.keys().collect();

        let mut free_weights: Vec<WeightRef> = Vec::new();
        for (name_idx, name) in names.iter().enumerate() {
            let vals = match params.get(*name) {
                Some(v) => v,
                None => continue,
            };
            let alloc = self.allocation.entry((*name).clone()).or_insert_with(|| {
                vec![None; vals.len()]
            });
            for (elem_idx, (&v, slot)) in vals.iter().zip(alloc.iter()).enumerate() {
                if slot.is_none() {
                    free_weights.push(WeightRef {
                        name_idx,
                        elem_idx,
                        abs_val: v.abs(),
                    });
                }
            }
        }

        let total_free = free_weights.len();
        let n_prune = ((total_free as f64) * self.prune_fraction).round() as usize;

        // Sort by absolute value descending; prune the top-n_prune (highest magnitude)
        free_weights
            .sort_unstable_by(|a, b| b.abs_val.partial_cmp(&a.abs_val).unwrap_or(std::cmp::Ordering::Equal));

        for wr in free_weights.iter().take(n_prune) {
            let name = names[wr.name_idx];
            if let Some(alloc) = self.allocation.get_mut(name) {
                alloc[wr.elem_idx] = Some(task_id);
            }
        }

        self.num_tasks += 1;

        // Build free mask: true = free (can be updated)
        let mut masks: HashMap<String, Vec<bool>> = HashMap::new();
        for (name, alloc) in &self.allocation {
            if params.contains_key(name) {
                masks.insert(
                    name.clone(),
                    alloc.iter().map(|slot| slot.is_none()).collect(),
                );
            }
        }
        Ok(masks)
    }

    /// Get the binary mask for a specific task (true = belongs to that task).
    pub fn task_mask(&self, task_id: usize, param_name: &str) -> Option<Vec<bool>> {
        self.allocation.get(param_name).map(|alloc| {
            alloc.iter().map(|slot| *slot == Some(task_id)).collect()
        })
    }

    /// Number of free (unallocated) parameters for a given parameter tensor.
    pub fn free_count(&self, param_name: &str) -> usize {
        self.allocation
            .get(param_name)
            .map(|alloc| alloc.iter().filter(|s| s.is_none()).count())
            .unwrap_or(0)
    }

    /// Total parameter count for a given parameter tensor.
    pub fn total_count(&self, param_name: &str) -> usize {
        self.allocation
            .get(param_name)
            .map(|alloc| alloc.len())
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Progressive Neural Networks (Rusu et al. 2016)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single column in a Progressive Neural Network.
#[derive(Debug, Clone)]
pub struct PnnColumnConfig {
    /// Identifier for the column (one per task).
    pub task_id: String,
    /// Layer widths of this column (excluding the input dimension).
    pub layer_widths: Vec<usize>,
    /// Input dimensionality (same for all columns).
    pub input_dim: usize,
}

impl PnnColumnConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.task_id.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "PNN column task_id must not be empty".to_string(),
            ));
        }
        if self.layer_widths.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "PNN column must have at least one layer".to_string(),
            ));
        }
        if self.input_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "PNN column input_dim must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Progressive Neural Network manager (Rusu et al., "Progressive Neural Networks", 2016).
///
/// A PNN adds a new column (sub-network) for each new task.  Previous columns are
/// **frozen**; the new column receives lateral connections from all previous columns
/// at every layer via learned adapters, enabling knowledge transfer without
/// catastrophic forgetting.
///
/// This implementation tracks column configurations and lateral-connection
/// weight shapes; the actual weight tensors are left to the user's parameter
/// management system (e.g. a HashMap of named parameters).
#[derive(Debug, Clone)]
pub struct ProgressiveNeuralNetwork {
    /// Column configurations, one per task.
    pub columns: Vec<PnnColumnConfig>,
}

impl ProgressiveNeuralNetwork {
    /// Create an empty PNN (no columns yet).
    pub fn new() -> Self {
        Self { columns: Vec::new() }
    }

    /// Number of columns (tasks) currently in the network.
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Add a new column for a new task.
    ///
    /// Returns the index of the new column.
    pub fn add_column(&mut self, config: PnnColumnConfig) -> Result<usize> {
        config.validate()?;
        let idx = self.columns.len();
        self.columns.push(config);
        Ok(idx)
    }

    /// Compute the shapes of lateral connection adapters needed when the
    /// column at `new_col_idx` connects to a previous column at `prev_col_idx`.
    ///
    /// For each layer `l`, the lateral connection adapter maps:
    /// `prev_column.layer_widths[l]` → `new_column.layer_widths[l]`
    ///
    /// # Returns
    /// A `Vec` of `(rows, cols)` tuples, one per shared layer depth.
    pub fn lateral_adapter_shapes(
        &self,
        prev_col_idx: usize,
        new_col_idx: usize,
    ) -> Result<Vec<(usize, usize)>> {
        if prev_col_idx >= self.columns.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "prev_col_idx {prev_col_idx} out of range (have {} columns)",
                self.columns.len()
            )));
        }
        if new_col_idx >= self.columns.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "new_col_idx {new_col_idx} out of range (have {} columns)",
                self.columns.len()
            )));
        }
        if prev_col_idx >= new_col_idx {
            return Err(NeuralError::InvalidArgument(
                "prev_col_idx must be < new_col_idx".to_string(),
            ));
        }

        let prev = &self.columns[prev_col_idx];
        let new = &self.columns[new_col_idx];
        let depth = prev.layer_widths.len().min(new.layer_widths.len());

        let shapes = (0..depth)
            .map(|l| (prev.layer_widths[l], new.layer_widths[l]))
            .collect();
        Ok(shapes)
    }

    /// Return all lateral adapter shapes for the newest column
    /// (from all previous columns).
    pub fn all_lateral_shapes_for_new_column(
        &self,
    ) -> Result<Vec<(usize, Vec<(usize, usize)>)>> {
        let new_idx = self.columns.len().checked_sub(1).ok_or_else(|| {
            NeuralError::InvalidState("PNN has no columns yet".to_string())
        })?;

        let mut result = Vec::new();
        for prev_idx in 0..new_idx {
            let shapes = self.lateral_adapter_shapes(prev_idx, new_idx)?;
            result.push((prev_idx, shapes));
        }
        Ok(result)
    }
}

impl Default for ProgressiveNeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rehearsal Buffer — Experience Replay
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for managing which samples are kept in the rehearsal buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum RehearsalStrategy {
    /// Keep a fixed random subset of samples per class (reservoir sampling).
    Reservoir,
    /// Keep the samples with the highest predicted confidence (herding).
    Herding,
    /// Round-robin: replace oldest sample from the same class.
    RingBuffer,
}

/// A named sample stored in the rehearsal buffer.
#[derive(Debug, Clone)]
pub struct RehearsalSample {
    /// Class / task label.
    pub label: usize,
    /// Flat feature vector.
    pub features: Vec<f64>,
    /// Optional soft-target (logits from when the sample was stored).
    pub soft_targets: Option<Vec<f64>>,
    /// Step at which the sample was inserted (for ring-buffer eviction).
    pub insertion_step: u64,
}

/// Experience replay buffer for continual learning.
///
/// Maintains a bounded memory of past samples so they can be interleaved with
/// new-task data during training.
#[derive(Debug, Clone)]
pub struct RehearsalBuffer {
    /// Maximum total samples in the buffer.
    pub capacity: usize,
    /// Replacement strategy.
    pub strategy: RehearsalStrategy,
    /// Stored samples.
    samples: Vec<RehearsalSample>,
    /// Total samples that have ever been added (for reservoir sampling).
    total_seen: u64,
    /// Global insertion step counter.
    step: u64,
}

impl RehearsalBuffer {
    /// Create a new `RehearsalBuffer`.
    pub fn new(capacity: usize, strategy: RehearsalStrategy) -> Result<Self> {
        if capacity == 0 {
            return Err(NeuralError::InvalidArgument(
                "RehearsalBuffer capacity must be > 0".to_string(),
            ));
        }
        Ok(Self {
            capacity,
            strategy,
            samples: Vec::with_capacity(capacity),
            total_seen: 0,
            step: 0,
        })
    }

    /// Add a sample to the buffer.
    ///
    /// The eviction policy is determined by `self.strategy`.
    pub fn add(&mut self, sample: RehearsalSample) {
        self.total_seen += 1;
        self.step += 1;

        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return;
        }

        match self.strategy {
            RehearsalStrategy::Reservoir => {
                // Reservoir sampling: replace position j uniformly at random
                // where j < total_seen
                let j = (simple_hash(self.total_seen) % self.total_seen) as usize;
                if j < self.capacity {
                    self.samples[j] = sample;
                }
            }
            RehearsalStrategy::RingBuffer => {
                // Replace the oldest sample
                let oldest_idx = self
                    .samples
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.insertion_step)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.samples[oldest_idx] = sample;
            }
            RehearsalStrategy::Herding => {
                // Replace the sample with the smallest L2 norm
                // (proxy for "least representative" when no explicit confidence)
                let smallest_idx = self
                    .samples
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let na: f64 = a.features.iter().map(|&x| x * x).sum::<f64>().sqrt();
                        let nb: f64 = b.features.iter().map(|&x| x * x).sum::<f64>().sqrt();
                        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.samples[smallest_idx] = sample;
            }
        }
    }

    /// Retrieve all stored samples.
    pub fn samples(&self) -> &[RehearsalSample] {
        &self.samples
    }

    /// Number of samples currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Retrieve samples for a specific label.
    pub fn samples_for_label(&self, label: usize) -> Vec<&RehearsalSample> {
        self.samples
            .iter()
            .filter(|s| s.label == label)
            .collect()
    }

    /// Sample up to `n` samples from the buffer (round-robin over classes).
    ///
    /// The returned indices are selected deterministically (for reproducibility).
    pub fn sample_batch(&self, n: usize) -> Vec<&RehearsalSample> {
        if self.samples.is_empty() || n == 0 {
            return Vec::new();
        }
        let take = n.min(self.samples.len());
        // Simple pseudo-random subset without external RNG
        let step = (self.samples.len() / take).max(1);
        self.samples.iter().step_by(step).take(take).collect()
    }

    /// Clear all stored samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.total_seen = 0;
        self.step = 0;
    }
}

/// Convenience function — create a `RehearsalBuffer` and populate it.
///
/// # Parameters
/// - `capacity`: Maximum buffer size.
/// - `strategy`: Eviction strategy.
/// - `initial_samples`: Samples to add immediately.
pub fn rehearsal_buffer(
    capacity: usize,
    strategy: RehearsalStrategy,
    initial_samples: Vec<RehearsalSample>,
) -> Result<RehearsalBuffer> {
    let mut buf = RehearsalBuffer::new(capacity, strategy)?;
    for s in initial_samples {
        buf.add(s);
    }
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal deterministic hash for use in reservoir sampling (no external RNG).
#[inline]
fn simple_hash(v: u64) -> u64 {
    // Splitmix64 step
    let mut x = v.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_params() -> HashMap<String, Vec<f64>> {
        HashMap::from([
            ("w1".to_string(), vec![0.5_f64, -0.3, 0.1]),
            ("w2".to_string(), vec![1.2_f64, 0.0]),
        ])
    }

    fn sample_fisher() -> HashMap<String, Vec<f64>> {
        HashMap::from([
            ("w1".to_string(), vec![1.0_f64, 2.0, 0.5]),
            ("w2".to_string(), vec![3.0_f64, 1.0]),
        ])
    }

    // ── EWC functions ─────────────────────────────────────────────────────

    #[test]
    fn test_ewc_penalty_zero_for_identical_params() {
        let params = sample_params();
        let anchors = params.clone();
        let fisher = sample_fisher();
        let cfg = EwcConfig { lambda: 100.0 };
        let penalty = ewc_penalty(&params, &anchors, &fisher, &cfg).expect("ewc");
        assert!(
            penalty.abs() < 1e-10,
            "identical params → penalty should be 0, got {penalty}"
        );
    }

    #[test]
    fn test_ewc_penalty_positive_for_different_params() {
        let params = sample_params();
        let mut anchors = sample_params();
        // Shift anchors
        for v in anchors.get_mut("w1").expect("w1") {
            *v += 0.5;
        }
        let fisher = sample_fisher();
        let cfg = EwcConfig { lambda: 100.0 };
        let penalty = ewc_penalty(&params, &anchors, &fisher, &cfg).expect("ewc");
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_ewc_gradient_zero_for_identical_params() {
        let params = sample_params();
        let anchors = params.clone();
        let fisher = sample_fisher();
        let cfg = EwcConfig { lambda: 100.0 };
        let grads = ewc_gradient(&params, &anchors, &fisher, &cfg).expect("ewc grad");
        for g_vec in grads.values() {
            for &g in g_vec {
                assert!(g.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ewc_missing_anchor() {
        let params = sample_params();
        let anchors: HashMap<String, Vec<f64>> = HashMap::new();
        let fisher = sample_fisher();
        let cfg = EwcConfig::default();
        assert!(ewc_penalty(&params, &anchors, &fisher, &cfg).is_err());
    }

    // ── compute_fisher_information ─────────────────────────────────────────

    #[test]
    fn test_fisher_information_average() {
        let sq_grads = vec![
            HashMap::from([("w".to_string(), vec![4.0_f64, 0.0])]),
            HashMap::from([("w".to_string(), vec![0.0_f64, 2.0])]),
        ];
        let fisher = compute_fisher_information(&sq_grads).expect("fisher");
        let w = fisher.get("w").expect("w fisher");
        assert!((w[0] - 2.0).abs() < 1e-10);
        assert!((w[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_information_empty_error() {
        assert!(compute_fisher_information(&[]).is_err());
    }

    // ── ElasticWeightConsolidation struct ─────────────────────────────────

    #[test]
    fn test_ewc_struct_no_penalty_before_consolidation() {
        let ewc = ElasticWeightConsolidation::new(EwcConfig::default()).expect("ewc init");
        let params = sample_params();
        let penalty = ewc.penalty(&params).expect("penalty");
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_ewc_struct_consolidate_and_penalise() {
        let mut ewc =
            ElasticWeightConsolidation::new(EwcConfig { lambda: 1000.0 }).expect("ewc init");
        let params = sample_params();
        let fisher = sample_fisher();
        ewc.consolidate(&params, &fisher).expect("consolidate");

        // Same params → zero penalty
        assert!(ewc.penalty(&params).expect("penalty 0") < 1e-10);

        // Different params → positive penalty
        let mut shifted = sample_params();
        for v in shifted.get_mut("w1").expect("w1") {
            *v += 0.3;
        }
        assert!(ewc.penalty(&shifted).expect("penalty shifted") > 0.0);
    }

    // ── PackNet ───────────────────────────────────────────────────────────

    #[test]
    fn test_packnet_prune_reduces_free_count() {
        let mut pn = PackNet::new(0.5).expect("packnet");
        let mut param_shapes = HashMap::new();
        param_shapes.insert("w".to_string(), 10usize);
        pn.init_params(&param_shapes);

        let params = HashMap::from([("w".to_string(), (0..10).map(|i| i as f64).collect::<Vec<_>>())]);
        let _ = pn.prune_for_task(&params).expect("prune t0");
        assert_eq!(pn.free_count("w"), 5, "50% pruned → 5 free out of 10");
        assert_eq!(pn.total_count("w"), 10);
    }

    #[test]
    fn test_packnet_task_mask() {
        let mut pn = PackNet::new(0.5).expect("packnet");
        let param_shapes = HashMap::from([("w".to_string(), 4usize)]);
        pn.init_params(&param_shapes);
        let params = HashMap::from([("w".to_string(), vec![3.0, 2.0, 1.0, 0.5])]);
        let _ = pn.prune_for_task(&params).expect("prune");
        let mask = pn.task_mask(0, "w").expect("mask");
        let active: usize = mask.iter().filter(|&&v| v).count();
        assert_eq!(active, 2);
    }

    // ── ProgressiveNeuralNetwork ──────────────────────────────────────────

    #[test]
    fn test_pnn_add_columns() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        let c0 = PnnColumnConfig {
            task_id: "task0".to_string(),
            layer_widths: vec![64, 32],
            input_dim: 784,
        };
        let c1 = PnnColumnConfig {
            task_id: "task1".to_string(),
            layer_widths: vec![64, 32],
            input_dim: 784,
        };
        pnn.add_column(c0).expect("col0");
        pnn.add_column(c1).expect("col1");
        assert_eq!(pnn.num_columns(), 2);
    }

    #[test]
    fn test_pnn_lateral_shapes() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        pnn.add_column(PnnColumnConfig {
            task_id: "t0".to_string(),
            layer_widths: vec![64, 32],
            input_dim: 16,
        })
        .expect("c0");
        pnn.add_column(PnnColumnConfig {
            task_id: "t1".to_string(),
            layer_widths: vec![128, 64],
            input_dim: 16,
        })
        .expect("c1");

        let shapes = pnn.lateral_adapter_shapes(0, 1).expect("shapes");
        assert_eq!(shapes.len(), 2);
        // Layer 0: prev=64, new=128
        assert_eq!(shapes[0], (64, 128));
        // Layer 1: prev=32, new=64
        assert_eq!(shapes[1], (32, 64));
    }

    #[test]
    fn test_pnn_all_lateral_shapes() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        for i in 0..3 {
            pnn.add_column(PnnColumnConfig {
                task_id: format!("t{i}"),
                layer_widths: vec![64],
                input_dim: 16,
            })
            .expect("col");
        }
        let all = pnn.all_lateral_shapes_for_new_column().expect("all shapes");
        // New column is index 2; should have laterals from 0 and 1
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, 0);
        assert_eq!(all[1].0, 1);
    }

    // ── RehearsalBuffer ───────────────────────────────────────────────────

    fn make_sample(label: usize, val: f64, step: u64) -> RehearsalSample {
        RehearsalSample {
            label,
            features: vec![val],
            soft_targets: None,
            insertion_step: step,
        }
    }

    #[test]
    fn test_rehearsal_buffer_capacity() {
        let mut buf =
            RehearsalBuffer::new(3, RehearsalStrategy::RingBuffer).expect("buf");
        for i in 0..6 {
            buf.add(make_sample(0, i as f64, i as u64));
        }
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_rehearsal_buffer_by_label() {
        let mut buf =
            RehearsalBuffer::new(10, RehearsalStrategy::Reservoir).expect("buf");
        for i in 0..4 {
            buf.add(make_sample(i % 2, i as f64, i as u64));
        }
        let class0 = buf.samples_for_label(0);
        assert_eq!(class0.len(), 2);
    }

    #[test]
    fn test_rehearsal_buffer_sample_batch() {
        let mut buf =
            RehearsalBuffer::new(10, RehearsalStrategy::Reservoir).expect("buf");
        for i in 0..8 {
            buf.add(make_sample(i, i as f64, i as u64));
        }
        let batch = buf.sample_batch(4);
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_rehearsal_buffer_fn() {
        let samples: Vec<RehearsalSample> = (0..5)
            .map(|i| make_sample(i, i as f64, i as u64))
            .collect();
        let buf = rehearsal_buffer(10, RehearsalStrategy::Reservoir, samples)
            .expect("rehearsal_buffer fn");
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_rehearsal_buffer_clear() {
        let mut buf =
            RehearsalBuffer::new(5, RehearsalStrategy::RingBuffer).expect("buf");
        buf.add(make_sample(0, 1.0, 0));
        buf.clear();
        assert!(buf.is_empty());
    }
}
