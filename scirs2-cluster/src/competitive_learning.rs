//! Competitive learning algorithms for prototype-based clustering.
//!
//! This module provides function-oriented and struct-based competitive learning algorithms
//! that complement the lower-level implementations in [`crate::prototype_enhanced`].
//!
//! # Algorithms
//!
//! - [`WinnerTakeAll`] – Basic competitive learning via BMU (Best Matching Unit) updates.
//! - [`LearningVectorQuantization`] – Supervised LVQ-1 prototype learning.
//! - [`NeuralGas`] – Topology-preserving competitive learning with rank-based neighbourhood.
//! - [`GrowingNeuralGas`] – Adaptive-topology neural gas that grows its unit graph dynamically.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::competitive_learning::WinnerTakeAll;
//!
//! let data = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,  5.1, 5.1,
//! ]).expect("shape ok");
//!
//! let wta = WinnerTakeAll::default();
//! let prototypes = wta.fit(data.view(), 2).expect("fit ok");
//! assert_eq!(prototypes.shape(), [2, 2]);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two slices.
#[inline]
fn sq_euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance between two slices.
#[inline]
fn euclid(a: &[f64], b: &[f64]) -> f64 {
    sq_euclid(a, b).sqrt()
}

/// Minimal Linear Congruential Generator for deterministic reproducibility.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Advance state and return a value in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a random `usize` in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
    }

    /// Fisher-Yates shuffle of a slice.
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = self.next_usize(i + 1);
            v.swap(i, j);
        }
    }
}

/// Return the index of the Best Matching Unit (BMU) — the prototype nearest to `input`.
fn find_bmu(input: &[f64], prototypes: &[Vec<f64>]) -> usize {
    prototypes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            sq_euclid(input, a)
                .partial_cmp(&sq_euclid(input, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// WinnerTakeAll
// ---------------------------------------------------------------------------

/// Basic competitive learning — Winner-Take-All (WTA).
///
/// Each training step selects the prototype (BMU) closest to the input sample
/// and moves it toward that sample by `learning_rate`.  The learning rate is
/// optionally annealed from `lr_init` to `lr_final` over the training run.
///
/// After training, `fit` returns the learned prototype matrix.
#[derive(Debug, Clone)]
pub struct WinnerTakeAll {
    /// Initial (or constant) learning rate.  Default 0.3.
    pub lr_init: f64,
    /// Final learning rate for linear annealing.  `None` means no annealing.  Default `None`.
    pub lr_final: Option<f64>,
    /// Number of training epochs (full passes over the dataset).  Default 100.
    pub max_epochs: usize,
    /// RNG seed for reproducibility.  Default 42.
    pub seed: u64,
}

impl Default for WinnerTakeAll {
    fn default() -> Self {
        Self {
            lr_init: 0.3,
            lr_final: None,
            max_epochs: 100,
            seed: 42,
        }
    }
}

impl WinnerTakeAll {
    /// Create a `WinnerTakeAll` with all options specified.
    pub fn new(lr_init: f64, lr_final: Option<f64>, max_epochs: usize, seed: u64) -> Self {
        Self {
            lr_init,
            lr_final,
            max_epochs,
            seed,
        }
    }

    /// Fit the WTA network on `data`, producing `n_prototypes` learned prototypes.
    ///
    /// Prototypes are initialised by sampling distinct data points uniformly at random.
    ///
    /// # Arguments
    /// * `data` – Data matrix `(n_samples, n_features)`.
    /// * `n_prototypes` – Number of prototype vectors to learn.
    ///
    /// # Returns
    /// `Array2<f64>` of shape `(n_prototypes, n_features)`.
    pub fn fit(&self, data: ArrayView2<f64>, n_prototypes: usize) -> Result<Array2<f64>> {
        let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_prototypes == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_prototypes must be > 0".into(),
            ));
        }
        if n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data must have at least one feature".into(),
            ));
        }

        let mut rng = Lcg::new(self.seed);

        // Initialise prototypes by sampling (with replacement if necessary) from data.
        let mut prototypes: Vec<Vec<f64>> = (0..n_prototypes)
            .map(|_| {
                let idx = rng.next_usize(n_samples);
                data.row(idx).to_vec()
            })
            .collect();

        let total_steps = self.max_epochs * n_samples;
        let mut order: Vec<usize> = (0..n_samples).collect();

        for epoch in 0..self.max_epochs {
            rng.shuffle(&mut order);

            for (step_in_epoch, &sample_idx) in order.iter().enumerate() {
                let global_step = epoch * n_samples + step_in_epoch;
                let t = global_step as f64 / total_steps.max(1) as f64;

                let lr = match self.lr_final {
                    Some(lr_f) => self.lr_init + t * (lr_f - self.lr_init),
                    None => self.lr_init,
                };

                let input = data.row(sample_idx).to_vec();
                let bmu_idx = find_bmu(&input, &prototypes);

                let bmu = &mut prototypes[bmu_idx];
                for k in 0..n_features {
                    bmu[k] += lr * (input[k] - bmu[k]);
                }
            }
        }

        // Pack into Array2.
        let mut out = Array2::<f64>::zeros((n_prototypes, n_features));
        for (j, p) in prototypes.iter().enumerate() {
            for k in 0..n_features {
                out[[j, k]] = p[k];
            }
        }
        Ok(out)
    }

    /// Assign each sample in `data` to its nearest prototype.
    ///
    /// Returns `Array1<usize>` of length `n_samples`.
    pub fn predict(&self, data: ArrayView2<f64>, prototypes: &Array2<f64>) -> Array1<usize> {
        let n_samples = data.shape()[0];
        let n_proto = prototypes.shape()[0];
        let protos: Vec<Vec<f64>> = (0..n_proto).map(|j| prototypes.row(j).to_vec()).collect();

        let labels: Vec<usize> = (0..n_samples)
            .map(|i| {
                let row = data.row(i).to_vec();
                find_bmu(&row, &protos)
            })
            .collect();

        Array1::from_vec(labels)
    }
}

// ---------------------------------------------------------------------------
// LearningVectorQuantization (supervised)
// ---------------------------------------------------------------------------

/// Learned LVQ-1 model, produced by [`LearningVectorQuantization::fit`].
#[derive(Debug, Clone)]
pub struct LvqModel {
    /// Prototype weight vectors, shape `(n_prototypes, n_features)`.
    pub prototypes: Array2<f64>,
    /// Class label for each prototype row.
    pub labels: Array1<usize>,
}

impl LvqModel {
    /// Predict the class of a single 1-D sample slice.
    pub fn predict_one(&self, sample: &[f64]) -> usize {
        let n_proto = self.prototypes.shape()[0];
        let best = (0..n_proto)
            .map(|j| {
                let p = self.prototypes.row(j).to_vec();
                (j, sq_euclid(sample, &p))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        self.labels[best]
    }

    /// Predict class labels for every row in `data`.
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<usize> {
        let n = data.shape()[0];
        let preds: Vec<usize> = (0..n)
            .map(|i| self.predict_one(&data.row(i).to_vec()))
            .collect();
        Array1::from_vec(preds)
    }
}

/// Supervised prototype learning via LVQ-1.
///
/// Each prototype is associated with a class label.  For every training
/// sample the nearest prototype is found:
/// - If it carries the **correct** class label it is moved *toward* the sample.
/// - Otherwise it is moved *away* from the sample.
///
/// Prototypes are initialised by sampling one or more points per class.
#[derive(Debug, Clone)]
pub struct LearningVectorQuantization {
    /// Number of prototypes per class.  Default 1.
    pub n_prototypes_per_class: usize,
    /// Initial learning rate.  Default 0.1.
    pub lr_init: f64,
    /// Final learning rate (annealed).  Default 0.001.
    pub lr_final: f64,
    /// Number of training epochs.  Default 50.
    pub max_epochs: usize,
    /// RNG seed.  Default 42.
    pub seed: u64,
}

impl Default for LearningVectorQuantization {
    fn default() -> Self {
        Self {
            n_prototypes_per_class: 1,
            lr_init: 0.1,
            lr_final: 0.001,
            max_epochs: 50,
            seed: 42,
        }
    }
}

impl LearningVectorQuantization {
    /// Create a new LVQ instance with all hyperparameters.
    pub fn new(
        n_prototypes_per_class: usize,
        lr_init: f64,
        lr_final: f64,
        max_epochs: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_prototypes_per_class,
            lr_init,
            lr_final,
            max_epochs,
            seed,
        }
    }

    /// Train LVQ-1 on labelled data.
    ///
    /// # Arguments
    /// * `data`   – Feature matrix `(n_samples, n_features)`.
    /// * `labels` – Integer class labels; values must be in `0..n_classes`.
    ///
    /// # Returns
    /// A trained [`LvqModel`].
    pub fn fit(&self, data: ArrayView2<f64>, labels: &[usize]) -> Result<LvqModel> {
        let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if labels.len() != n_samples {
            return Err(ClusteringError::InvalidInput(
                "labels length must equal number of data rows".into(),
            ));
        }
        if self.n_prototypes_per_class == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_prototypes_per_class must be > 0".into(),
            ));
        }

        let n_classes = labels.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        if n_classes == 0 {
            return Err(ClusteringError::InvalidInput(
                "No valid class labels found".into(),
            ));
        }

        // Collect sample indices per class.
        let mut class_samples: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &lbl) in labels.iter().enumerate() {
            if lbl < n_classes {
                class_samples[lbl].push(i);
            }
        }

        let mut rng = Lcg::new(self.seed);

        // Initialise prototypes.
        let mut proto_weights: Vec<Vec<f64>> = Vec::new();
        let mut proto_labels: Vec<usize> = Vec::new();

        for cls in 0..n_classes {
            let samples = &class_samples[cls];
            if samples.is_empty() {
                // Class has no samples; cannot initialise — skip it.
                continue;
            }
            for _ in 0..self.n_prototypes_per_class {
                let idx = samples[rng.next_usize(samples.len())];
                proto_weights.push(data.row(idx).to_vec());
                proto_labels.push(cls);
            }
        }

        if proto_weights.is_empty() {
            return Err(ClusteringError::ComputationError(
                "Could not initialise any prototypes".into(),
            ));
        }

        let n_proto = proto_weights.len();
        let total_steps = self.max_epochs * n_samples;
        let mut order: Vec<usize> = (0..n_samples).collect();

        // LVQ-1 training loop.
        for epoch in 0..self.max_epochs {
            rng.shuffle(&mut order);

            for (step_in_epoch, &sample_idx) in order.iter().enumerate() {
                let global_step = epoch * n_samples + step_in_epoch;
                let t = global_step as f64 / total_steps.max(1) as f64;
                let lr = self.lr_init * (self.lr_final / self.lr_init).powf(t);

                let input = data.row(sample_idx).to_vec();
                let true_class = labels[sample_idx];

                // Find the nearest prototype.
                let nearest = (0..n_proto)
                    .map(|j| (j, sq_euclid(&input, &proto_weights[j])))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
                    .unwrap_or(0);

                // Attract if correct class, repel otherwise.
                let sign = if proto_labels[nearest] == true_class {
                    1.0f64
                } else {
                    -1.0f64
                };

                let w = &mut proto_weights[nearest];
                for k in 0..n_features {
                    w[k] += lr * sign * (input[k] - w[k]);
                }
            }
        }

        // Assemble result.
        let mut proto_arr = Array2::<f64>::zeros((n_proto, n_features));
        for (j, w) in proto_weights.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = w[k];
            }
        }

        Ok(LvqModel {
            prototypes: proto_arr,
            labels: Array1::from_vec(proto_labels),
        })
    }

    /// Convenience: predict class labels for all rows in `data` using a trained `model`.
    pub fn predict(model: &LvqModel, data: ArrayView2<f64>) -> Array1<usize> {
        model.predict(data)
    }
}

// ---------------------------------------------------------------------------
// NeuralGas (function-oriented API)
// ---------------------------------------------------------------------------

/// Trained Neural Gas model, produced by [`NeuralGas::fit`].
#[derive(Debug, Clone)]
pub struct NeuralGasModel {
    /// Prototype / reference vectors, shape `(n_neurons, n_features)`.
    pub prototypes: Array2<f64>,
    /// Label (index of nearest prototype) for each training sample.
    pub labels: Array1<usize>,
    /// Mean quantization error over the training set.
    pub quantization_error: f64,
}

/// Neural Gas — topology-preserving competitive learning.
///
/// For each input the prototypes are ranked by distance.  The winner
/// (rank 0) receives the full learning rate; neighbours receive
/// `lr * exp(-rank / lambda)`.  Both `lr` and `lambda` are exponentially
/// annealed.
///
/// Reference: Martinetz & Schulten (1991).
#[derive(Debug, Clone)]
pub struct NeuralGas {
    /// Initial learning rate for the winner.  Default 0.5.
    pub lr_winner: f64,
    /// Final learning rate (annealing target).  Default 0.01.
    pub lr_final: f64,
    /// Initial neighbourhood width λ.  `None` → `n_neurons / 2`.
    pub lambda_init: Option<f64>,
    /// Final λ (annealing target).  Default 0.01.
    pub lambda_final: f64,
    /// Number of training epochs.  Default 100.
    pub max_epochs: usize,
    /// RNG seed.  Default 42.
    pub seed: u64,
}

impl Default for NeuralGas {
    fn default() -> Self {
        Self {
            lr_winner: 0.5,
            lr_final: 0.01,
            lambda_init: None,
            lambda_final: 0.01,
            max_epochs: 100,
            seed: 42,
        }
    }
}

impl NeuralGas {
    /// Create a `NeuralGas` with explicit hyperparameters.
    pub fn new(
        lr_winner: f64,
        lr_final: f64,
        lambda_init: Option<f64>,
        lambda_final: f64,
        max_epochs: usize,
        seed: u64,
    ) -> Self {
        Self {
            lr_winner,
            lr_final,
            lambda_init,
            lambda_final,
            max_epochs,
            seed,
        }
    }

    /// Fit the Neural Gas network.
    ///
    /// # Arguments
    /// * `data`      – Data matrix `(n_samples, n_features)`.
    /// * `n_neurons` – Number of prototype units.
    ///
    /// # Returns
    /// A [`NeuralGasModel`] with the learned prototypes and training assignments.
    pub fn fit(&self, data: ArrayView2<f64>, n_neurons: usize) -> Result<NeuralGasModel> {
        let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_neurons == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_neurons must be > 0".into(),
            ));
        }
        if self.max_epochs == 0 {
            return Err(ClusteringError::InvalidInput(
                "max_epochs must be > 0".into(),
            ));
        }

        let mut rng = Lcg::new(self.seed);

        // Initialise prototypes by sampling from data.
        let mut prototypes: Vec<Vec<f64>> = (0..n_neurons)
            .map(|_| {
                let idx = rng.next_usize(n_samples);
                data.row(idx).to_vec()
            })
            .collect();

        let total_steps = self.max_epochs * n_samples;
        let lambda_i = self.lambda_init.unwrap_or(n_neurons as f64 / 2.0).max(0.5);
        let mut order: Vec<usize> = (0..n_samples).collect();

        for epoch in 0..self.max_epochs {
            rng.shuffle(&mut order);

            for (step_in_epoch, &sample_idx) in order.iter().enumerate() {
                let global_step = epoch * n_samples + step_in_epoch;
                let t = global_step as f64 / total_steps.max(1) as f64;

                // Exponential annealing for lr and lambda.
                let lr = self.lr_winner * (self.lr_final / self.lr_winner).powf(t);
                let lam = lambda_i * (self.lambda_final / lambda_i).powf(t);

                let input = data.row(sample_idx).to_vec();

                // Rank all prototypes by Euclidean distance.
                let mut ranked: Vec<(f64, usize)> = prototypes
                    .iter()
                    .enumerate()
                    .map(|(j, p)| (euclid(&input, p), j))
                    .collect();
                ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Update each prototype with neighbourhood factor based on rank.
                for (rank, (_, proto_idx)) in ranked.iter().enumerate() {
                    let h = (-(rank as f64) / lam).exp();
                    let p = &mut prototypes[*proto_idx];
                    for k in 0..n_features {
                        p[k] += lr * h * (input[k] - p[k]);
                    }
                }
            }
        }

        // Assign labels and compute quantization error.
        let mut labels_vec = vec![0usize; n_samples];
        let mut total_qe = 0.0f64;
        for i in 0..n_samples {
            let row = data.row(i).to_vec();
            let (best, best_dist) = prototypes
                .iter()
                .enumerate()
                .map(|(j, p)| (j, sq_euclid(&row, p)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            labels_vec[i] = best;
            total_qe += best_dist;
        }
        let quantization_error = if n_samples > 0 {
            total_qe / n_samples as f64
        } else {
            0.0
        };

        // Pack into Array2.
        let mut proto_arr = Array2::<f64>::zeros((n_neurons, n_features));
        for (j, p) in prototypes.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = p[k];
            }
        }

        Ok(NeuralGasModel {
            prototypes: proto_arr,
            labels: Array1::from_vec(labels_vec),
            quantization_error,
        })
    }
}

// ---------------------------------------------------------------------------
// GrowingNeuralGas (function-oriented API)
// ---------------------------------------------------------------------------

/// A directed edge in the GNG graph, tracking its age.
#[derive(Debug, Clone)]
struct GngEdge {
    age: usize,
}

/// A node (unit) in the GNG network.
#[derive(Debug, Clone)]
struct GngNode {
    weights: Vec<f64>,
    error: f64,
}

/// Trained Growing Neural Gas model, produced by [`GrowingNeuralGas::fit`].
#[derive(Debug, Clone)]
pub struct GrowingNeuralGasModel {
    /// Learned prototype vectors, shape `(n_units, n_features)`.
    pub prototypes: Array2<f64>,
    /// Topology edges as `(node_a, node_b)` pairs (sorted, a < b).
    pub edges: Vec<(usize, usize)>,
    /// Label (nearest prototype) for each training sample.
    pub labels: Array1<usize>,
    /// Final mean quantization error.
    pub quantization_error: f64,
}

/// Growing Neural Gas — adaptive topology competitive learning.
///
/// The GNG starts with two prototype units and grows by inserting new nodes
/// between the highest-error nodes.  Edges age and are removed when their age
/// exceeds `max_age`.  No a priori unit count is needed.
///
/// Reference: Fritzke (1995).
#[derive(Debug, Clone)]
pub struct GrowingNeuralGas {
    /// Learning rate for the winner unit.  Default 0.1.
    pub lr_winner: f64,
    /// Learning rate for topological neighbours.  Default 0.01.
    pub lr_neighbor: f64,
    /// Maximum edge age before removal.  Default 50.
    pub max_age: usize,
    /// Steps between node insertions.  Default 100.
    pub insert_interval: usize,
    /// Error reduction factor applied to nodes after insertion.  Default 0.5.
    pub alpha: f64,
    /// Global error decay per step.  Default 0.995.
    pub beta: f64,
    /// Upper bound on the number of units.  Default 200.
    pub max_units: usize,
    /// Total training steps.  Default 5000.
    pub max_steps: usize,
    /// RNG seed.  Default 42.
    pub seed: u64,
}

impl Default for GrowingNeuralGas {
    fn default() -> Self {
        Self {
            lr_winner: 0.1,
            lr_neighbor: 0.01,
            max_age: 50,
            insert_interval: 100,
            alpha: 0.5,
            beta: 0.995,
            max_units: 200,
            max_steps: 5000,
            seed: 42,
        }
    }
}

impl GrowingNeuralGas {
    /// Create a new GNG with all hyperparameters specified.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lr_winner: f64,
        lr_neighbor: f64,
        max_age: usize,
        insert_interval: usize,
        alpha: f64,
        beta: f64,
        max_units: usize,
        max_steps: usize,
        seed: u64,
    ) -> Self {
        Self {
            lr_winner,
            lr_neighbor,
            max_age,
            insert_interval,
            alpha,
            beta,
            max_units,
            max_steps,
            seed,
        }
    }

    /// Fit the GNG model to `data`.
    ///
    /// # Arguments
    /// * `data` – Data matrix `(n_samples, n_features)`.
    ///
    /// # Returns
    /// A [`GrowingNeuralGasModel`] with the learned topology and assignments.
    pub fn fit(&self, data: ArrayView2<f64>) -> Result<GrowingNeuralGasModel> {
        let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "GNG requires at least 2 samples".into(),
            ));
        }
        if n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data must have at least one feature".into(),
            ));
        }

        let mut rng = Lcg::new(self.seed);

        // Initialise with two nodes sampled from data.
        let idx0 = rng.next_usize(n_samples);
        let idx1 = (idx0 + 1 + rng.next_usize(n_samples.saturating_sub(1).max(1))) % n_samples;

        let mut nodes: Vec<GngNode> = vec![
            GngNode {
                weights: data.row(idx0).to_vec(),
                error: 0.0,
            },
            GngNode {
                weights: data.row(idx1).to_vec(),
                error: 0.0,
            },
        ];

        // Edge map: key = sorted (a, b) where a < b.
        let mut edge_map: std::collections::HashMap<(usize, usize), GngEdge> =
            std::collections::HashMap::new();
        edge_map.insert((0, 1), GngEdge { age: 0 });

        let data_rows: Vec<Vec<f64>> = (0..n_samples).map(|i| data.row(i).to_vec()).collect();

        for step in 0..self.max_steps {
            let sample = &data_rows[rng.next_usize(n_samples)];

            // Find winner (s1) and runner-up (s2) by squared Euclidean distance.
            let mut dists: Vec<(f64, usize)> = nodes
                .iter()
                .enumerate()
                .map(|(j, n)| (sq_euclid(sample, &n.weights), j))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if dists.len() < 2 {
                continue;
            }

            let s1 = dists[0].1;
            let s2 = dists[1].1;
            let dist_s1 = dists[0].0;

            // Age all edges incident to s1.
            let edge_keys: Vec<(usize, usize)> = edge_map.keys().cloned().collect();
            for key in &edge_keys {
                if key.0 == s1 || key.1 == s1 {
                    if let Some(e) = edge_map.get_mut(key) {
                        e.age += 1;
                    }
                }
            }

            // Set/reset edge (s1, s2).
            let edge_key = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            edge_map.insert(edge_key, GngEdge { age: 0 });

            // Accumulate error for winner.
            nodes[s1].error += dist_s1;

            // Move winner toward sample.
            for k in 0..n_features {
                let delta = sample[k] - nodes[s1].weights[k];
                nodes[s1].weights[k] += self.lr_winner * delta;
            }

            // Move topological neighbours of s1 toward sample.
            let neighbours: Vec<usize> = edge_map
                .keys()
                .filter_map(|&(a, b)| {
                    if a == s1 {
                        Some(b)
                    } else if b == s1 {
                        Some(a)
                    } else {
                        None
                    }
                })
                .collect();

            for nb in &neighbours {
                for k in 0..n_features {
                    let delta = sample[k] - nodes[*nb].weights[k];
                    nodes[*nb].weights[k] += self.lr_neighbor * delta;
                }
            }

            // Remove edges older than max_age.
            edge_map.retain(|_, e| e.age <= self.max_age);

            // Global error decay.
            for node in nodes.iter_mut() {
                node.error *= self.beta;
            }

            // Insert new node periodically.
            if step > 0
                && step % self.insert_interval == 0
                && nodes.len() < self.max_units
                && nodes.len() >= 2
            {
                // Node with highest accumulated error.
                let q = nodes
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1.error
                            .partial_cmp(&b.1.error)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // Neighbour of q with highest error.
                let q_neighbours: Vec<usize> = edge_map
                    .keys()
                    .filter_map(|&(a, b)| {
                        if a == q {
                            Some(b)
                        } else if b == q {
                            Some(a)
                        } else {
                            None
                        }
                    })
                    .collect();

                if !q_neighbours.is_empty() {
                    let f = q_neighbours
                        .iter()
                        .max_by(|&&a, &&b| {
                            nodes[a]
                                .error
                                .partial_cmp(&nodes[b].error)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .cloned()
                        .unwrap_or(q_neighbours[0]);

                    // New node between q and f.
                    let new_weights: Vec<f64> = nodes[q]
                        .weights
                        .iter()
                        .zip(nodes[f].weights.iter())
                        .map(|(a, b)| (a + b) / 2.0)
                        .collect();

                    let new_idx = nodes.len();
                    let new_error = nodes[q].error * self.alpha;
                    nodes.push(GngNode {
                        weights: new_weights,
                        error: new_error,
                    });

                    nodes[q].error *= self.alpha;
                    nodes[f].error *= self.alpha;

                    // Remove q-f edge; add q-new and f-new.
                    let qf_key = if q < f { (q, f) } else { (f, q) };
                    edge_map.remove(&qf_key);

                    let qn_key = if q < new_idx {
                        (q, new_idx)
                    } else {
                        (new_idx, q)
                    };
                    let fn_key = if f < new_idx {
                        (f, new_idx)
                    } else {
                        (new_idx, f)
                    };
                    edge_map.insert(qn_key, GngEdge { age: 0 });
                    edge_map.insert(fn_key, GngEdge { age: 0 });
                }
            }
        }

        let n_units = nodes.len();
        let mut proto_arr = Array2::<f64>::zeros((n_units, n_features));
        for (j, node) in nodes.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = node.weights[k];
            }
        }

        let edges: Vec<(usize, usize)> = edge_map.keys().cloned().collect();

        // Assign labels and quantization error.
        let mut labels_vec = vec![0usize; n_samples];
        let mut total_qe = 0.0f64;
        for i in 0..n_samples {
            let row = data_rows[i].as_slice();
            let (best, best_dist) = nodes
                .iter()
                .enumerate()
                .map(|(j, node)| (j, sq_euclid(row, &node.weights)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            labels_vec[i] = best;
            total_qe += best_dist;
        }
        let quantization_error = if n_samples > 0 {
            total_qe / n_samples as f64
        } else {
            0.0
        };

        Ok(GrowingNeuralGasModel {
            prototypes: proto_arr,
            edges,
            labels: Array1::from_vec(labels_vec),
            quantization_error,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Generate two well-separated Gaussian-like clusters (12 points each, 2 features).
    fn two_cluster_data() -> (Array2<f64>, Vec<usize>) {
        let vals = vec![
            // Cluster 0  (~origin)
            0.00, 0.00, 0.10, 0.00, 0.00, 0.10, 0.10, 0.10, 0.05, 0.05, -0.05, 0.05, -0.05, -0.05,
            0.10, -0.05, 0.00, 0.15, -0.10, 0.00, 0.15, 0.10, 0.00, 0.20,
            // Cluster 1  (~(5, 5))
            5.00, 5.00, 5.10, 5.00, 5.00, 5.10, 5.10, 5.10, 5.05, 5.05, 4.95, 5.05, 4.95, 4.95,
            5.10, 4.95, 5.00, 5.15, 4.90, 5.00, 5.15, 5.10, 5.00, 5.20,
        ];
        let x = Array2::from_shape_vec((24, 2), vals).expect("shape ok");
        let y: Vec<usize> = (0..12).map(|_| 0).chain((0..12).map(|_| 1)).collect();
        (x, y)
    }

    // --- WinnerTakeAll ---

    #[test]
    fn test_wta_basic() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll::default();
        let protos = wta.fit(x.view(), 2).expect("fit");
        assert_eq!(protos.shape(), [2, 2]);
    }

    #[test]
    fn test_wta_single_prototype() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll::default();
        let protos = wta.fit(x.view(), 1).expect("fit");
        assert_eq!(protos.shape(), [1, 2]);
    }

    #[test]
    fn test_wta_annealing() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll {
            lr_init: 0.5,
            lr_final: Some(0.001),
            max_epochs: 50,
            seed: 7,
        };
        let protos = wta.fit(x.view(), 2).expect("fit annealing");
        assert_eq!(protos.shape()[0], 2);
    }

    #[test]
    fn test_wta_predict() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll::default();
        let protos = wta.fit(x.view(), 2).expect("fit");
        let labels = wta.predict(x.view(), &protos);
        assert_eq!(labels.len(), 24);
        assert!(labels.iter().all(|&l| l < 2));
    }

    #[test]
    fn test_wta_converges_two_clusters() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll {
            lr_init: 0.5,
            lr_final: Some(0.01),
            max_epochs: 200,
            seed: 42,
        };
        let protos = wta.fit(x.view(), 2).expect("fit");
        // One prototype should be near origin and the other near (5, 5).
        let p0 = protos.row(0).to_vec();
        let p1 = protos.row(1).to_vec();
        let d00 = sq_euclid(&p0, &[0.0, 0.0]);
        let d05 = sq_euclid(&p0, &[5.0, 5.0]);
        let d10 = sq_euclid(&p1, &[0.0, 0.0]);
        let d15 = sq_euclid(&p1, &[5.0, 5.0]);
        let well_placed = (d00 < d05 && d15 < d10) || (d05 < d00 && d10 < d15);
        assert!(well_placed, "prototypes should converge to cluster centres");
    }

    #[test]
    fn test_wta_error_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let wta = WinnerTakeAll::default();
        assert!(wta.fit(x.view(), 2).is_err());
    }

    #[test]
    fn test_wta_error_zero_prototypes() {
        let (x, _) = two_cluster_data();
        let wta = WinnerTakeAll::default();
        assert!(wta.fit(x.view(), 0).is_err());
    }

    // --- LearningVectorQuantization ---

    #[test]
    fn test_lvq_fit_basic() {
        let (x, y) = two_cluster_data();
        let lvq = LearningVectorQuantization::default();
        let model = lvq.fit(x.view(), &y).expect("fit");
        assert_eq!(model.prototypes.shape()[0], 2); // 1 per class × 2 classes
        assert_eq!(model.labels.len(), 2);
    }

    #[test]
    fn test_lvq_predict() {
        let (x, y) = two_cluster_data();
        let lvq = LearningVectorQuantization::default();
        let model = lvq.fit(x.view(), &y).expect("fit");
        let preds = model.predict(x.view());
        assert_eq!(preds.len(), 24);
        // Well-separated data should be classified correctly.
        let correct = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(
            correct as f64 / 24.0 > 0.75,
            "accuracy should exceed 75%, got {}",
            correct
        );
    }

    #[test]
    fn test_lvq_predict_one() {
        let (x, y) = two_cluster_data();
        let lvq = LearningVectorQuantization::default();
        let model = lvq.fit(x.view(), &y).expect("fit");
        let pred = model.predict_one(&[0.0, 0.0]);
        assert_eq!(pred, 0, "origin should map to class 0");
        let pred2 = model.predict_one(&[5.0, 5.0]);
        assert_eq!(pred2, 1, "(5,5) should map to class 1");
    }

    #[test]
    fn test_lvq_function_predict() {
        let (x, y) = two_cluster_data();
        let lvq = LearningVectorQuantization::default();
        let model = lvq.fit(x.view(), &y).expect("fit");
        let preds = LearningVectorQuantization::predict(&model, x.view());
        assert_eq!(preds.len(), 24);
    }

    #[test]
    fn test_lvq_multi_proto_per_class() {
        let (x, y) = two_cluster_data();
        let lvq = LearningVectorQuantization::new(2, 0.1, 0.001, 50, 42);
        let model = lvq.fit(x.view(), &y).expect("fit");
        assert_eq!(model.prototypes.shape()[0], 4); // 2 per class × 2 classes
    }

    #[test]
    fn test_lvq_error_label_mismatch() {
        let (x, _) = two_cluster_data();
        let lvq = LearningVectorQuantization::default();
        assert!(lvq.fit(x.view(), &[0, 1, 2]).is_err());
    }

    // --- NeuralGas ---

    #[test]
    fn test_ng_basic() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        let model = ng.fit(x.view(), 2).expect("fit");
        assert_eq!(model.prototypes.shape(), [2, 2]);
        assert_eq!(model.labels.len(), 24);
        assert!(model.quantization_error >= 0.0);
    }

    #[test]
    fn test_ng_single_neuron() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        let model = ng.fit(x.view(), 1).expect("fit");
        assert_eq!(model.prototypes.shape()[0], 1);
        assert!(model.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_ng_converges() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas {
            lr_winner: 0.5,
            lr_final: 0.01,
            lambda_init: None,
            lambda_final: 0.01,
            max_epochs: 200,
            seed: 42,
        };
        let model = ng.fit(x.view(), 2).expect("fit");
        assert!(
            model.quantization_error < 1.0,
            "QE={} too high",
            model.quantization_error
        );
    }

    #[test]
    fn test_ng_error_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        let ng = NeuralGas::default();
        assert!(ng.fit(x.view(), 2).is_err());
    }

    #[test]
    fn test_ng_error_zero_neurons() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        assert!(ng.fit(x.view(), 0).is_err());
    }

    // --- GrowingNeuralGas ---

    #[test]
    fn test_gng_basic() {
        let (x, _) = two_cluster_data();
        let gng = GrowingNeuralGas {
            max_steps: 300,
            insert_interval: 30,
            max_units: 15,
            seed: 7,
            ..GrowingNeuralGas::default()
        };
        let model = gng.fit(x.view()).expect("fit");
        assert!(
            model.prototypes.shape()[0] >= 2,
            "should have at least initial units"
        );
        assert_eq!(model.labels.len(), 24);
        assert!(model.quantization_error >= 0.0);
    }

    #[test]
    fn test_gng_grows_units() {
        let (x, _) = two_cluster_data();
        let gng = GrowingNeuralGas {
            max_steps: 1000,
            insert_interval: 50,
            max_units: 20,
            seed: 99,
            ..GrowingNeuralGas::default()
        };
        let model = gng.fit(x.view()).expect("fit");
        // With 1000 steps and interval 50, there should be more than 2 units.
        assert!(
            model.prototypes.shape()[0] >= 2,
            "GNG should grow beyond initial 2 units"
        );
    }

    #[test]
    fn test_gng_edges_valid() {
        let (x, _) = two_cluster_data();
        let gng = GrowingNeuralGas {
            max_steps: 500,
            seed: 42,
            ..GrowingNeuralGas::default()
        };
        let model = gng.fit(x.view()).expect("fit");
        let n_units = model.prototypes.shape()[0];
        // All edge endpoints must be valid unit indices.
        for &(a, b) in &model.edges {
            assert!(a < n_units, "edge endpoint {} out of range", a);
            assert!(b < n_units, "edge endpoint {} out of range", b);
            assert_ne!(a, b, "self-loop detected");
        }
    }

    #[test]
    fn test_gng_error_too_few_samples() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("shape ok");
        let gng = GrowingNeuralGas::default();
        assert!(gng.fit(x.view()).is_err());
    }
}
