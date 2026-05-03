//! Enhanced prototype-based clustering algorithms
//!
//! This module provides advanced prototype-based methods beyond standard K-means,
//! including competitive learning networks and learning vector quantization variants.
//!
//! # Algorithms
//!
//! - **Neural Gas**: Topology-preserving competitive learning with rank-ordered updates
//! - **Growing Neural Gas (GNG)**: Adaptive topology without a fixed unit count
//! - **LVQ** (Learning Vector Quantization): Supervised prototype adaptation
//! - **GLVQ** (Generalized LVQ): Soft-margin prototype learning with adaptive metrics

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Shared distance helpers
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

/// LCG pseudo-random number generator state.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Map to [0, 1).
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Draw a usize in [0, n) using a LCG state.
#[inline]
fn lcg_usize(state: &mut u64, n: usize) -> usize {
    lcg_next(state) as usize % n
}

// ---------------------------------------------------------------------------
// Neural Gas
// ---------------------------------------------------------------------------

/// Result of Neural Gas clustering.
#[derive(Debug, Clone)]
pub struct NeuralGasResult {
    /// Prototype / reference vectors, shape `(n_units, n_features)`.
    pub prototypes: Array2<f64>,
    /// Label of the nearest prototype for each training sample.
    pub labels: Array1<usize>,
    /// Number of prototype units.
    pub n_units: usize,
    /// Final quantization error (mean squared distance to nearest prototype).
    pub quantization_error: f64,
}

/// Neural Gas unsupervised competitive learning network.
///
/// For each input, ranks all prototypes by distance and applies a
/// neighbourhood function `h(k, λ)` that decreases with rank `k`.
/// Over training, both the learning rate `ε` and neighbourhood
/// parameter `λ` are annealed from their initial to their final values.
///
/// Reference: Martinetz & Schulten, 1991.
pub struct NeuralGas {
    /// Initial learning rate (default 0.5).
    pub lr_i: f64,
    /// Final learning rate (default 0.01).
    pub lr_f: f64,
    /// Initial neighbourhood parameter λ (default `n_units / 2`).
    pub lambda_i: Option<f64>,
    /// Final neighbourhood parameter λ (default 0.01).
    pub lambda_f: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for NeuralGas {
    fn default() -> Self {
        Self {
            lr_i: 0.5,
            lr_f: 0.01,
            lambda_i: None,
            lambda_f: 0.01,
            seed: 42,
        }
    }
}

impl NeuralGas {
    /// Fit Neural Gas.
    ///
    /// # Arguments
    /// * `x` – Data matrix `(n_samples, n_features)`.
    /// * `n_units` – Number of prototype units.
    /// * `max_iter` – Number of training epochs (passes over the data).
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        n_units: usize,
        max_iter: usize,
    ) -> Result<NeuralGasResult> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if n_units == 0 {
            return Err(ClusteringError::InvalidInput("n_units must be > 0".into()));
        }
        if max_iter == 0 {
            return Err(ClusteringError::InvalidInput("max_iter must be > 0".into()));
        }

        let mut rng = self.seed;

        // Initialise prototypes by sampling data points.
        let mut protos: Vec<Vec<f64>> = (0..n_units)
            .map(|_| {
                let idx = lcg_usize(&mut rng, n_samples);
                x.row(idx).to_vec()
            })
            .collect();

        let total_steps = max_iter * n_samples;
        let lambda_i = self.lambda_i.unwrap_or((n_units as f64) / 2.0).max(0.5);

        for epoch in 0..max_iter {
            // Shuffle sample order each epoch.
            let mut order: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = lcg_usize(&mut rng, i + 1);
                order.swap(i, j);
            }

            for &sample_idx in &order {
                // Global step index for annealing schedule.
                let step = epoch * n_samples + sample_idx;
                let t = step as f64 / total_steps.max(1) as f64;

                // Anneal learning rate and lambda.
                let lr = self.lr_i * (self.lr_f / self.lr_i).powf(t);
                let lam = lambda_i * (self.lambda_f / lambda_i).powf(t);

                let input = x.row(sample_idx).to_vec();

                // Rank all prototypes by distance to input.
                let mut ranked: Vec<(f64, usize)> = protos
                    .iter()
                    .enumerate()
                    .map(|(j, p)| (euclid(&input, p), j))
                    .collect();
                ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Update each prototype with neighbourhood factor based on rank.
                for (rank, (_, proto_idx)) in ranked.iter().enumerate() {
                    let h = (-(rank as f64) / lam).exp();
                    let p = &mut protos[*proto_idx];
                    for k in 0..n_features {
                        p[k] += lr * h * (input[k] - p[k]);
                    }
                }
            }
        }

        // Assign labels and compute quantization error.
        let mut labels = vec![0usize; n_samples];
        let mut total_qe = 0.0f64;
        for i in 0..n_samples {
            let row = x.row(i).to_vec();
            let (best, best_dist) = protos
                .iter()
                .enumerate()
                .map(|(j, p)| (j, sq_euclid(&row, p)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            labels[i] = best;
            total_qe += best_dist;
        }
        let quantization_error = total_qe / n_samples as f64;

        // Pack prototypes into Array2.
        let mut proto_arr = Array2::<f64>::zeros((n_units, n_features));
        for (j, p) in protos.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = p[k];
            }
        }

        Ok(NeuralGasResult {
            prototypes: proto_arr,
            labels: Array1::from_vec(labels),
            n_units,
            quantization_error,
        })
    }
}

// ---------------------------------------------------------------------------
// Growing Neural Gas
// ---------------------------------------------------------------------------

/// An edge in the GNG topology graph.
#[derive(Debug, Clone)]
struct GngEdge {
    /// Age of the edge (incremented each time a winner is updated without
    /// this edge being refreshed).
    age: usize,
}

/// A node (unit) in the Growing Neural Gas network.
#[derive(Debug, Clone)]
struct GngNode {
    /// Reference vector (prototype).
    weights: Vec<f64>,
    /// Accumulated local error.
    error: f64,
}

/// Configuration for Growing Neural Gas.
#[derive(Debug, Clone)]
pub struct GngConfig {
    /// Learning rate for the winner unit (default 0.1).
    pub lr_winner: f64,
    /// Learning rate for winner's neighbours (default 0.01).
    pub lr_neighbor: f64,
    /// Maximum edge age before removal (default 50).
    pub max_age: usize,
    /// How often (in steps) a new node is inserted (default 100).
    pub insert_interval: usize,
    /// Error reduction factor for all nodes after node insertion (default 0.5).
    pub alpha: f64,
    /// Global error decay per step (default 0.995).
    pub beta: f64,
    /// Maximum number of units (stops inserting when reached, default 200).
    pub max_units: usize,
    /// Total training steps.
    pub max_steps: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GngConfig {
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

/// Result of Growing Neural Gas.
#[derive(Debug, Clone)]
pub struct GngResult {
    /// Learned prototype weights `(n_units, n_features)`.
    pub prototypes: Array2<f64>,
    /// Edges as (node_a, node_b) pairs.
    pub edges: Vec<(usize, usize)>,
    /// Label of the nearest prototype for each training sample.
    pub labels: Array1<usize>,
    /// Final quantization error.
    pub quantization_error: f64,
}

/// Growing Neural Gas — adaptive topology competitive learning.
///
/// Unlike Neural Gas, GNG starts with two units and grows by inserting new
/// units between high-error units. Edges are added/removed dynamically.
///
/// Reference: Fritzke, 1995.
pub struct GrowingNeuralGas {
    /// Configuration.
    pub config: GngConfig,
}

impl Default for GrowingNeuralGas {
    fn default() -> Self {
        Self {
            config: GngConfig::default(),
        }
    }
}

impl GrowingNeuralGas {
    /// Create a new GNG with the given config.
    pub fn new(config: GngConfig) -> Self {
        Self { config }
    }

    /// Fit the GNG model to data `x`.
    pub fn fit(&self, x: ArrayView2<f64>) -> Result<GngResult> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 samples for GNG".into(),
            ));
        }

        let cfg = &self.config;
        let mut rng = cfg.seed;

        // Initialise with two nodes sampled from data.
        let idx0 = lcg_usize(&mut rng, n_samples);
        let idx1 = (idx0 + 1 + lcg_usize(&mut rng, n_samples - 1)) % n_samples;
        let mut nodes: Vec<GngNode> = vec![
            GngNode {
                weights: x.row(idx0).to_vec(),
                error: 0.0,
            },
            GngNode {
                weights: x.row(idx1).to_vec(),
                error: 0.0,
            },
        ];
        // Adjacency: edges[i][j] = Option<GngEdge>
        // Use a HashMap keyed by sorted (i, j) pairs.
        let mut edge_map: HashMap<(usize, usize), GngEdge> = HashMap::new();
        // Add initial edge.
        edge_map.insert((0, 1), GngEdge { age: 0 });

        let mut step = 0usize;
        let data_vec: Vec<Vec<f64>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        while step < cfg.max_steps {
            // Pick random sample.
            let sample = &data_vec[lcg_usize(&mut rng, n_samples)];

            // Find winner (s1) and runner-up (s2).
            let mut dists: Vec<(f64, usize)> = nodes
                .iter()
                .enumerate()
                .map(|(j, n)| (sq_euclid(sample, &n.weights), j))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if dists.len() < 2 {
                step += 1;
                continue;
            }

            let s1 = dists[0].1;
            let s2 = dists[1].1;
            let dist_s1 = dists[0].0;

            // Increment age of all edges incident to s1.
            let edge_keys: Vec<(usize, usize)> = edge_map.keys().cloned().collect();
            for key in &edge_keys {
                if key.0 == s1 || key.1 == s1 {
                    if let Some(e) = edge_map.get_mut(key) {
                        e.age += 1;
                    }
                }
            }

            // Add/reset edge (s1, s2).
            let edge_key = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            edge_map.insert(edge_key, GngEdge { age: 0 });

            // Accumulate error for winner.
            nodes[s1].error += dist_s1;

            // Move winner and its topological neighbours toward sample.
            let n_nodes = nodes.len();
            let winner_w: Vec<f64> = nodes[s1].weights.clone();
            for k in 0..n_features {
                nodes[s1].weights[k] += cfg.lr_winner * (sample[k] - winner_w[k]);
            }

            let neighbor_ids: Vec<usize> = edge_map
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

            for nb in &neighbor_ids {
                let nb_w: Vec<f64> = nodes[*nb].weights.clone();
                for k in 0..n_features {
                    nodes[*nb].weights[k] += cfg.lr_neighbor * (sample[k] - nb_w[k]);
                }
            }

            // Remove edges older than max_age.
            edge_map.retain(|_, e| e.age <= cfg.max_age);

            // Remove isolated nodes (no edges).
            // (Only do this after removing edges.)
            let connected: std::collections::HashSet<usize> =
                edge_map.keys().flat_map(|&(a, b)| [a, b]).collect();
            // We'll skip removing nodes to keep index stability (just leave them).

            // Apply global error decay.
            for node in nodes.iter_mut() {
                node.error *= cfg.beta;
            }

            // Insert new node periodically.
            if step % cfg.insert_interval == 0 && nodes.len() < cfg.max_units && nodes.len() >= 2 {
                // Find node with highest error.
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

                // Find the neighbour of q with highest error.
                let q_neighbors: Vec<usize> = edge_map
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

                if !q_neighbors.is_empty() {
                    let f = q_neighbors
                        .iter()
                        .max_by(|&&a, &&b| {
                            nodes[a]
                                .error
                                .partial_cmp(&nodes[b].error)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .cloned()
                        .unwrap_or(q_neighbors[0]);

                    // Insert new node between q and f.
                    let new_weights: Vec<f64> = nodes[q]
                        .weights
                        .iter()
                        .zip(nodes[f].weights.iter())
                        .map(|(a, b)| (a + b) / 2.0)
                        .collect();
                    let new_idx = nodes.len();
                    nodes.push(GngNode {
                        weights: new_weights,
                        error: nodes[q].error * cfg.alpha,
                    });

                    // Adjust errors.
                    nodes[q].error *= cfg.alpha;
                    nodes[f].error *= cfg.alpha;

                    // Remove q-f edge, add q-new and f-new edges.
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

            step += 1;
        }

        let n_units = nodes.len();
        let mut proto_arr = Array2::<f64>::zeros((n_units, n_features));
        for (j, node) in nodes.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = node.weights[k];
            }
        }

        let edges: Vec<(usize, usize)> = edge_map.keys().cloned().collect();

        // Assign labels.
        let mut labels = vec![0usize; n_samples];
        let mut total_qe = 0.0f64;
        for i in 0..n_samples {
            let row = x.row(i).to_vec();
            let (best, best_dist) = nodes
                .iter()
                .enumerate()
                .map(|(j, node)| (j, sq_euclid(&row, &node.weights)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            labels[i] = best;
            total_qe += best_dist;
        }

        Ok(GngResult {
            prototypes: proto_arr,
            edges,
            labels: Array1::from_vec(labels),
            quantization_error: total_qe / n_samples as f64,
        })
    }
}

// ---------------------------------------------------------------------------
// LVQ — Learning Vector Quantization
// ---------------------------------------------------------------------------

/// Configuration for LVQ training.
#[derive(Debug, Clone)]
pub struct LvqConfig {
    /// Number of prototypes per class (default 1).
    pub prototypes_per_class: usize,
    /// Initial learning rate (default 0.1).
    pub lr_init: f64,
    /// Final learning rate (default 0.001).
    pub lr_final: f64,
    /// Number of training epochs.
    pub max_epochs: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for LvqConfig {
    fn default() -> Self {
        Self {
            prototypes_per_class: 1,
            lr_init: 0.1,
            lr_final: 0.001,
            max_epochs: 50,
            seed: 42,
        }
    }
}

/// Result of LVQ training.
#[derive(Debug, Clone)]
pub struct LvqResult {
    /// Prototype weights, shape `(n_prototypes, n_features)`.
    pub prototypes: Array2<f64>,
    /// Class label for each prototype.
    pub prototype_labels: Vec<usize>,
    /// Training accuracy on the training set.
    pub train_accuracy: f64,
}

impl LvqResult {
    /// Predict the class of each row in `x`.
    pub fn predict(&self, x: ArrayView2<f64>) -> Vec<usize> {
        let n = x.shape()[0];
        let n_proto = self.prototypes.shape()[0];
        (0..n)
            .map(|i| {
                let row = x.row(i).to_vec();
                let best = (0..n_proto)
                    .map(|j| {
                        let p: Vec<f64> = self.prototypes.row(j).to_vec();
                        (j, sq_euclid(&row, &p))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
                    .unwrap_or(0);
                self.prototype_labels[best]
            })
            .collect()
    }
}

/// LVQ-1: Learning Vector Quantization.
///
/// Supervised prototype learning: attracts the nearest correct-class prototype
/// and repels the nearest wrong-class prototype toward/away from each input.
pub struct LVQ {
    /// Configuration.
    pub config: LvqConfig,
}

impl Default for LVQ {
    fn default() -> Self {
        Self {
            config: LvqConfig::default(),
        }
    }
}

impl LVQ {
    /// Create a new LVQ with the given config.
    pub fn new(config: LvqConfig) -> Self {
        Self { config }
    }

    /// Fit LVQ to labelled data.
    ///
    /// # Arguments
    /// * `x` – Feature matrix `(n_samples, n_features)`.
    /// * `y` – Class labels, values in `0..n_classes`.
    pub fn fit(&self, x: ArrayView2<f64>, y: &[usize]) -> Result<LvqResult> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if y.len() != n_samples {
            return Err(ClusteringError::InvalidInput(
                "y must have the same length as x rows".into(),
            ));
        }

        let n_classes = y.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        if n_classes == 0 {
            return Err(ClusteringError::InvalidInput("Empty class labels".into()));
        }

        let ppc = self.config.prototypes_per_class;
        let mut rng = self.config.seed;

        // Initialise prototypes by sampling from each class.
        let mut class_samples: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &label) in y.iter().enumerate() {
            if label < n_classes {
                class_samples[label].push(i);
            }
        }

        let mut proto_weights: Vec<Vec<f64>> = Vec::new();
        let mut proto_labels: Vec<usize> = Vec::new();

        for class in 0..n_classes {
            let samples = &class_samples[class];
            if samples.is_empty() {
                continue;
            }
            for _ in 0..ppc {
                let idx = samples[lcg_usize(&mut rng, samples.len())];
                proto_weights.push(x.row(idx).to_vec());
                proto_labels.push(class);
            }
        }

        let n_proto = proto_weights.len();
        if n_proto == 0 {
            return Err(ClusteringError::ComputationError(
                "No prototypes initialized".into(),
            ));
        }

        let total_steps = self.config.max_epochs * n_samples;

        // LVQ-1 training loop.
        for epoch in 0..self.config.max_epochs {
            // Shuffle.
            let mut order: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = lcg_usize(&mut rng, i + 1);
                order.swap(i, j);
            }

            for (step_in_epoch, &sample_idx) in order.iter().enumerate() {
                let step = epoch * n_samples + step_in_epoch;
                let t = step as f64 / total_steps.max(1) as f64;
                let lr = self.config.lr_init * (self.config.lr_final / self.config.lr_init).powf(t);

                let input = x.row(sample_idx).to_vec();
                let true_class = y[sample_idx];

                // Find nearest prototype.
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

        // Build result array.
        let mut proto_arr = Array2::<f64>::zeros((n_proto, n_features));
        for (j, w) in proto_weights.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = w[k];
            }
        }

        // Compute training accuracy.
        let predictions = {
            let n = n_samples;
            (0..n)
                .map(|i| {
                    let row = x.row(i).to_vec();
                    let best = (0..n_proto)
                        .map(|j| (j, sq_euclid(&row, &proto_weights[j])))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(j, _)| j)
                        .unwrap_or(0);
                    proto_labels[best]
                })
                .collect::<Vec<usize>>()
        };

        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        let train_accuracy = correct as f64 / n_samples as f64;

        Ok(LvqResult {
            prototypes: proto_arr,
            prototype_labels: proto_labels,
            train_accuracy,
        })
    }
}

// ---------------------------------------------------------------------------
// GLVQ — Generalized Learning Vector Quantization
// ---------------------------------------------------------------------------

/// Result of GLVQ training.
#[derive(Debug, Clone)]
pub struct GlvqResult {
    /// Prototype weights `(n_prototypes, n_features)`.
    pub prototypes: Array2<f64>,
    /// Class label for each prototype.
    pub prototype_labels: Vec<usize>,
    /// Training accuracy.
    pub train_accuracy: f64,
    /// Final GLVQ cost.
    pub cost: f64,
}

impl GlvqResult {
    /// Predict class labels for `x`.
    pub fn predict(&self, x: ArrayView2<f64>) -> Vec<usize> {
        let n = x.shape()[0];
        let n_proto = self.prototypes.shape()[0];
        (0..n)
            .map(|i| {
                let row = x.row(i).to_vec();
                let best = (0..n_proto)
                    .map(|j| {
                        let p = self.prototypes.row(j).to_vec();
                        (j, sq_euclid(&row, &p))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
                    .unwrap_or(0);
                self.prototype_labels[best]
            })
            .collect()
    }
}

/// Configuration for GLVQ.
#[derive(Debug, Clone)]
pub struct GlvqConfig {
    /// Prototypes per class (default 1).
    pub prototypes_per_class: usize,
    /// Learning rate (default 0.01).
    pub lr: f64,
    /// Sigmoid squashing steepness (default 1.0).
    pub sigma: f64,
    /// Number of training epochs.
    pub max_epochs: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GlvqConfig {
    fn default() -> Self {
        Self {
            prototypes_per_class: 1,
            lr: 0.01,
            sigma: 1.0,
            max_epochs: 100,
            seed: 42,
        }
    }
}

/// Generalized LVQ (GLVQ) — soft-margin prototype learning.
///
/// Minimises a differentiable cost function based on the relative distances
/// to the nearest correct (d+) and nearest incorrect (d-) prototypes:
///
///   μ(x) = (d+ - d-) / (d+ + d-)
///
/// The sigmoid of μ is minimised.  Gradients are computed w.r.t. both d+
/// and d- prototypes.
///
/// Reference: Sato & Yamada, 1996.
pub struct GLVQ {
    /// Configuration.
    pub config: GlvqConfig,
}

impl Default for GLVQ {
    fn default() -> Self {
        Self {
            config: GlvqConfig::default(),
        }
    }
}

impl GLVQ {
    /// Create a new GLVQ with the given config.
    pub fn new(config: GlvqConfig) -> Self {
        Self { config }
    }

    /// Fit GLVQ to labelled data.
    pub fn fit(&self, x: ArrayView2<f64>, y: &[usize]) -> Result<GlvqResult> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }
        if y.len() != n_samples {
            return Err(ClusteringError::InvalidInput("y length mismatch".into()));
        }

        let n_classes = y.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        if n_classes < 2 {
            return Err(ClusteringError::InvalidInput(
                "GLVQ requires at least 2 classes".into(),
            ));
        }

        let ppc = self.config.prototypes_per_class;
        let mut rng = self.config.seed;

        // Initialise prototypes from class samples.
        let mut class_samples: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &label) in y.iter().enumerate() {
            if label < n_classes {
                class_samples[label].push(i);
            }
        }

        let mut proto_weights: Vec<Vec<f64>> = Vec::new();
        let mut proto_labels: Vec<usize> = Vec::new();

        for class in 0..n_classes {
            let samples = &class_samples[class];
            if samples.is_empty() {
                continue;
            }
            for _ in 0..ppc {
                let idx = samples[lcg_usize(&mut rng, samples.len())];
                proto_weights.push(x.row(idx).to_vec());
                proto_labels.push(class);
            }
        }

        let n_proto = proto_weights.len();
        let lr = self.config.lr;
        let sigma = self.config.sigma;

        let mut total_cost = 0.0f64;

        // GLVQ training loop.
        for _epoch in 0..self.config.max_epochs {
            // Shuffle.
            let mut order: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = lcg_usize(&mut rng, i + 1);
                order.swap(i, j);
            }

            total_cost = 0.0;
            for &sample_idx in &order {
                let input = x.row(sample_idx).to_vec();
                let true_class = y[sample_idx];

                // Find nearest same-class prototype (winner+) and nearest other-class prototype (winner-).
                let mut d_plus = f64::INFINITY;
                let mut d_minus = f64::INFINITY;
                let mut winner_plus = 0usize;
                let mut winner_minus = 0usize;

                for j in 0..n_proto {
                    let d = sq_euclid(&input, &proto_weights[j]);
                    if proto_labels[j] == true_class {
                        if d < d_plus {
                            d_plus = d;
                            winner_plus = j;
                        }
                    } else if d < d_minus {
                        d_minus = d;
                        winner_minus = j;
                    }
                }

                if d_plus.is_infinite() || d_minus.is_infinite() {
                    continue;
                }

                let denom = d_plus + d_minus;
                if denom < 1e-12 {
                    continue;
                }

                let mu = (d_plus - d_minus) / denom;
                // Sigmoid activation: f(mu) = 1 / (1 + exp(-sigma * mu))
                let f_mu = 1.0 / (1.0 + (-sigma * mu).exp());
                // Derivative: f'(mu) = sigma * f(mu) * (1 - f(mu))
                let f_prime = sigma * f_mu * (1.0 - f_mu);

                total_cost += f_mu;

                // Gradient w.r.t. d+:  f'(mu) * 2 * d- / denom^2
                let grad_dp = f_prime * (2.0 * d_minus) / (denom * denom);
                // Gradient w.r.t. d-: -f'(mu) * 2 * d+ / denom^2
                let grad_dm = -f_prime * (2.0 * d_plus) / (denom * denom);

                // Update winner+: gradient descent w.r.t. d+ = ||x - w+||^2
                // dL/dw+ = 2 * grad_dp * (w+ - x)
                let wp = &mut proto_weights[winner_plus];
                for k in 0..n_features {
                    wp[k] -= lr * 2.0 * grad_dp * (wp[k] - input[k]);
                }

                // Update winner-: gradient descent w.r.t. d- = ||x - w-||^2
                // dL/dw- = 2 * grad_dm * (w- - x)
                let wm = &mut proto_weights[winner_minus];
                for k in 0..n_features {
                    wm[k] -= lr * 2.0 * grad_dm * (wm[k] - input[k]);
                }
            }
        }

        // Build output.
        let mut proto_arr = Array2::<f64>::zeros((n_proto, n_features));
        for (j, w) in proto_weights.iter().enumerate() {
            for k in 0..n_features {
                proto_arr[[j, k]] = w[k];
            }
        }

        // Compute training accuracy.
        let mut correct = 0usize;
        for i in 0..n_samples {
            let row = x.row(i).to_vec();
            let best = (0..n_proto)
                .map(|j| (j, sq_euclid(&row, &proto_weights[j])))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            if proto_labels[best] == y[i] {
                correct += 1;
            }
        }
        let train_accuracy = correct as f64 / n_samples as f64;

        Ok(GlvqResult {
            prototypes: proto_arr,
            prototype_labels: proto_labels,
            train_accuracy,
            cost: total_cost,
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

    fn two_cluster_data() -> (Array2<f64>, Vec<usize>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0, 0.2, 5.0, 5.0, 5.1, 5.0,
                5.0, 5.1, 5.2, 5.0, 5.1, 5.1, 5.0, 5.2,
            ],
        )
        .expect("valid shape");
        let y = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_neural_gas_basic() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        let result = ng.fit(x.view(), 2, 20).expect("neural gas fit");
        assert_eq!(result.n_units, 2);
        assert_eq!(result.labels.len(), 12);
        assert!(result.quantization_error >= 0.0);
    }

    #[test]
    fn test_neural_gas_n_units_gt_samples() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        // More units than well-separated samples still works.
        let result = ng.fit(x.view(), 5, 10).expect("ng many units");
        assert_eq!(result.n_units, 5);
    }

    #[test]
    fn test_neural_gas_single_unit() {
        let (x, _) = two_cluster_data();
        let ng = NeuralGas::default();
        let result = ng.fit(x.view(), 1, 10).expect("ng 1 unit");
        assert_eq!(result.n_units, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_growing_neural_gas_basic() {
        let (x, _) = two_cluster_data();
        let config = GngConfig {
            max_steps: 200,
            insert_interval: 20,
            max_units: 10,
            seed: 7,
            ..GngConfig::default()
        };
        let gng = GrowingNeuralGas::new(config);
        let result = gng.fit(x.view()).expect("gng fit");
        assert!(result.prototypes.shape()[0] >= 2, "should have grown");
        assert_eq!(result.labels.len(), 12);
    }

    #[test]
    fn test_lvq_two_classes() {
        let (x, y) = two_cluster_data();
        let config = LvqConfig {
            prototypes_per_class: 1,
            lr_init: 0.3,
            lr_final: 0.01,
            max_epochs: 100,
            seed: 42,
        };
        let lvq = LVQ::new(config);
        let result = lvq.fit(x.view(), &y).expect("lvq fit");
        assert_eq!(result.prototypes.shape()[0], 2); // 1 per class × 2 classes
                                                     // Well-separated data should give high accuracy.
        assert!(
            result.train_accuracy > 0.8,
            "expected > 80% accuracy, got {}",
            result.train_accuracy
        );
    }

    #[test]
    fn test_lvq_predict() {
        let (x, y) = two_cluster_data();
        let lvq = LVQ::default();
        let result = lvq.fit(x.view(), &y).expect("lvq fit");
        let preds = result.predict(x.view());
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_glvq_two_classes() {
        let (x, y) = two_cluster_data();
        let config = GlvqConfig {
            prototypes_per_class: 1,
            lr: 0.05,
            sigma: 1.0,
            max_epochs: 200,
            seed: 42,
        };
        let glvq = GLVQ::new(config);
        let result = glvq.fit(x.view(), &y).expect("glvq fit");
        assert_eq!(result.prototypes.shape()[0], 2);
        assert!(
            result.train_accuracy > 0.8,
            "expected > 80% accuracy, got {}",
            result.train_accuracy
        );
    }

    #[test]
    fn test_glvq_predict() {
        let (x, y) = two_cluster_data();
        let glvq = GLVQ::default();
        let result = glvq.fit(x.view(), &y).expect("glvq fit");
        let preds = result.predict(x.view());
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_lvq_invalid_input() {
        let (x, _y) = two_cluster_data();
        let lvq = LVQ::default();
        // Wrong y length.
        assert!(lvq.fit(x.view(), &[0, 1, 0]).is_err());
    }

    #[test]
    fn test_glvq_single_class_error() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.3, 0.1])
            .expect("shape");
        let y = vec![0usize, 0, 0, 0];
        let glvq = GLVQ::default();
        assert!(glvq.fit(x.view(), &y).is_err(), "single class should error");
    }
}
