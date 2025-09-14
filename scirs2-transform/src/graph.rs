//! Graph embedding transformers for graph-based feature extraction
//!
//! This module provides utilities for transforming graph structures into
//! numerical feature representations suitable for machine learning.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::Rng;

use crate::error::{Result, TransformError};
use scirs2_linalg::eigh;
use statrs::statistics::Statistics;

/// Node embeddings using spectral decomposition of graph Laplacian
pub struct SpectralEmbedding {
    /// Number of embedding dimensions
    _ncomponents: usize,
    /// Type of Laplacian to use
    laplacian_type: LaplacianType,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

/// Types of graph Laplacian
#[derive(Clone, Copy, Debug)]
pub enum LaplacianType {
    /// Unnormalized Laplacian: L = D - A
    Unnormalized,
    /// Normalized symmetric Laplacian: L = I - D^(-1/2) A D^(-1/2)
    NormalizedSymmetric,
    /// Random walk normalized Laplacian: L = I - D^(-1) A
    NormalizedRandomWalk,
}

impl SpectralEmbedding {
    /// Create a new spectral embedding transformer
    pub fn new(_ncomponents: usize) -> Self {
        SpectralEmbedding {
            _ncomponents,
            laplacian_type: LaplacianType::NormalizedSymmetric,
            random_state: None,
        }
    }

    /// Set the type of Laplacian to use
    pub fn with_laplacian_type(mut self, laplaciantype: LaplacianType) -> Self {
        self.laplacian_type = laplaciantype;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute the graph Laplacian matrix
    fn compute_laplacian(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let n = adjacency.shape()[0];
        if adjacency.shape()[1] != n {
            return Err(TransformError::InvalidInput(
                "Adjacency matrix must be square".into(),
            ));
        }

        // Compute degree matrix
        let degrees: Array1<f64> = adjacency.sum_axis(ndarray::Axis(1));

        match self.laplacian_type {
            LaplacianType::Unnormalized => {
                // L = D - A
                let mut laplacian = -adjacency.clone();
                for i in 0..n {
                    laplacian[[i, i]] += degrees[i];
                }
                Ok(laplacian)
            }
            LaplacianType::NormalizedSymmetric => {
                // L = I - D^(-1/2) A D^(-1/2)
                let mut laplacian = Array2::eye(n);
                let d_sqrt_inv = degrees.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

                for i in 0..n {
                    for j in 0..n {
                        if adjacency[[i, j]] != 0.0 {
                            laplacian[[i, j]] -= d_sqrt_inv[i] * adjacency[[i, j]] * d_sqrt_inv[j];
                        }
                    }
                }
                Ok(laplacian)
            }
            LaplacianType::NormalizedRandomWalk => {
                // L = I - D^(-1) A
                let mut laplacian = Array2::eye(n);
                let d_inv = degrees.mapv(|d| if d > 0.0 { 1.0 / d } else { 0.0 });

                for i in 0..n {
                    for j in 0..n {
                        if adjacency[[i, j]] != 0.0 {
                            laplacian[[i, j]] -= d_inv[i] * adjacency[[i, j]];
                        }
                    }
                }
                Ok(laplacian)
            }
        }
    }

    /// Fit and transform the adjacency matrix to node embeddings
    pub fn fit_transform(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let laplacian = self.compute_laplacian(adjacency)?;

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = eigh(&laplacian.view(), None).map_err(|e| {
            TransformError::ComputationError(format!("Eigendecomposition failed: {e}"))
        })?;

        // Select the smallest _ncomponents eigenvalues (excluding zero)
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
            .iter()
            .zip(eigenvectors.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        // Sort by eigenvalue
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Skip the first eigenvalue if it's (approximately) zero
        let start_idx = if eigen_pairs[0].0.abs() < 1e-10 { 1 } else { 0 };
        let end_idx = (start_idx + self._ncomponents).min(eigen_pairs.len());

        // Build embedding matrix
        let nnodes = adjacency.shape()[0];
        let actual_components = end_idx - start_idx;
        let mut embedding = Array2::zeros((nnodes, actual_components));

        for (col_idx, idx) in (start_idx..end_idx).enumerate() {
            embedding.column_mut(col_idx).assign(&eigen_pairs[idx].1);
        }

        Ok(embedding)
    }
}

/// DeepWalk: Random walk-based graph embedding
pub struct DeepWalk {
    /// Embedding dimension
    _embeddingdim: usize,
    /// Number of random walks per node
    n_walks: usize,
    /// Length of each random walk
    walk_length: usize,
    /// Context window size
    window_size: usize,
    /// Number of negative samples
    negative_samples: usize,
    /// Learning rate
    learning_rate: f64,
    /// Number of training epochs
    n_epochs: usize,
    /// Random state
    random_state: Option<u64>,
}

impl DeepWalk {
    /// Create a new DeepWalk transformer
    pub fn new(_embeddingdim: usize) -> Self {
        DeepWalk {
            _embeddingdim,
            n_walks: 10,
            walk_length: 80,
            window_size: 5,
            negative_samples: 5,
            learning_rate: 0.025,
            n_epochs: 1,
            random_state: None,
        }
    }

    /// Set the number of walks per node
    pub fn with_n_walks(mut self, nwalks: usize) -> Self {
        self.n_walks = nwalks;
        self
    }

    /// Set the walk length
    pub fn with_walk_length(mut self, walklength: usize) -> Self {
        self.walk_length = walklength;
        self
    }

    /// Set the window size
    pub fn with_window_size(mut self, windowsize: usize) -> Self {
        self.window_size = windowsize;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate random walks from the graph
    fn generate_walks(&self, adjacency: &Array2<f64>) -> Vec<Vec<usize>> {
        let nnodes = adjacency.shape()[0];
        let mut walks = Vec::with_capacity(nnodes * self.n_walks);

        let mut rng = rand::rng();

        // Build neighbor lists for efficient sampling
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nnodes];
        for i in 0..nnodes {
            for j in 0..nnodes {
                if adjacency[[i, j]] > 0.0 {
                    neighbors[i].push(j);
                }
            }
        }

        // Generate walks
        for start_node in 0..nnodes {
            for _ in 0..self.n_walks {
                let mut walk = vec![start_node];
                let mut current = start_node;

                for _ in 1..self.walk_length {
                    let node_neighbors = &neighbors[current];
                    if node_neighbors.is_empty() {
                        break;
                    }

                    // Randomly select next node
                    let next_idx = rng.gen_range(0..node_neighbors.len());
                    current = node_neighbors[next_idx];
                    walk.push(current);
                }

                if walk.len() > 1 {
                    walks.push(walk);
                }
            }
        }

        walks
    }

    /// Train embeddings using Skip-gram with negative sampling
    fn train_embeddings(&self, walks: &[Vec<usize>], nnodes: usize) -> Array2<f64> {
        let mut rng = rand::rng();

        // Initialize embeddings randomly
        let mut embeddings = Array2::zeros((nnodes, self._embeddingdim));
        for i in 0..nnodes {
            for j in 0..self._embeddingdim {
                embeddings[[i, j]] = rng.gen_range(-0.5..0.5) / self._embeddingdim as f64;
            }
        }

        let mut context_embeddings = embeddings.clone();

        // Training loop
        for epoch in 0..self.n_epochs {
            let mut _total_loss = 0.0;
            let lr = self.learning_rate * (1.0 - epoch as f64 / self.n_epochs as f64);

            for walk in walks {
                for (idx, &center) in walk.iter().enumerate() {
                    // Define context window
                    let window_start = idx.saturating_sub(self.window_size);
                    let window_end = (idx + self.window_size + 1).min(walk.len());

                    #[allow(clippy::needless_range_loop)]
                    for context_idx in window_start..window_end {
                        if context_idx == idx {
                            continue;
                        }

                        let context = walk[context_idx];

                        // Positive sample
                        let center_vec = embeddings.row(center).to_owned();
                        let context_vec = context_embeddings.row(context).to_owned();
                        let dot_product = center_vec.dot(&context_vec);
                        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());

                        // Gradient for positive sample
                        let grad_coef = lr * (1.0 - sigmoid);
                        let center_grad = &context_vec * grad_coef;
                        let context_grad = &center_vec * grad_coef;

                        // Update embeddings
                        embeddings.row_mut(center).scaled_add(1.0, &center_grad);
                        context_embeddings
                            .row_mut(context)
                            .scaled_add(1.0, &context_grad);

                        _total_loss += -(sigmoid.ln());

                        // Negative samples
                        for _ in 0..self.negative_samples {
                            let negative = rng.gen_range(0..nnodes);
                            if negative == context {
                                continue;
                            }

                            let neg_vec = context_embeddings.row(negative).to_owned();
                            let neg_dot = center_vec.dot(&neg_vec);
                            let neg_sigmoid = 1.0 / (1.0 + (-neg_dot).exp());

                            // Gradient for negative sample
                            let neg_grad_coef = -lr * neg_sigmoid;
                            let center_neg_grad = &neg_vec * neg_grad_coef;
                            let neg_grad = &center_vec * neg_grad_coef;

                            // Update embeddings
                            embeddings.row_mut(center).scaled_add(1.0, &center_neg_grad);
                            context_embeddings
                                .row_mut(negative)
                                .scaled_add(1.0, &neg_grad);

                            _total_loss += -((1.0 - neg_sigmoid).ln());
                        }
                    }
                }
            }
        }

        embeddings
    }

    /// Fit and transform the adjacency matrix to node embeddings
    pub fn fit_transform(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let walks = self.generate_walks(adjacency);
        if walks.is_empty() {
            return Err(TransformError::InvalidInput(
                "No valid walks generated".into(),
            ));
        }

        let embeddings = self.train_embeddings(&walks, adjacency.shape()[0]);
        Ok(embeddings)
    }
}

/// Node2Vec: Biased random walk-based graph embedding
pub struct Node2Vec {
    /// Base DeepWalk model
    base_model: DeepWalk,
    /// Return parameter (p)
    p: f64,
    /// In-out parameter (q)
    q: f64,
}

impl Node2Vec {
    /// Create a new Node2Vec transformer
    pub fn new(_embeddingdim: usize, p: f64, q: f64) -> Self {
        Node2Vec {
            base_model: DeepWalk::new(_embeddingdim),
            p,
            q,
        }
    }

    /// Configure the base DeepWalk model
    pub fn configure_base<F>(mut self, f: F) -> Self
    where
        F: FnOnce(DeepWalk) -> DeepWalk,
    {
        self.base_model = f(self.base_model);
        self
    }

    /// Generate biased random walks
    fn generate_biased_walks(&self, adjacency: &Array2<f64>) -> Vec<Vec<usize>> {
        let nnodes = adjacency.shape()[0];
        let mut walks = Vec::with_capacity(nnodes * self.base_model.n_walks);

        let mut rng = if let Some(seed) = self.base_model.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random::<u64>())
        };

        // Build neighbor lists
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nnodes];
        for i in 0..nnodes {
            for j in 0..nnodes {
                if adjacency[[i, j]] > 0.0 {
                    neighbors[i].push(j);
                }
            }
        }

        // Generate walks
        for start_node in 0..nnodes {
            for _ in 0..self.base_model.n_walks {
                let mut walk = vec![start_node];

                if neighbors[start_node].is_empty() {
                    continue;
                }

                // First step: uniform random
                let first_step =
                    neighbors[start_node][rng.gen_range(0..neighbors[start_node].len())];
                walk.push(first_step);

                // Subsequent steps: biased by p and q
                for _ in 2..self.base_model.walk_length {
                    let current = *walk.last().unwrap();
                    let prev = walk[walk.len() - 2];

                    let current_neighbors = &neighbors[current];
                    if current_neighbors.is_empty() {
                        break;
                    }

                    // Calculate transition probabilities
                    let mut probs = Vec::with_capacity(current_neighbors.len());
                    let mut total_prob = 0.0;

                    for &next in current_neighbors {
                        let prob = if next == prev {
                            1.0 / self.p // Return to previous node
                        } else if adjacency[[next, prev]] > 0.0 {
                            1.0 // Move to neighbor of previous node
                        } else {
                            1.0 / self.q // Move to non-neighbor of previous node
                        };

                        probs.push(prob);
                        total_prob += prob;
                    }

                    // Normalize probabilities
                    for p in &mut probs {
                        *p /= total_prob;
                    }

                    // Sample next node
                    let rand_val: f64 = rng.random();
                    let mut cumsum = 0.0;
                    let mut next_node = current_neighbors[0];

                    for (idx, &prob) in probs.iter().enumerate() {
                        cumsum += prob;
                        if rand_val <= cumsum {
                            next_node = current_neighbors[idx];
                            break;
                        }
                    }

                    walk.push(next_node);
                }

                if walk.len() > 1 {
                    walks.push(walk);
                }
            }
        }

        walks
    }

    /// Fit and transform the adjacency matrix to node embeddings
    pub fn fit_transform(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let walks = self.generate_biased_walks(adjacency);
        if walks.is_empty() {
            return Err(TransformError::InvalidInput(
                "No valid walks generated".into(),
            ));
        }

        let embeddings = self
            .base_model
            .train_embeddings(&walks, adjacency.shape()[0]);
        Ok(embeddings)
    }
}

/// Graph autoencoder for unsupervised graph embedding
pub struct GraphAutoencoder {
    /// Encoder dimensions (including input dimension)
    _encoderdims: Vec<usize>,
    /// Activation function
    activation: ActivationType,
    /// Learning rate
    learning_rate: f64,
    /// Number of epochs
    n_epochs: usize,
}

/// Activation function types
#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
}

impl GraphAutoencoder {
    /// Create a new graph autoencoder
    pub fn new(_encoderdims: Vec<usize>) -> Self {
        GraphAutoencoder {
            _encoderdims,
            activation: ActivationType::ReLU,
            learning_rate: 0.01,
            n_epochs: 200,
        }
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs
    pub fn with_n_epochs(mut self, nepochs: usize) -> Self {
        self.n_epochs = nepochs;
        self
    }

    /// Apply activation function
    fn activate(&self, x: &Array2<f64>) -> Array2<f64> {
        match self.activation {
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
        }
    }

    /// Compute activation derivative
    fn activate_derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        match self.activation {
            ActivationType::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationType::Sigmoid => {
                let sig = self.activate(x);
                &sig * &(1.0 - &sig)
            }
            ActivationType::Tanh => {
                let tanh = x.mapv(|v| v.tanh());
                1.0 - &tanh * &tanh
            }
        }
    }

    /// Fit and transform adjacency matrix to embeddings
    pub fn fit_transform(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let nnodes = adjacency.shape()[0];

        if self._encoderdims.is_empty() || self._encoderdims[0] != nnodes {
            return Err(TransformError::InvalidInput(format!(
                "First encoder dimension must match number of nodes ({nnodes})"
            )));
        }

        let mut rng = rand::rng();

        // Initialize weights
        let mut encoder_weights = Vec::new();
        for i in 0..self._encoderdims.len() - 1 {
            let (n_in, n_out) = (self._encoderdims[i], self._encoderdims[i + 1]);
            let scale = (2.0 / n_in as f64).sqrt();
            let mut w = Array2::zeros((n_in, n_out));

            for j in 0..n_in {
                for k in 0..n_out {
                    w[[j, k]] = rng.gen_range(-scale..scale);
                }
            }
            encoder_weights.push(w);
        }

        // Decoder weights (transpose of encoder)
        let mut decoder_weights: Vec<Array2<f64>> = encoder_weights
            .iter()
            .rev()
            .map(|w| w.t().to_owned())
            .collect();

        // Training loop
        let features = adjacency.clone();

        for _epoch in 0..self.n_epochs {
            // Forward pass - encoding
            let mut activations = vec![features.clone()];
            let mut z = features.clone();

            for (i, w) in encoder_weights.iter().enumerate() {
                z = z.dot(w);
                if i < encoder_weights.len() - 1 {
                    z = self.activate(&z);
                }
                activations.push(z.clone());
            }

            // Get embeddings (bottleneck layer)
            let _embeddings = z.clone();

            // Forward pass - decoding
            for (i, w) in decoder_weights.iter().enumerate() {
                z = z.dot(w);
                if i < decoder_weights.len() - 1 {
                    z = self.activate(&z);
                }
            }

            // Reconstruction (sigmoid for adjacency reconstruction)
            let reconstruction = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));

            // Compute loss (binary cross-entropy)
            let loss = -adjacency * &reconstruction.mapv(|v| (v + 1e-8).ln())
                - (1.0 - adjacency) * &reconstruction.mapv(|v| (1.0 - v + 1e-8).ln());
            let _avg_loss = loss.mean();

            // Backward pass
            let mut delta = &reconstruction - adjacency;
            delta *= self.learning_rate;

            // Update decoder weights
            let decoder_len = decoder_weights.len();
            for (i, w) in decoder_weights.iter_mut().rev().enumerate() {
                let layer_idx = activations.len() - 2 - i;
                let grad = activations[layer_idx].t().dot(&delta);
                *w -= &grad;

                if i < decoder_len - 1 {
                    delta = delta.dot(&w.t());
                    delta *= &self.activate_derivative(&activations[layer_idx]);
                }
            }

            // Update encoder weights
            let encoder_len = encoder_weights.len();
            for (i, w) in encoder_weights.iter_mut().enumerate() {
                let grad = activations[i].t().dot(&delta);
                *w -= &grad;

                if i < encoder_len - 1 {
                    delta = delta.dot(&w.t());
                    delta *= &self.activate_derivative(&activations[i]);
                }
            }
        }

        // Final forward pass to get embeddings
        let mut z = features;
        for (i, w) in encoder_weights.iter().enumerate() {
            z = z.dot(w);
            if i < encoder_weights.len() - 1 {
                z = self.activate(&z);
            }
        }

        Ok(z)
    }
}

/// Convert edge list to adjacency matrix
#[allow(dead_code)]
pub fn edge_list_to_adjacency(edges: &[(usize, usize)], nnodes: Option<usize>) -> Array2<f64> {
    let max_node = edges
        .iter()
        .flat_map(|(u, v)| vec![*u, *v])
        .max()
        .unwrap_or(0);

    let n = nnodes.unwrap_or(max_node + 1);
    let mut adjacency = Array2::zeros((n, n));

    for &(u, v) in edges {
        if u < n && v < n {
            adjacency[[u, v]] = 1.0;
            adjacency[[v, u]] = 1.0; // Assuming undirected graph
        }
    }

    adjacency
}

/// Convert adjacency matrix to edge list
#[allow(dead_code)]
pub fn adjacency_to_edge_list(adjacency: &Array2<f64>) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let n = adjacency.shape()[0];

    for i in 0..n {
        for j in i + 1..n {
            // Only upper triangle for undirected graphs
            if adjacency[[i, j]] > 0.0 {
                edges.push((i, j));
            }
        }
    }

    edges
}
