//! Streaming transformations for continuous data processing
//!
//! This module provides utilities for processing data streams in real-time,
//! maintaining running statistics and transforming data incrementally.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::eigh;
use std::collections::VecDeque;

use crate::error::{Result, TransformError};

/// Trait for transformers that support streaming/incremental updates
pub trait StreamingTransformer: Send + Sync {
    /// Update the transformer with a new batch of data
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()>;

    /// Transform a batch of data using current statistics
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Reset the transformer to initial state
    fn reset(&mut self);

    /// Get the number of samples seen so far
    fn n_samples_seen(&self) -> usize;
}

/// Streaming standard scaler that maintains running statistics
pub struct StreamingStandardScaler {
    /// Running mean for each feature
    mean: Array1<f64>,
    /// Running variance for each feature
    variance: Array1<f64>,
    /// Number of samples seen
    n_samples: usize,
    /// Whether to center the data
    with_mean: bool,
    /// Whether to scale to unit variance
    withstd: bool,
    /// Epsilon for numerical stability
    epsilon: f64,
}

impl StreamingStandardScaler {
    /// Create a new streaming standard scaler
    pub fn new(_nfeatures: usize, with_mean: bool, withstd: bool) -> Self {
        StreamingStandardScaler {
            mean: Array1::zeros(_nfeatures),
            variance: Array1::zeros(_nfeatures),
            n_samples: 0,
            with_mean,
            withstd,
            epsilon: 1e-8,
        }
    }

    /// Update statistics using Welford's online algorithm
    fn update_statistics(&mut self, x: &Array2<f64>) {
        let batch_size = x.shape()[0];
        let nfeatures = x.shape()[1];

        for i in 0..batch_size {
            self.n_samples += 1;
            let n = self.n_samples as f64;

            for j in 0..nfeatures {
                let value = x[[i, j]];
                let delta = value - self.mean[j];
                self.mean[j] += delta / n;

                if self.withstd {
                    let delta2 = value - self.mean[j];
                    self.variance[j] += delta * delta2;
                }
            }
        }
    }

    /// Get the current standard deviation
    fn get_std(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            Array1::ones(self.mean.len())
        } else {
            self.variance
                .mapv(|v| (v / (self.n_samples - 1) as f64).sqrt().max(self.epsilon))
        }
    }
}

impl StreamingTransformer for StreamingStandardScaler {
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.mean.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.mean.len(),
                x.shape()[1]
            )));
        }

        self.update_statistics(x);
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.shape()[1] != self.mean.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.mean.len(),
                x.shape()[1]
            )));
        }

        let mut result = x.to_owned();

        if self.with_mean {
            for i in 0..result.shape()[0] {
                for j in 0..result.shape()[1] {
                    result[[i, j]] -= self.mean[j];
                }
            }
        }

        if self.withstd {
            let std = self.get_std();
            for i in 0..result.shape()[0] {
                for j in 0..result.shape()[1] {
                    result[[i, j]] /= std[j];
                }
            }
        }

        Ok(result)
    }

    fn reset(&mut self) {
        self.mean.fill(0.0);
        self.variance.fill(0.0);
        self.n_samples = 0;
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }
}

/// Streaming min-max scaler that tracks min and max values
pub struct StreamingMinMaxScaler {
    /// Minimum values for each feature
    min: Array1<f64>,
    /// Maximum values for each feature
    max: Array1<f64>,
    /// Target range
    featurerange: (f64, f64),
    /// Number of samples seen
    n_samples: usize,
}

impl StreamingMinMaxScaler {
    /// Create a new streaming min-max scaler
    pub fn new(_nfeatures: usize, featurerange: (f64, f64)) -> Self {
        StreamingMinMaxScaler {
            min: Array1::from_elem(_nfeatures, f64::INFINITY),
            max: Array1::from_elem(_nfeatures, f64::NEG_INFINITY),
            featurerange,
            n_samples: 0,
        }
    }

    /// Update min and max values
    fn update_bounds(&mut self, x: &Array2<f64>) {
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let value = x[[i, j]];
                self.min[j] = self.min[j].min(value);
                self.max[j] = self.max[j].max(value);
            }
            self.n_samples += 1;
        }
    }
}

impl StreamingTransformer for StreamingMinMaxScaler {
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.min.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.min.len(),
                x.shape()[1]
            )));
        }

        self.update_bounds(x);
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.shape()[1] != self.min.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.min.len(),
                x.shape()[1]
            )));
        }

        let mut result = Array2::zeros((x.nrows(), x.ncols()));
        let (min_val, max_val) = self.featurerange;
        let scale = max_val - min_val;

        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let range = self.max[j] - self.min[j];
                if range > 1e-10 {
                    result[[i, j]] = (x[[i, j]] - self.min[j]) / range * scale + min_val;
                } else {
                    result[[i, j]] = (min_val + max_val) / 2.0;
                }
            }
        }

        Ok(result)
    }

    fn reset(&mut self) {
        self.min.fill(f64::INFINITY);
        self.max.fill(f64::NEG_INFINITY);
        self.n_samples = 0;
    }

    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }
}

/// Streaming quantile tracker using P² algorithm
pub struct StreamingQuantileTracker {
    /// Quantiles to track
    quantiles: Vec<f64>,
    /// P² algorithm state for each feature and quantile
    p2_states: Vec<Vec<P2State>>,
    /// Number of features
    nfeatures: usize,
}

/// P² algorithm state for a single quantile
struct P2State {
    /// Marker positions
    n: [f64; 5],
    /// Marker values
    q: [f64; 5],
    /// Desired marker positions
    n_prime: [f64; 5],
    /// Number of observations
    count: usize,
    /// Target quantile
    p: f64,
}

impl P2State {
    fn new(p: f64) -> Self {
        P2State {
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            q: [0.0; 5],
            n_prime: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            count: 0,
            p,
        }
    }

    fn update(&mut self, value: f64) {
        if self.count < 5 {
            self.q[self.count] = value;
            self.count += 1;

            if self.count == 5 {
                // Sort initial observations
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            return;
        }

        // Find cell k such that q[k] <= value < q[k+1]
        let mut k = 0;
        for i in 1..5 {
            if value < self.q[i] {
                k = i - 1;
                break;
            }
        }
        if value >= self.q[4] {
            k = 3;
        }

        // Update marker positions
        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }

        // Update desired marker positions
        for i in 0..5 {
            self.n_prime[i] += match i {
                0 => 0.0,
                1 => self.p / 2.0,
                2 => self.p,
                3 => (1.0 + self.p) / 2.0,
                4 => 1.0,
                _ => unreachable!(),
            };
        }

        // Adjust marker values
        for i in 1..4 {
            let d = self.n_prime[i] - self.n[i];

            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                let d_sign = d.signum();

                // Try parabolic interpolation
                let qi = self.parabolic_interpolation(i, d_sign);

                if self.q[i - 1] < qi && qi < self.q[i + 1] {
                    self.q[i] = qi;
                } else {
                    // Fall back to linear interpolation
                    self.q[i] = self.linear_interpolation(i, d_sign);
                }

                self.n[i] += d_sign;
            }
        }
    }

    fn parabolic_interpolation(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qim1 = self.q[i - 1];
        let qip1 = self.q[i + 1];
        let ni = self.n[i];
        let nim1 = self.n[i - 1];
        let nip1 = self.n[i + 1];

        qi + d / (nip1 - nim1)
            * ((ni - nim1 + d) * (qip1 - qi) / (nip1 - ni)
                + (nip1 - ni - d) * (qi - qim1) / (ni - nim1))
    }

    fn linear_interpolation(&self, i: usize, d: f64) -> f64 {
        let j = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i])
    }

    fn quantile(&self) -> f64 {
        if self.count < 5 {
            // Not enough data, return median of available values
            let mut sorted = self.q[..self.count].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        } else {
            self.q[2] // Middle marker estimates the quantile
        }
    }
}

impl StreamingQuantileTracker {
    /// Create a new streaming quantile tracker
    pub fn new(nfeatures: usize, quantiles: Vec<f64>) -> Result<Self> {
        // Validate quantiles
        for &q in &quantiles {
            if !(0.0..=1.0).contains(&q) {
                return Err(TransformError::InvalidInput(format!(
                    "Quantile {q} must be between 0 and 1"
                )));
            }
        }

        let mut p2_states = Vec::with_capacity(nfeatures);
        for _ in 0..nfeatures {
            let feature_states: Vec<P2State> = quantiles.iter().map(|&q| P2State::new(q)).collect();
            p2_states.push(feature_states);
        }

        Ok(StreamingQuantileTracker {
            quantiles,
            p2_states,
            nfeatures,
        })
    }

    /// Update quantile estimates with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let value = x[[i, j]];
                for k in 0..self.quantiles.len() {
                    self.p2_states[j][k].update(value);
                }
            }
        }

        Ok(())
    }

    /// Get current quantile estimates
    pub fn get_quantiles(&self) -> Array2<f64> {
        let mut result = Array2::zeros((self.nfeatures, self.quantiles.len()));

        for j in 0..self.nfeatures {
            for k in 0..self.quantiles.len() {
                result[[j, k]] = self.p2_states[j][k].quantile();
            }
        }

        result
    }
}

/// Window-based streaming transformer that maintains a sliding window
pub struct WindowedStreamingTransformer<T: StreamingTransformer> {
    /// Underlying transformer
    transformer: T,
    /// Sliding window of recent data
    window: VecDeque<Array2<f64>>,
    /// Maximum window size
    windowsize: usize,
    /// Current number of samples in window
    current_size: usize,
}

impl<T: StreamingTransformer> WindowedStreamingTransformer<T> {
    /// Create a new windowed streaming transformer
    pub fn new(_transformer: T, windowsize: usize) -> Self {
        WindowedStreamingTransformer {
            transformer: _transformer,
            window: VecDeque::with_capacity(windowsize),
            windowsize,
            current_size: 0,
        }
    }

    /// Update the transformer with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        // Add new data to window
        self.window.push_back(x.to_owned());
        self.current_size += x.shape()[0];

        // Remove old data if window is full
        while self.current_size > self.windowsize && !self.window.is_empty() {
            if let Some(old_data) = self.window.pop_front() {
                self.current_size -= old_data.shape()[0];
            }
        }

        // Refit transformer on window data
        self.transformer.reset();
        for data in &self.window {
            self.transformer.partial_fit(data)?;
        }

        Ok(())
    }

    /// Transform data using the windowed statistics
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.transformer.transform(x)
    }
}

/// Streaming Principal Component Analysis using incremental SVD
pub struct StreamingPCA {
    /// Current mean of the data
    mean: Array1<f64>,
    /// Principal components (loading vectors)
    components: Option<Array2<f64>>,
    /// Explained variance for each component
    explained_variance: Option<Array1<f64>>,
    /// Number of components to keep
    n_components: usize,
    /// Number of features
    nfeatures: usize,
    /// Number of samples seen
    n_samples: usize,
    /// Forgetting factor for incremental updates (0 < alpha <= 1)
    alpha: f64,
    /// Minimum number of samples before PCA is computed
    min_samples: usize,
    /// Accumulated covariance matrix
    cov_matrix: Array2<f64>,
}

impl StreamingPCA {
    /// Create a new streaming PCA
    pub fn new(
        nfeatures: usize,
        n_components: usize,
        alpha: f64,
        min_samples: usize,
    ) -> Result<Self> {
        if n_components > nfeatures {
            return Err(TransformError::InvalidInput(
                "n_components cannot be larger than nfeatures".to_string(),
            ));
        }
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TransformError::InvalidInput(
                "alpha must be in (0, 1]".to_string(),
            ));
        }

        Ok(StreamingPCA {
            mean: Array1::zeros(nfeatures),
            components: None,
            explained_variance: None,
            n_components,
            nfeatures,
            n_samples: 0,
            alpha,
            min_samples,
            cov_matrix: Array2::zeros((nfeatures, nfeatures)),
        })
    }

    /// Update PCA with new batch of data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        let _batch_size = x.shape()[0];

        // Update mean using exponential moving average
        for sample in x.rows() {
            self.n_samples += 1;
            let weight = if self.n_samples == 1 { 1.0 } else { self.alpha };

            for (i, &value) in sample.iter().enumerate() {
                self.mean[i] = (1.0 - weight) * self.mean[i] + weight * value;
            }
        }

        // Update covariance matrix
        if self.n_samples >= self.min_samples {
            for sample in x.rows() {
                let centered = &sample.to_owned() - &self.mean;
                let outer_product = centered
                    .clone()
                    .insert_axis(scirs2_core::ndarray::Axis(1))
                    .dot(&centered.insert_axis(scirs2_core::ndarray::Axis(0)));

                let weight = self.alpha;
                self.cov_matrix = (1.0 - weight) * &self.cov_matrix + weight * outer_product;
            }

            // Compute PCA from covariance matrix
            self.compute_pca()?;
        }

        Ok(())
    }

    fn compute_pca(&mut self) -> Result<()> {
        // Perform proper eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = eigh(&self.cov_matrix.view(), None).map_err(|e| {
            TransformError::ComputationError(format!("Eigendecomposition failed: {e}"))
        })?;

        // Sort by eigenvalues in descending order
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
            .iter()
            .zip(eigenvectors.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top n_components
        let mut components = Array2::zeros((self.n_components, self.nfeatures));
        let mut explained_var = Array1::zeros(self.n_components);

        for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(self.n_components).enumerate() {
            components.row_mut(i).assign(eigenvec);
            explained_var[i] = eigenval.max(0.0);
        }

        self.components = Some(components);
        self.explained_variance = Some(explained_var);
        Ok(())
    }

    /// Transform data using current PCA
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(ref components) = self.components {
            if x.shape()[1] != self.nfeatures {
                return Err(TransformError::InvalidInput(format!(
                    "Expected {} features, got {}",
                    self.nfeatures,
                    x.shape()[1]
                )));
            }

            let mut result = Array2::zeros((x.shape()[0], self.n_components));

            for (i, sample) in x.rows().into_iter().enumerate() {
                let centered = &sample.to_owned() - &self.mean;
                let transformed = components.dot(&centered);
                result.row_mut(i).assign(&transformed);
            }

            Ok(result)
        } else {
            Err(TransformError::TransformationError(
                "PCA not computed yet, need more samples".to_string(),
            ))
        }
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<Array1<f64>> {
        self.explained_variance.as_ref().map(|var| {
            let total_var = var.sum();
            if total_var > 0.0 {
                var / total_var
            } else {
                Array1::zeros(var.len())
            }
        })
    }

    /// Reset the PCA to initial state
    pub fn reset(&mut self) {
        self.mean.fill(0.0);
        self.components = None;
        self.explained_variance = None;
        self.n_samples = 0;
        self.cov_matrix.fill(0.0);
    }
}

/// Streaming outlier detector using statistical methods
pub struct StreamingOutlierDetector {
    /// Running statistics for each feature
    means: Array1<f64>,
    variances: Array1<f64>,
    /// Number of samples seen
    n_samples: usize,
    /// Number of features
    nfeatures: usize,
    /// Threshold for outlier detection (standard deviations)
    threshold: f64,
    /// Method for outlier detection
    method: OutlierMethod,
}

/// Methods for outlier detection in streaming data
#[derive(Debug, Clone)]
pub enum OutlierMethod {
    /// Z-score based detection
    ZScore,
    /// Modified Z-score using median absolute deviation
    ModifiedZScore,
    /// Isolation forest-like scoring
    IsolationScore,
}

impl StreamingOutlierDetector {
    /// Create a new streaming outlier detector
    pub fn new(nfeatures: usize, threshold: f64, method: OutlierMethod) -> Self {
        StreamingOutlierDetector {
            means: Array1::zeros(nfeatures),
            variances: Array1::zeros(nfeatures),
            n_samples: 0,
            nfeatures,
            threshold,
            method,
        }
    }

    /// Update statistics with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        // Update running statistics using Welford's algorithm
        for sample in x.rows() {
            self.n_samples += 1;
            let n = self.n_samples as f64;

            for (i, &value) in sample.iter().enumerate() {
                let delta = value - self.means[i];
                self.means[i] += delta / n;
                let delta2 = value - self.means[i];
                self.variances[i] += delta * delta2;
            }
        }

        Ok(())
    }

    /// Detect outliers in new data
    pub fn detect_outliers(&self, x: &Array2<f64>) -> Result<Array1<bool>> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        if self.n_samples < 2 {
            // Not enough data for outlier detection
            return Ok(Array1::from_elem(x.shape()[0], false));
        }

        let mut outliers = Array1::from_elem(x.shape()[0], false);

        match self.method {
            OutlierMethod::ZScore => {
                let stds = self.get_standard_deviations();

                for (i, sample) in x.rows().into_iter().enumerate() {
                    let mut is_outlier = false;
                    for (j, &value) in sample.iter().enumerate() {
                        if stds[j] > 1e-8 {
                            let z_score = (value - self.means[j]).abs() / stds[j];
                            if z_score > self.threshold {
                                is_outlier = true;
                                break;
                            }
                        }
                    }
                    outliers[i] = is_outlier;
                }
            }
            OutlierMethod::ModifiedZScore => {
                // Simplified modified z-score using running estimates
                for (i, sample) in x.rows().into_iter().enumerate() {
                    let mut is_outlier = false;
                    for (j, &value) in sample.iter().enumerate() {
                        let mad_estimate =
                            (self.variances[j] / (self.n_samples - 1) as f64).sqrt() * 0.6745;
                        if mad_estimate > 1e-8 {
                            let modified_z = 0.6745 * (value - self.means[j]).abs() / mad_estimate;
                            if modified_z > self.threshold {
                                is_outlier = true;
                                break;
                            }
                        }
                    }
                    outliers[i] = is_outlier;
                }
            }
            OutlierMethod::IsolationScore => {
                // Enhanced isolation forest-like scoring using path length estimation
                for (i, sample) in x.rows().into_iter().enumerate() {
                    let anomaly_score = self.compute_isolation_score(sample);
                    outliers[i] = anomaly_score > self.threshold;
                }
            }
        }

        Ok(outliers)
    }

    fn get_standard_deviations(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            Array1::ones(self.nfeatures)
        } else {
            self.variances
                .mapv(|v| (v / (self.n_samples - 1) as f64).sqrt().max(1e-8))
        }
    }

    /// Compute isolation score using path length estimation
    fn compute_isolation_score(&self, sample: scirs2_core::ndarray::ArrayView1<f64>) -> f64 {
        let mut path_length = 0.0;
        let mut current_sample = sample.to_owned();

        // Simulate isolation tree path length with statistical approximation
        let max_depth = ((self.n_samples as f64).log2().ceil() as usize).min(20);

        for depth in 0..max_depth {
            let mut min_split_distance = f64::INFINITY;

            // Find the most isolating dimension
            for j in 0..self.nfeatures {
                let std_dev = (self.variances[j] / (self.n_samples - 1) as f64).sqrt();
                if std_dev > 1e-8 {
                    // Distance from mean normalized by standard deviation
                    let normalized_distance = (current_sample[j] - self.means[j]).abs() / std_dev;

                    // Estimate how "isolating" this split would be
                    let split_effectiveness = normalized_distance * (1.0 + depth as f64 * 0.1);
                    min_split_distance = min_split_distance.min(split_effectiveness);
                }
            }

            // If sample is well-isolated in any dimension, break early
            if min_split_distance > 3.0 {
                path_length += depth as f64 + min_split_distance / 3.0;
                break;
            }

            path_length += 1.0;

            // Adjust sample position for next iteration (simulating tree traversal)
            for j in 0..self.nfeatures {
                let adjustment = (current_sample[j] - self.means[j]) * 0.1;
                current_sample[j] -= adjustment;
            }
        }

        // Shorter path lengths indicate anomalies
        // Normalize by expected path length for a dataset of this size
        let expected_path_length = if self.n_samples > 2 {
            2.0 * ((self.n_samples - 1) as f64).ln() + (std::f64::consts::E * 0.57721566)
                - 2.0 * (self.n_samples - 1) as f64 / self.n_samples as f64
        } else {
            1.0
        };

        // Return anomaly score (higher = more anomalous)
        2.0_f64.powf(-path_length / expected_path_length)
    }

    /// Get anomaly scores for samples
    pub fn anomaly_scores(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        if self.n_samples < 2 {
            return Ok(Array1::zeros(x.shape()[0]));
        }

        let mut scores = Array1::zeros(x.shape()[0]);
        let stds = self.get_standard_deviations();

        for (i, sample) in x.rows().into_iter().enumerate() {
            let mut score = 0.0;
            for (j, &value) in sample.iter().enumerate() {
                if stds[j] > 1e-8 {
                    let z_score = (value - self.means[j]).abs() / stds[j];
                    score += z_score;
                }
            }
            scores[i] = score / self.nfeatures as f64;
        }

        Ok(scores)
    }

    /// Reset detector to initial state
    pub fn reset(&mut self) {
        self.means.fill(0.0);
        self.variances.fill(0.0);
        self.n_samples = 0;
    }
}

/// Streaming feature selector based on variance or correlation thresholds
pub struct StreamingFeatureSelector {
    /// Feature variances
    variances: Array1<f64>,
    /// Feature means
    means: Array1<f64>,
    /// Correlation matrix (upper triangular)
    correlations: Array2<f64>,
    /// Number of samples seen
    n_samples: usize,
    /// Number of features
    nfeatures: usize,
    /// Variance threshold for feature selection
    variance_threshold: f64,
    /// Correlation threshold for removing highly correlated features
    correlationthreshold: f64,
    /// Selected feature indices
    selected_features: Option<Vec<usize>>,
}

impl StreamingFeatureSelector {
    /// Create a new streaming feature selector
    pub fn new(nfeatures: usize, variance_threshold: f64, correlationthreshold: f64) -> Self {
        StreamingFeatureSelector {
            variances: Array1::zeros(nfeatures),
            means: Array1::zeros(nfeatures),
            correlations: Array2::zeros((nfeatures, nfeatures)),
            n_samples: 0,
            nfeatures,
            variance_threshold,
            correlationthreshold,
            selected_features: None,
        }
    }

    /// Update statistics with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.nfeatures {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.nfeatures,
                x.shape()[1]
            )));
        }

        // Update running statistics
        for sample in x.rows() {
            self.n_samples += 1;
            let n = self.n_samples as f64;

            // Update means and variances
            for (i, &value) in sample.iter().enumerate() {
                let delta = value - self.means[i];
                self.means[i] += delta / n;
                let delta2 = value - self.means[i];
                self.variances[i] += delta * delta2;
            }

            // Update correlations (simplified running correlation)
            if self.n_samples > 1 {
                for i in 0..self.nfeatures {
                    for j in (i + 1)..self.nfeatures {
                        let val_i = sample[i] - self.means[i];
                        let val_j = sample[j] - self.means[j];
                        let covar_update = val_i * val_j / (n - 1.0);
                        self.correlations[[i, j]] =
                            (self.correlations[[i, j]] * (n - 2.0) + covar_update) / (n - 1.0);
                    }
                }
            }
        }

        // Update selected features based on current statistics
        if self.n_samples >= 10 {
            // Minimum samples for stable statistics
            self.update_selected_features();
        }

        Ok(())
    }

    fn update_selected_features(&mut self) {
        let mut selected = Vec::new();
        let current_variances = self.get_current_variances();

        // First pass: select features based on variance threshold
        for i in 0..self.nfeatures {
            if current_variances[i] > self.variance_threshold {
                selected.push(i);
            }
        }

        // Second pass: remove highly correlated features
        let mut final_selected = Vec::new();
        for &i in &selected {
            let mut should_include = true;

            for &j in &final_selected {
                if i != j {
                    let corr = self.get_correlation(i, j, &current_variances);
                    if corr.abs() > self.correlationthreshold {
                        // Keep feature with higher variance
                        if current_variances[i] <= current_variances[j] {
                            should_include = false;
                            break;
                        }
                    }
                }
            }

            if should_include {
                final_selected.push(i);
            }
        }

        self.selected_features = Some(final_selected);
    }

    fn get_current_variances(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            Array1::zeros(self.nfeatures)
        } else {
            self.variances.mapv(|v| v / (self.n_samples - 1) as f64)
        }
    }

    fn get_correlation(&self, i: usize, j: usize, variances: &Array1<f64>) -> f64 {
        let var_i = variances[i];
        let var_j = variances[j];

        if var_i > 1e-8 && var_j > 1e-8 {
            let idx = if i < j { (i, j) } else { (j, i) };
            self.correlations[idx] / (var_i * var_j).sqrt()
        } else {
            0.0
        }
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(ref selected) = self.selected_features {
            if x.shape()[1] != self.nfeatures {
                return Err(TransformError::InvalidInput(format!(
                    "Expected {} features, got {}",
                    self.nfeatures,
                    x.shape()[1]
                )));
            }

            if selected.is_empty() {
                return Ok(Array2::zeros((x.shape()[0], 0)));
            }

            let mut result = Array2::zeros((x.shape()[0], selected.len()));

            for (row_idx, sample) in x.rows().into_iter().enumerate() {
                for (col_idx, &feature_idx) in selected.iter().enumerate() {
                    result[[row_idx, col_idx]] = sample[feature_idx];
                }
            }

            Ok(result)
        } else {
            // No features selected yet, return original data
            Ok(x.to_owned())
        }
    }

    /// Get indices of selected features
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features.as_ref()
    }

    /// Get number of selected features
    pub fn n_features_selected(&self) -> usize {
        self.selected_features
            .as_ref()
            .map_or(self.nfeatures, |s| s.len())
    }

    /// Reset selector to initial state
    pub fn reset(&mut self) {
        self.variances.fill(0.0);
        self.means.fill(0.0);
        self.correlations.fill(0.0);
        self.n_samples = 0;
        self.selected_features = None;
    }
}
