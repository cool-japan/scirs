//! Weighted ensemble clustering.
//!
//! Provides:
//!
//! - [`WeightedVoting`] – evidence-accumulation ensemble with per-member
//!   quality weights.
//! - [`SelectiveEnsemble`] – selects a diverse subset of base clusterings
//!   to form the final ensemble.
//! - [`ClusterSimilarity`] – NMI, ARI, and Fowlkes-Mallows similarity
//!   measures used to score ensemble diversity.
//! - [`BootstrapEnsemble`] – bootstrap-based cluster ensemble.
//! - [`StackedClustering`] – stacked generalization for clustering (the
//!   consensus labels of base clusterings are used as meta-features for a
//!   second-level clustering).

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// ClusterSimilarity – pairwise similarity metrics
// ---------------------------------------------------------------------------

/// Pairwise similarity metrics for comparing two cluster label vectors.
pub struct ClusterSimilarity;

impl ClusterSimilarity {
    /// Adjusted Rand Index between two label vectors.
    ///
    /// Returns a value in [-0.5, 1.0] where 1.0 indicates perfect agreement.
    pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> f64 {
        if labels_a.len() != labels_b.len() || labels_a.is_empty() {
            return 0.0;
        }
        let n = labels_a.len();

        let ka = *labels_a.iter().max().unwrap_or(&0) + 1;
        let kb = *labels_b.iter().max().unwrap_or(&0) + 1;

        // Build contingency table
        let mut contingency = vec![vec![0usize; kb]; ka];
        for i in 0..n {
            let a = labels_a[i];
            let b = labels_b[i];
            if a < ka && b < kb {
                contingency[a][b] += 1;
            }
        }

        // Row / column sums
        let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
        let col_sums: Vec<usize> = (0..kb)
            .map(|j| contingency.iter().map(|r| r[j]).sum())
            .collect();

        let sum_comb_c: f64 = contingency
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| comb2(v))
            .sum();
        let sum_comb_a: f64 = row_sums.iter().map(|&v| comb2(v)).sum();
        let sum_comb_b: f64 = col_sums.iter().map(|&v| comb2(v)).sum();
        let comb_n = comb2(n);

        let expected = sum_comb_a * sum_comb_b / comb_n.max(1.0);
        let max_val = (sum_comb_a + sum_comb_b) / 2.0;
        let denom = max_val - expected;
        if denom.abs() < 1e-15 {
            if (sum_comb_c - expected).abs() < 1e-15 {
                1.0
            } else {
                0.0
            }
        } else {
            (sum_comb_c - expected) / denom
        }
    }

    /// Normalized Mutual Information (arithmetic mean normalisation).
    pub fn normalized_mutual_info(labels_a: &[usize], labels_b: &[usize]) -> f64 {
        if labels_a.len() != labels_b.len() || labels_a.is_empty() {
            return 0.0;
        }
        let n = labels_a.len() as f64;
        let ka = *labels_a.iter().max().unwrap_or(&0) + 1;
        let kb = *labels_b.iter().max().unwrap_or(&0) + 1;

        let mut contingency = vec![vec![0usize; kb]; ka];
        for (&a, &b) in labels_a.iter().zip(labels_b.iter()) {
            if a < ka && b < kb {
                contingency[a][b] += 1;
            }
        }

        let row_sums: Vec<f64> = contingency
            .iter()
            .map(|r| r.iter().sum::<usize>() as f64)
            .collect();
        let col_sums: Vec<f64> = (0..kb)
            .map(|j| contingency.iter().map(|r| r[j]).sum::<usize>() as f64)
            .collect();

        // Mutual information
        let mut mi = 0.0_f64;
        for i in 0..ka {
            for j in 0..kb {
                let nij = contingency[i][j] as f64;
                if nij > 0.0 {
                    mi += nij / n * (nij * n / (row_sums[i] * col_sums[j])).ln();
                }
            }
        }

        // Entropies
        let h_a: f64 = row_sums
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / n;
                -p * p.ln()
            })
            .sum();
        let h_b: f64 = col_sums
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / n;
                -p * p.ln()
            })
            .sum();

        let denom = (h_a + h_b) / 2.0;
        if denom < 1e-15 {
            1.0
        } else {
            mi / denom
        }
    }

    /// Fowlkes-Mallows index between two label vectors.
    pub fn fowlkes_mallows(labels_a: &[usize], labels_b: &[usize]) -> f64 {
        if labels_a.len() != labels_b.len() || labels_a.is_empty() {
            return 0.0;
        }
        let n = labels_a.len();
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_ = 0u64;

        for i in 0..n {
            for j in (i + 1)..n {
                let same_a = labels_a[i] == labels_a[j];
                let same_b = labels_b[i] == labels_b[j];
                match (same_a, same_b) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    _ => {}
                }
            }
        }
        let denom = ((tp + fp) as f64 * (tp + fn_) as f64).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            tp as f64 / denom
        }
    }
}

fn comb2(n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        (n * (n - 1)) as f64 / 2.0
    }
}

// ---------------------------------------------------------------------------
// WeightedVoting
// ---------------------------------------------------------------------------

/// Configuration for weighted voting ensemble clustering.
#[derive(Debug, Clone)]
pub struct WeightedVotingConfig {
    /// Number of base clusterings to combine.
    pub n_base: usize,
    /// Quality metric used to assign weights to base clusterings.
    pub quality_metric: EnsembleQualityMetric,
    /// Minimum quality threshold for inclusion.
    pub min_quality: f64,
    /// Number of clusters in the final ensemble.
    pub n_clusters: usize,
    /// Maximum iterations for the consensus step.
    pub max_iter: usize,
}

impl Default for WeightedVotingConfig {
    fn default() -> Self {
        Self {
            n_base: 10,
            quality_metric: EnsembleQualityMetric::NMI,
            min_quality: 0.0,
            n_clusters: 3,
            max_iter: 100,
        }
    }
}

/// Quality metric for scoring base clusterings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleQualityMetric {
    /// Normalized Mutual Information vs. a reference clustering.
    NMI,
    /// Adjusted Rand Index vs. a reference clustering.
    ARI,
    /// Fowlkes-Mallows index vs. a reference clustering.
    FowlkesMallows,
    /// Uniform weights (no quality weighting).
    Uniform,
}

/// Weighted voting ensemble with evidence accumulation.
///
/// Combines multiple base clusterings by building a weighted co-association
/// matrix: `S[i,j] += w_k * (labels_k[i] == labels_k[j])`.  The final
/// clustering is obtained by running k-means on the dissimilarity
/// `D = 1 - S`.
pub struct WeightedVoting {
    config: WeightedVotingConfig,
}

impl WeightedVoting {
    /// Create a new WeightedVoting instance.
    pub fn new(config: WeightedVotingConfig) -> Self {
        Self { config }
    }

    /// Combine base clusterings using weighted evidence accumulation.
    ///
    /// `base_labels`: each row is a labelling from one base clusterer (shape:
    ///   `[n_base, n_samples]`).
    /// `weights`: per-base-clusterer quality weight (length `n_base`).  If
    ///   `None`, uniform weights are used.
    pub fn combine(
        &self,
        base_labels: &[Vec<usize>],
        weights: Option<&[f64]>,
    ) -> Result<WeightedVotingResult> {
        if base_labels.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No base clusterings provided".into(),
            ));
        }
        let n = base_labels[0].len();
        for bl in base_labels.iter() {
            if bl.len() != n {
                return Err(ClusteringError::InvalidInput(
                    "All base clusterings must have the same length".into(),
                ));
            }
        }

        let m = base_labels.len();
        let default_w = vec![1.0 / m as f64; m];
        let w: &[f64] = weights.unwrap_or(&default_w);

        // Filter by minimum quality (quality vs. the average / a reference)
        // Here we use ARI between each pair and assign weight proportional
        // to average similarity with other clusterings.
        let mut effective_weights: Vec<f64> = match self.config.quality_metric {
            EnsembleQualityMetric::Uniform => w.to_vec(),
            EnsembleQualityMetric::NMI => self.compute_diversity_weights(base_labels, |a, b| {
                ClusterSimilarity::normalized_mutual_info(a, b)
            }),
            EnsembleQualityMetric::ARI => self.compute_diversity_weights(base_labels, |a, b| {
                ClusterSimilarity::adjusted_rand_index(a, b)
            }),
            EnsembleQualityMetric::FowlkesMallows => self
                .compute_diversity_weights(base_labels, |a, b| {
                    ClusterSimilarity::fowlkes_mallows(a, b)
                }),
        };

        // Apply supplied weights multiplicatively
        for (i, ew) in effective_weights.iter_mut().enumerate() {
            *ew *= w[i];
        }

        // Normalise
        let w_sum: f64 = effective_weights.iter().sum();
        if w_sum < 1e-15 {
            for ew in effective_weights.iter_mut() {
                *ew = 1.0 / m as f64;
            }
        } else {
            for ew in effective_weights.iter_mut() {
                *ew /= w_sum;
            }
        }

        // Build weighted co-association matrix
        let mut co_assoc = vec![vec![0.0f64; n]; n];
        for (k, bl) in base_labels.iter().enumerate() {
            let wk = effective_weights[k];
            if wk < 1e-15 {
                continue;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    if bl[i] == bl[j] {
                        co_assoc[i][j] += wk;
                        co_assoc[j][i] += wk;
                    }
                }
            }
        }
        // Self-similarity = 1
        for i in 0..n {
            co_assoc[i][i] = 1.0;
        }

        // Final consensus clustering: k-means on dissimilarity embedding rows
        let labels = self.consensus_from_coassoc(&co_assoc, n)?;
        let used_bases = base_labels.len();

        Ok(WeightedVotingResult {
            labels,
            weights: effective_weights,
            co_association: co_assoc,
            n_clusters: self.config.n_clusters,
            n_base_clusterings: used_bases,
        })
    }

    /// Compute per-clustering weights as their average similarity to others.
    fn compute_diversity_weights(
        &self,
        base_labels: &[Vec<usize>],
        sim_fn: impl Fn(&[usize], &[usize]) -> f64,
    ) -> Vec<f64> {
        let m = base_labels.len();
        let mut weights = vec![0.0f64; m];
        if m == 1 {
            weights[0] = 1.0;
            return weights;
        }
        for i in 0..m {
            let sum: f64 = (0..m)
                .filter(|&j| j != i)
                .map(|j| sim_fn(&base_labels[i], &base_labels[j]))
                .sum();
            weights[i] = sum / (m - 1) as f64;
        }
        weights
    }

    /// Simple k-means on the dissimilarity rows of the co-association matrix.
    fn consensus_from_coassoc(&self, co_assoc: &[Vec<f64>], n: usize) -> Result<Vec<usize>> {
        let k = self.config.n_clusters.min(n);
        if k == 0 || n == 0 {
            return Ok(vec![0; n]);
        }

        // Use co-assoc rows as feature vectors
        let mut centroids: Vec<Vec<f64>> = (0..k).map(|i| co_assoc[i].clone()).collect();
        let mut labels = vec![0usize; n];

        for _ in 0..self.config.max_iter {
            // Assign
            for i in 0..n {
                let mut best = 0;
                let mut best_d = f64::MAX;
                for (j, c) in centroids.iter().enumerate() {
                    let d: f64 = co_assoc[i]
                        .iter()
                        .zip(c.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if d < best_d {
                        best_d = d;
                        best = j;
                    }
                }
                labels[i] = best;
            }

            // Update
            let mut new_cents = vec![vec![0.0f64; n]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let j = labels[i];
                counts[j] += 1;
                for dim in 0..n {
                    new_cents[j][dim] += co_assoc[i][dim];
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    let nf = counts[j] as f64;
                    for dim in 0..n {
                        new_cents[j][dim] /= nf;
                    }
                }
            }
            centroids = new_cents;
        }
        Ok(labels)
    }
}

/// Result from WeightedVoting.
#[derive(Debug, Clone)]
pub struct WeightedVotingResult {
    /// Consensus cluster labels for each data point.
    pub labels: Vec<usize>,
    /// Effective weights assigned to each base clustering.
    pub weights: Vec<f64>,
    /// Weighted co-association matrix (n × n).
    pub co_association: Vec<Vec<f64>>,
    /// Number of consensus clusters.
    pub n_clusters: usize,
    /// Number of base clusterings used.
    pub n_base_clusterings: usize,
}

impl WeightedVotingResult {
    /// Average weight of the base clusterings.
    pub fn mean_weight(&self) -> f64 {
        if self.weights.is_empty() {
            return 0.0;
        }
        self.weights.iter().sum::<f64>() / self.weights.len() as f64
    }

    /// Weight variance (spread of quality scores).
    pub fn weight_variance(&self) -> f64 {
        if self.weights.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_weight();
        let var: f64 = self
            .weights
            .iter()
            .map(|&w| (w - mean).powi(2))
            .sum::<f64>()
            / (self.weights.len() - 1) as f64;
        var
    }
}

// ---------------------------------------------------------------------------
// SelectiveEnsemble
// ---------------------------------------------------------------------------

/// Configuration for selective ensemble.
#[derive(Debug, Clone)]
pub struct SelectiveEnsembleConfig {
    /// Target ensemble size (number of base clusterings to select).
    pub target_size: usize,
    /// Diversity metric for selection.
    pub diversity_metric: DiversityMeasure,
    /// Minimum pairwise diversity threshold (clusterings below this are
    /// considered redundant).
    pub diversity_threshold: f64,
}

impl Default for SelectiveEnsembleConfig {
    fn default() -> Self {
        Self {
            target_size: 5,
            diversity_metric: DiversityMeasure::NMI,
            diversity_threshold: 0.3,
        }
    }
}

/// Diversity measure for SelectiveEnsemble.
#[derive(Debug, Clone, Copy)]
pub enum DiversityMeasure {
    /// Use 1 - NMI as diversity.
    NMI,
    /// Use 1 - ARI as diversity.
    ARI,
    /// Use 1 - FowlkesMallows as diversity.
    FowlkesMallows,
}

/// SelectiveEnsemble: greedily selects a diverse subset of base clusterings.
///
/// Starting from the clustering with the highest average diversity, it
/// iteratively adds the clustering that maximises the minimum pairwise
/// diversity with those already selected.
pub struct SelectiveEnsemble {
    config: SelectiveEnsembleConfig,
}

impl SelectiveEnsemble {
    /// Create a new SelectiveEnsemble.
    pub fn new(config: SelectiveEnsembleConfig) -> Self {
        Self { config }
    }

    /// Select a diverse subset of base clusterings.
    ///
    /// Returns the indices of the selected clusterings.
    pub fn select(&self, base_labels: &[Vec<usize>]) -> Result<SelectiveEnsembleResult> {
        let m = base_labels.len();
        if m == 0 {
            return Err(ClusteringError::InvalidInput(
                "No base clusterings to select from".into(),
            ));
        }

        let target = self.config.target_size.min(m);
        let sim_fn: Box<dyn Fn(&[usize], &[usize]) -> f64> = match self.config.diversity_metric {
            DiversityMeasure::NMI => {
                Box::new(|a, b| ClusterSimilarity::normalized_mutual_info(a, b))
            }
            DiversityMeasure::ARI => Box::new(|a, b| ClusterSimilarity::adjusted_rand_index(a, b)),
            DiversityMeasure::FowlkesMallows => {
                Box::new(|a, b| ClusterSimilarity::fowlkes_mallows(a, b))
            }
        };

        // Compute full diversity matrix (diversity = 1 - similarity)
        let mut diversity = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            for j in (i + 1)..m {
                let d = 1.0 - sim_fn(&base_labels[i], &base_labels[j]).max(0.0);
                diversity[i][j] = d;
                diversity[j][i] = d;
            }
        }

        // Greedy max-min selection
        // Start with the clustering that has the highest average diversity
        let avg_div: Vec<f64> = diversity
            .iter()
            .map(|row| row.iter().sum::<f64>() / (m - 1).max(1) as f64)
            .collect();
        let start = avg_div
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut selected = vec![start];
        let mut remaining: Vec<usize> = (0..m).filter(|&i| i != start).collect();

        while selected.len() < target && !remaining.is_empty() {
            // For each remaining, compute its minimum diversity with the selected set
            let mut best_idx_in_remaining = 0;
            let mut best_min_div = -1.0_f64;
            for (ri, &cand) in remaining.iter().enumerate() {
                let min_div = selected
                    .iter()
                    .map(|&s| diversity[cand][s])
                    .fold(f64::MAX, f64::min);
                if min_div > best_min_div {
                    best_min_div = min_div;
                    best_idx_in_remaining = ri;
                }
            }
            let chosen = remaining.remove(best_idx_in_remaining);
            selected.push(chosen);
        }

        // Compute average pairwise diversity of selected set
        let avg_diversity = if selected.len() < 2 {
            0.0
        } else {
            let pairs = selected.len() * (selected.len() - 1) / 2;
            let sum: f64 = selected
                .iter()
                .enumerate()
                .flat_map(|(i, &a)| {
                    let div_ref = &diversity;
                    selected[(i + 1)..]
                        .iter()
                        .map(move |&b| div_ref[a][b])
                        .collect::<Vec<_>>()
                })
                .sum();
            sum / pairs as f64
        };

        Ok(SelectiveEnsembleResult {
            selected_indices: selected,
            diversity_matrix: diversity,
            average_diversity: avg_diversity,
        })
    }
}

/// Result from SelectiveEnsemble.
#[derive(Debug, Clone)]
pub struct SelectiveEnsembleResult {
    /// Indices of selected base clusterings.
    pub selected_indices: Vec<usize>,
    /// Full pairwise diversity matrix (m × m).
    pub diversity_matrix: Vec<Vec<f64>>,
    /// Average pairwise diversity of the selected set.
    pub average_diversity: f64,
}

// ---------------------------------------------------------------------------
// BootstrapEnsemble
// ---------------------------------------------------------------------------

/// Configuration for bootstrap ensemble clustering.
#[derive(Debug, Clone)]
pub struct BootstrapEnsembleConfig {
    /// Number of bootstrap samples.
    pub n_bootstrap: usize,
    /// Fraction of the dataset to sample per bootstrap.
    pub sample_ratio: f64,
    /// Number of clusters per base clustering.
    pub n_clusters: usize,
    /// Maximum iterations for each base k-means run.
    pub max_iter: usize,
    /// Random seed base.
    pub seed: u64,
}

impl Default for BootstrapEnsembleConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 10,
            sample_ratio: 0.8,
            n_clusters: 3,
            max_iter: 50,
            seed: 42,
        }
    }
}

/// Bootstrap-based cluster ensemble.
///
/// Generates multiple bootstrap sub-samples of the data, clusters each
/// with k-means, and then combines the base clusterings via a weighted
/// co-association matrix.
pub struct BootstrapEnsemble {
    config: BootstrapEnsembleConfig,
}

impl BootstrapEnsemble {
    /// Create a new BootstrapEnsemble.
    pub fn new(config: BootstrapEnsembleConfig) -> Self {
        Self { config }
    }

    /// Fit the bootstrap ensemble and return consensus labels.
    pub fn fit<F>(&self, data: ArrayView2<F>) -> Result<BootstrapEnsembleResult>
    where
        F: Float + FromPrimitive + Debug + Clone,
        f64: From<F>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".into()));
        }
        let k = self.config.n_clusters.min(n);
        let sample_n = ((n as f64 * self.config.sample_ratio) as usize).max(k);

        let mut base_labels_all: Vec<Vec<usize>> = Vec::new();

        for b in 0..self.config.n_bootstrap {
            // Simple deterministic bootstrap: use modular stride
            let stride = (b * 7 + 3) % n + 1;
            let indices: Vec<usize> = (0..sample_n).map(|i| (i * stride) % n).collect();

            // Extract sub-sample centroids via k-means
            let sample_centroids = self.fit_kmeans_on_indices(data, &indices, k)?;

            // Assign all points to nearest centroid
            let labels: Vec<usize> = (0..n)
                .map(|i| {
                    let row: Vec<f64> = data.row(i).iter().map(|&v| f64::from(v)).collect();
                    nearest_centroid_f64(&sample_centroids, &row)
                })
                .collect();
            base_labels_all.push(labels);
        }

        // Combine via co-association
        let voting = WeightedVoting::new(WeightedVotingConfig {
            n_base: self.config.n_bootstrap,
            quality_metric: EnsembleQualityMetric::NMI,
            min_quality: 0.0,
            n_clusters: k,
            max_iter: self.config.max_iter,
        });
        let voting_result = voting.combine(&base_labels_all, None)?;

        // Compute stability: average NMI across bootstrap pairs
        let stability = compute_average_nmi(&base_labels_all);

        Ok(BootstrapEnsembleResult {
            labels: voting_result.labels,
            base_labels: base_labels_all,
            stability,
            n_bootstrap: self.config.n_bootstrap,
            n_clusters: k,
        })
    }

    fn fit_kmeans_on_indices<F>(
        &self,
        data: ArrayView2<F>,
        indices: &[usize],
        k: usize,
    ) -> Result<Vec<Vec<f64>>>
    where
        F: Float + FromPrimitive + Debug + Clone,
        f64: From<F>,
    {
        let d = data.ncols();
        let n = indices.len();
        let k = k.min(n);
        // Initial centroids: evenly spaced within the sample
        let step = n / k;
        let mut cents: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let idx = indices[i * step];
                data.row(idx).iter().map(|&v| f64::from(v)).collect()
            })
            .collect();

        for _ in 0..self.config.max_iter {
            let mut new_cents = vec![vec![0.0f64; d]; k];
            let mut counts = vec![0usize; k];
            for &idx in indices {
                let row: Vec<f64> = data.row(idx).iter().map(|&v| f64::from(v)).collect();
                let best = nearest_centroid_f64(&cents, &row);
                counts[best] += 1;
                for dim in 0..d {
                    new_cents[best][dim] += row[dim];
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    let nf = counts[j] as f64;
                    for dim in 0..d {
                        new_cents[j][dim] /= nf;
                    }
                } else {
                    new_cents[j] = cents[j].clone();
                }
            }
            cents = new_cents;
        }
        Ok(cents)
    }
}

/// Result from BootstrapEnsemble.
#[derive(Debug, Clone)]
pub struct BootstrapEnsembleResult {
    /// Consensus cluster labels.
    pub labels: Vec<usize>,
    /// Labels from each bootstrap run.
    pub base_labels: Vec<Vec<usize>>,
    /// Average NMI across bootstrap pairs (stability estimate).
    pub stability: f64,
    /// Number of bootstrap runs.
    pub n_bootstrap: usize,
    /// Number of clusters.
    pub n_clusters: usize,
}

// ---------------------------------------------------------------------------
// StackedClustering
// ---------------------------------------------------------------------------

/// Configuration for stacked clustering.
#[derive(Debug, Clone)]
pub struct StackedClusteringConfig {
    /// Number of base clusterings.
    pub n_base: usize,
    /// Number of clusters for each base clustering.
    pub n_base_clusters: usize,
    /// Number of meta-clusters in the second level.
    pub n_meta_clusters: usize,
    /// Maximum iterations for each level.
    pub max_iter: usize,
    /// Whether to append the original features to the meta-features.
    pub append_original: bool,
}

impl Default for StackedClusteringConfig {
    fn default() -> Self {
        Self {
            n_base: 5,
            n_base_clusters: 5,
            n_meta_clusters: 3,
            max_iter: 100,
            append_original: false,
        }
    }
}

/// Stacked generalization for clustering.
///
/// Base clusterers produce soft or hard label vectors that are used as
/// meta-features for a second-level k-means clustering.
pub struct StackedClustering {
    config: StackedClusteringConfig,
}

impl StackedClustering {
    /// Create a new StackedClustering instance.
    pub fn new(config: StackedClusteringConfig) -> Self {
        Self { config }
    }

    /// Fit the stacked ensemble.
    ///
    /// Uses k-means with varied random offsets as base clusterers, then
    /// clusters the label matrix with a second-level k-means.
    pub fn fit<F>(&self, data: ArrayView2<F>) -> Result<StackedClusteringResult>
    where
        F: Float + FromPrimitive + Debug + Clone,
        f64: From<F>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".into()));
        }
        let kb = self.config.n_base_clusters.min(n);
        let km = self.config.n_meta_clusters.min(n);

        // Generate base label vectors with different centroid offsets
        let mut meta_features: Vec<Vec<f64>> = vec![Vec::new(); n];

        for b in 0..self.config.n_base {
            let offset = b as f64 * 0.01; // slight deterministic perturbation
            let labels = self.kmeans_with_offset(data, kb, offset)?;
            for i in 0..n {
                meta_features[i].push(labels[i] as f64);
            }
        }

        // Optionally append normalised original features
        if self.config.append_original && d > 0 {
            // Compute per-dimension range for normalisation
            let mut min_d = vec![f64::MAX; d];
            let mut max_d = vec![f64::MIN; d];
            for row in data.rows() {
                for (j, &v) in row.iter().enumerate() {
                    let vf = f64::from(v);
                    if vf < min_d[j] {
                        min_d[j] = vf;
                    }
                    if vf > max_d[j] {
                        max_d[j] = vf;
                    }
                }
            }
            for (i, row) in data.rows().into_iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let vf = f64::from(v);
                    let range = (max_d[j] - min_d[j]).max(1e-15);
                    meta_features[i].push((vf - min_d[j]) / range);
                }
            }
        }

        // Second-level k-means on meta-features
        let meta_d = meta_features.first().map(|r| r.len()).unwrap_or(0);
        let mut meta_cents: Vec<Vec<f64>> = (0..km).map(|i| meta_features[i % n].clone()).collect();
        let mut final_labels = vec![0usize; n];

        for _ in 0..self.config.max_iter {
            for i in 0..n {
                final_labels[i] = nearest_centroid_f64(&meta_cents, &meta_features[i]);
            }
            let mut new_cents = vec![vec![0.0; meta_d]; km];
            let mut counts = vec![0usize; km];
            for i in 0..n {
                let j = final_labels[i];
                counts[j] += 1;
                for k in 0..meta_d {
                    new_cents[j][k] += meta_features[i][k];
                }
            }
            for j in 0..km {
                if counts[j] > 0 {
                    let nf = counts[j] as f64;
                    for k in 0..meta_d {
                        new_cents[j][k] /= nf;
                    }
                }
            }
            meta_cents = new_cents;
        }

        Ok(StackedClusteringResult {
            labels: final_labels,
            meta_features,
            n_base: self.config.n_base,
            n_meta_clusters: km,
        })
    }

    fn kmeans_with_offset<F>(
        &self,
        data: ArrayView2<F>,
        k: usize,
        offset: f64,
    ) -> Result<Vec<usize>>
    where
        F: Float + FromPrimitive + Debug + Clone,
        f64: From<F>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        let k = k.min(n);
        let offset_f = F::from_f64(offset).unwrap_or(F::zero());

        let mut cents: Vec<Vec<f64>> = (0..k)
            .map(|i| data.row(i).iter().map(|&v| f64::from(v) + offset).collect())
            .collect();
        let mut labels = vec![0usize; n];

        for _ in 0..self.config.max_iter {
            for i in 0..n {
                let row: Vec<f64> = data.row(i).iter().map(|&v| f64::from(v)).collect();
                labels[i] = nearest_centroid_f64(&cents, &row);
            }
            let mut new_cents = vec![vec![0.0; d]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let j = labels[i];
                counts[j] += 1;
                let row: Vec<f64> = data.row(i).iter().map(|&v| f64::from(v)).collect();
                for dim in 0..d {
                    new_cents[j][dim] += row[dim];
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    let nf = counts[j] as f64;
                    for dim in 0..d {
                        new_cents[j][dim] /= nf;
                    }
                }
            }
            cents = new_cents;
        }
        Ok(labels)
    }
}

/// Result from StackedClustering.
#[derive(Debug, Clone)]
pub struct StackedClusteringResult {
    /// Final consensus cluster labels.
    pub labels: Vec<usize>,
    /// Meta-features matrix (base clustering label vectors per point).
    pub meta_features: Vec<Vec<f64>>,
    /// Number of base clusterings used.
    pub n_base: usize,
    /// Number of meta-level clusters.
    pub n_meta_clusters: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn nearest_centroid_f64(centroids: &[Vec<f64>], point: &[f64]) -> usize {
    let mut best = 0;
    let mut best_d = f64::MAX;
    for (j, c) in centroids.iter().enumerate() {
        let d: f64 = c
            .iter()
            .zip(point.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

fn compute_average_nmi(base_labels: &[Vec<usize>]) -> f64 {
    let m = base_labels.len();
    if m < 2 {
        return 1.0;
    }
    let pairs = m * (m - 1) / 2;
    let sum: f64 = (0..m)
        .flat_map(|i| (i + 1..m).map(move |j| (i, j)))
        .map(|(i, j)| ClusterSimilarity::normalized_mutual_info(&base_labels[i], &base_labels[j]))
        .sum();
    sum / pairs as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_labels() -> (Vec<usize>, Vec<usize>) {
        let a: Vec<usize> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        let b: Vec<usize> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        (a, b)
    }

    #[test]
    fn test_ari_perfect() {
        let (a, b) = two_cluster_labels();
        let ari = ClusterSimilarity::adjusted_rand_index(&a, &b);
        assert!((ari - 1.0).abs() < 1e-9, "ARI = {}", ari);
    }

    #[test]
    fn test_nmi_perfect() {
        let (a, b) = two_cluster_labels();
        let nmi = ClusterSimilarity::normalized_mutual_info(&a, &b);
        assert!((nmi - 1.0).abs() < 1e-9, "NMI = {}", nmi);
    }

    #[test]
    fn test_fowlkes_mallows_perfect() {
        let (a, b) = two_cluster_labels();
        let fm = ClusterSimilarity::fowlkes_mallows(&a, &b);
        assert!((fm - 1.0).abs() < 1e-9, "FM = {}", fm);
    }

    #[test]
    fn test_weighted_voting() {
        let labels1: Vec<usize> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        let labels2: Vec<usize> = (0..20).map(|i| if i < 12 { 0 } else { 1 }).collect();
        let base = vec![labels1, labels2];

        let wv = WeightedVoting::new(WeightedVotingConfig {
            n_base: 2,
            n_clusters: 2,
            ..Default::default()
        });
        let result = wv.combine(&base, None).expect("combine ok");
        assert_eq!(result.labels.len(), 20);
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_selective_ensemble() {
        let labels: Vec<Vec<usize>> = (0..5)
            .map(|b| (0..20).map(|i| if i < 10 + b { 0 } else { 1 }).collect())
            .collect();
        let se = SelectiveEnsemble::new(SelectiveEnsembleConfig {
            target_size: 3,
            ..Default::default()
        });
        let result = se.select(&labels).expect("select ok");
        assert_eq!(result.selected_indices.len(), 3);
    }

    #[test]
    fn test_bootstrap_ensemble() {
        let data: Array2<f64> = {
            let mut v = Vec::new();
            for i in 0..20 {
                let offset = if i < 10 { 0.0 } else { 10.0 };
                v.extend_from_slice(&[offset + i as f64 * 0.1, offset + i as f64 * 0.1]);
            }
            Array2::from_shape_vec((20, 2), v).expect("ok")
        };
        let be = BootstrapEnsemble::new(BootstrapEnsembleConfig {
            n_bootstrap: 3,
            n_clusters: 2,
            ..Default::default()
        });
        let result = be.fit(data.view()).expect("fit ok");
        assert_eq!(result.labels.len(), 20);
        assert_eq!(result.n_bootstrap, 3);
    }

    #[test]
    fn test_stacked_clustering() {
        let data: Array2<f64> = {
            let mut v = Vec::new();
            for i in 0..20 {
                let offset = if i < 10 { 0.0 } else { 10.0 };
                v.extend_from_slice(&[offset + i as f64 * 0.1, offset + i as f64 * 0.1]);
            }
            Array2::from_shape_vec((20, 2), v).expect("ok")
        };
        let sc = StackedClustering::new(StackedClusteringConfig {
            n_base: 3,
            n_base_clusters: 2,
            n_meta_clusters: 2,
            ..Default::default()
        });
        let result = sc.fit(data.view()).expect("fit ok");
        assert_eq!(result.labels.len(), 20);
    }
}
