//! Text clustering algorithms with semantic similarity support
//!
//! This module provides specialized clustering algorithms for text data that leverage
//! semantic similarity measures rather than traditional distance metrics. It includes
//! algorithms optimized for document clustering, sentence clustering, and topic modeling.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;
use statrs::statistics::Statistics;

/// Text representation types for clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(SerdeSerialize, SerdeDeserialize))]
pub enum TextRepresentation {
    /// Term Frequency-Inverse Document Frequency vectors
    TfIdf {
        /// TF-IDF matrix (documents x terms)
        vectors: Array2<f64>,
        /// Vocabulary mapping
        vocabulary: HashMap<String, usize>,
    },
    /// Word embeddings (Word2Vec, GloVe, etc.)
    WordEmbeddings {
        /// Embedding vectors (documents x embedding_dim)
        vectors: Array2<f64>,
        /// Embedding dimension
        embedding_dim: usize,
    },
    /// Contextualized embeddings (BERT, RoBERTa, etc.)
    ContextualEmbeddings {
        /// Embedding vectors (documents x embedding_dim)
        vectors: Array2<f64>,
        /// Model name used for embeddings
        model_name: String,
    },
    /// Document-Term matrix for traditional approaches
    DocumentTerm {
        /// Document-term matrix (documents x terms)
        matrix: Array2<f64>,
        /// Term vocabulary
        vocabulary: Vec<String>,
    },
}

/// Semantic similarity metrics for text clustering
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(SerdeSerialize, SerdeDeserialize))]
pub enum SemanticSimilarity {
    /// Cosine similarity (most common for text)
    Cosine,
    /// Jaccard similarity for binary/sparse features
    Jaccard,
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Pearson correlation coefficient
    Pearson,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Hellinger distance
    Hellinger,
    /// Bhattacharyya distance
    Bhattacharyya,
}

/// Configuration for semantic text clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(SerdeSerialize, SerdeDeserialize))]
pub struct SemanticClusteringConfig {
    /// Similarity metric to use
    pub similarity_metric: SemanticSimilarity,
    /// Number of clusters (for k-means style algorithms)
    pub n_clusters: Option<usize>,
    /// Minimum similarity threshold for hierarchical clustering
    pub similarity_threshold: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use dimensionality reduction preprocessing
    pub use_dimension_reduction: bool,
    /// Target dimensions for dimensionality reduction
    pub target_dimensions: Option<usize>,
    /// Preprocessing options
    pub preprocessing: TextPreprocessing,
}

impl Default for SemanticClusteringConfig {
    fn default() -> Self {
        Self {
            similarity_metric: SemanticSimilarity::Cosine,
            n_clusters: Some(10),
            similarity_threshold: 0.5,
            max_iterations: 100,
            tolerance: 1e-4,
            use_dimension_reduction: false,
            target_dimensions: None,
            preprocessing: TextPreprocessing::default(),
        }
    }
}

/// Text preprocessing configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(SerdeSerialize, SerdeDeserialize))]
pub struct TextPreprocessing {
    /// Normalize vectors to unit length
    pub normalize_vectors: bool,
    /// Remove zero-variance features
    pub remove_zero_variance: bool,
    /// Apply TF-IDF weighting (if using raw term frequencies)
    pub apply_tfidf: bool,
    /// Minimum document frequency for terms
    pub min_df: f64,
    /// Maximum document frequency for terms
    pub max_df: f64,
    /// Maximum number of features to keep
    pub max_features: Option<usize>,
}

impl Default for TextPreprocessing {
    fn default() -> Self {
        Self {
            normalize_vectors: true,
            remove_zero_variance: true,
            apply_tfidf: true,
            min_df: 0.01,
            max_df: 0.95,
            max_features: None,
        }
    }
}

/// Semantic K-means clustering for text data
pub struct SemanticKMeans {
    config: SemanticClusteringConfig,
    centroids: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    inertia: Option<f64>,
    n_iterations: Option<usize>,
}

impl SemanticKMeans {
    /// Create a new semantic K-means clusterer
    pub fn new(config: SemanticClusteringConfig) -> Self {
        Self {
            config,
            centroids: None,
            labels: None,
            inertia: None,
            n_iterations: None,
        }
    }

    /// Fit the model to text data
    pub fn fit(&mut self, text_repr: &TextRepresentation) -> Result<()> {
        let vectors = self.extract_vectors(text_repr)?;
        let preprocessed = self.preprocess_vectors(vectors)?;

        let n_clusters = self.config.n_clusters.unwrap_or(10);
        self.fit_kmeans(preprocessed.view(), n_clusters)?;

        Ok(())
    }

    /// Extract numerical vectors from text representation
    fn extract_vectors(&self, text_repr: &TextRepresentation) -> Result<Array2<f64>> {
        match text_repr {
            TextRepresentation::TfIdf { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::WordEmbeddings { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::ContextualEmbeddings { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::DocumentTerm { matrix, .. } => Ok(matrix.clone()),
        }
    }

    /// Preprocess vectors according to configuration
    fn preprocess_vectors(&self, vectors: Array2<f64>) -> Result<Array2<f64>> {
        let mut processed = vectors;

        // Apply TF-IDF if requested
        if self.config.preprocessing.apply_tfidf {
            processed = self.apply_tfidf_weighting(processed)?;
        }

        // Remove zero-variance features
        if self.config.preprocessing.remove_zero_variance {
            processed = self.remove_zero_variance_features(processed)?;
        }

        // Normalize vectors
        if self.config.preprocessing.normalize_vectors {
            processed = self.normalize_vectors(processed)?;
        }

        // Dimensionality reduction
        if self.config.use_dimension_reduction {
            if let Some(target_dim) = self.config.target_dimensions {
                processed = self.reduce_dimensions(processed, target_dim)?;
            }
        }

        Ok(processed)
    }

    /// Apply TF-IDF weighting to document-term matrix
    fn apply_tfidf_weighting(&self, matrix: Array2<f64>) -> Result<Array2<f64>> {
        let (n_docs, n_terms) = matrix.dim();
        let mut tfidf_matrix = matrix.clone();

        // Compute document frequencies
        let mut df = Array1::zeros(n_terms);
        for term_idx in 0..n_terms {
            let mut doc_count = 0;
            for doc_idx in 0..n_docs {
                if matrix[[doc_idx, term_idx]] > 0.0 {
                    doc_count += 1;
                }
            }
            df[term_idx] = doc_count as f64;
        }

        // Compute TF-IDF
        for doc_idx in 0..n_docs {
            for term_idx in 0..n_terms {
                let tf = matrix[[doc_idx, term_idx]];
                if tf > 0.0 && df[term_idx] > 0.0 {
                    let idf = (n_docs as f64 / df[term_idx]).ln();
                    tfidf_matrix[[doc_idx, term_idx]] = tf * idf;
                }
            }
        }

        Ok(tfidf_matrix)
    }

    /// Remove features with zero variance
    fn remove_zero_variance_features(&self, matrix: Array2<f64>) -> Result<Array2<f64>> {
        let mut feature_mask = Vec::new();

        for col_idx in 0..matrix.ncols() {
            let column = matrix.column(col_idx);
            let mean = column.mean();
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;

            feature_mask.push(variance > 1e-10); // Keep non-zero variance features
        }

        let valid_features: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        if valid_features.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "All features have zero variance".to_string(),
            ));
        }

        let filtered = matrix.select(Axis(1), &valid_features);
        Ok(filtered)
    }

    /// Normalize vectors to unit length
    fn normalize_vectors(&self, matrix: Array2<f64>) -> Result<Array2<f64>> {
        let mut normalized = matrix.clone();

        for mut row in normalized.rows_mut() {
            let norm = (row.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        Ok(normalized)
    }

    /// Reduce dimensionality using PCA (simplified)
    fn reduce_dimensions(&self, matrix: Array2<f64>, target_dim: usize) -> Result<Array2<f64>> {
        let (_n_samples, n_features) = matrix.dim();

        if target_dim >= n_features {
            return Ok(matrix);
        }

        // Simplified dimensionality reduction: just keep first N dimensions
        // In practice, this would use proper PCA or other dimensionality reduction
        let reduced = matrix.slice(s![.., 0..target_dim]).to_owned();
        Ok(reduced)
    }

    /// Fit K-means clustering with semantic similarity
    fn fit_kmeans(&mut self, data: ArrayView2<f64>, k: usize) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if k > n_samples {
            return Err(ClusteringError::InvalidInput(
                "Number of clusters cannot exceed number of samples".to_string(),
            ));
        }

        // Initialize centroids using k-means++
        let mut centroids = Array2::zeros((k, n_features));
        self.initialize_centroids_plus_plus(&mut centroids, data)?;

        let mut labels = Array1::zeros(n_samples);
        let mut prev_inertia = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.config.max_iterations {
            n_iter = iter + 1;

            // Assign points to clusters
            let mut inertia = 0.0;
            for (i, sample) in data.rows().into_iter().enumerate() {
                let (best_cluster, distance) =
                    self.find_closest_centroid(sample, centroids.view())?;
                labels[i] = best_cluster;
                inertia += distance;
            }

            // Check convergence
            if (prev_inertia - inertia).abs() < self.config.tolerance {
                break;
            }
            prev_inertia = inertia;

            // Update centroids
            self.update_centroids(&mut centroids, data, labels.view())?;
        }

        self.centroids = Some(centroids);
        self.labels = Some(labels);
        self.inertia = Some(prev_inertia);
        self.n_iterations = Some(n_iter);

        Ok(())
    }

    /// Initialize centroids using k-means++ method
    fn initialize_centroids_plus_plus(
        &self,
        centroids: &mut Array2<f64>,
        data: ArrayView2<f64>,
    ) -> Result<()> {
        let n_samples = data.nrows();
        let k = centroids.nrows();

        // Choose first centroid randomly
        centroids.row_mut(0).assign(&data.row(0));

        // Choose remaining centroids
        for i in 1..k {
            let mut distances = Array1::zeros(n_samples);
            let mut total_distance = 0.0;

            // Calculate distances to nearest existing centroid
            for (j, sample) in data.rows().into_iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                for centroid_idx in 0..i {
                    let dist = self.compute_distance(sample, centroids.row(centroid_idx))?;
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[j] = min_dist * min_dist;
                total_distance += distances[j];
            }

            // Select next centroid probabilistically
            if total_distance > 0.0 {
                let target = total_distance * 0.5; // Simplified: use middle point
                let mut cumsum = 0.0;
                for (j, &dist) in distances.iter().enumerate() {
                    cumsum += dist;
                    if cumsum >= target {
                        centroids.row_mut(i).assign(&data.row(j));
                        break;
                    }
                }
            } else {
                // Fallback: use next available point
                if i < n_samples {
                    centroids.row_mut(i).assign(&data.row(i));
                }
            }
        }

        Ok(())
    }

    /// Find the closest centroid to a sample
    fn find_closest_centroid(
        &self,
        sample: ArrayView1<f64>,
        centroids: ArrayView2<f64>,
    ) -> Result<(usize, f64)> {
        let mut min_distance = f64::INFINITY;
        let mut best_cluster = 0;

        for (i, centroid) in centroids.rows().into_iter().enumerate() {
            let distance = self.compute_distance(sample, centroid)?;
            if distance < min_distance {
                min_distance = distance;
                best_cluster = i;
            }
        }

        Ok((best_cluster, min_distance))
    }

    /// Compute distance based on configured similarity metric
    fn compute_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        match self.config.similarity_metric {
            SemanticSimilarity::Cosine => {
                let similarity = self.cosine_similarity(a, b)?;
                Ok(1.0 - similarity) // Convert similarity to distance
            }
            SemanticSimilarity::Euclidean => Ok(euclidean_distance(a, b)),
            SemanticSimilarity::Manhattan => Ok(self.manhattan_distance(a, b)?),
            SemanticSimilarity::Jaccard => {
                let similarity = self.jaccard_similarity(a, b)?;
                Ok(1.0 - similarity)
            }
            SemanticSimilarity::Pearson => {
                let correlation = self.pearson_correlation(a, b)?;
                Ok(1.0 - correlation.abs()) // Use absolute correlation
            }
            SemanticSimilarity::JensenShannon => self.jensen_shannon_distance(a, b),
            SemanticSimilarity::Hellinger => self.hellinger_distance(a, b),
            SemanticSimilarity::Bhattacharyya => self.bhattacharyya_distance(a, b),
        }
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        let dot_product = a.dot(&b);
        let norm_a = (a.dot(&a)).sqrt();
        let norm_b = (b.dot(&b)).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// Compute Manhattan distance
    fn manhattan_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum())
    }

    /// Compute Jaccard similarity
    fn jaccard_similarity(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        let threshold = 1e-10;
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let x_present = x > threshold;
            let y_present = y > threshold;

            if x_present && y_present {
                intersection += 1.0;
            }
            if x_present || y_present {
                union += 1.0;
            }
        }

        if union == 0.0 {
            Ok(1.0) // Both vectors are zero
        } else {
            Ok(intersection / union)
        }
    }

    /// Compute Pearson correlation coefficient
    fn pearson_correlation(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        let n = a.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_a = a.mean();
        let mean_b = b.mean();

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff_a = x - mean_a;
            let diff_b = y - mean_b;
            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Compute Jensen-Shannon distance
    fn jensen_shannon_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        // Normalize to probability distributions
        let sum_a: f64 = a.iter().map(|&x| x.max(0.0)).sum();
        let sum_b: f64 = b.iter().map(|&x| x.max(0.0)).sum();

        if sum_a == 0.0 || sum_b == 0.0 {
            return Ok(1.0);
        }

        let p: Vec<f64> = a.iter().map(|&x| x.max(0.0) / sum_a).collect();
        let q: Vec<f64> = b.iter().map(|&x| x.max(0.0) / sum_b).collect();

        // Compute average distribution
        let m: Vec<f64> = p
            .iter()
            .zip(q.iter())
            .map(|(&x, &y)| (x + y) / 2.0)
            .collect();

        // Compute KL divergences
        let kl_pm = self.kl_divergence(&p, &m);
        let kl_qm = self.kl_divergence(&q, &m);

        // Jensen-Shannon divergence
        let js = (kl_pm + kl_qm) / 2.0;
        Ok(js.sqrt()) // Return Jensen-Shannon distance
    }

    /// Compute KL divergence between two probability distributions
    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let mut kl = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-10 && qi > 1e-10 {
                kl += pi * (pi / qi).ln();
            }
        }
        kl
    }

    /// Compute Hellinger distance
    fn hellinger_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        // Normalize to probability distributions
        let sum_a: f64 = a.iter().map(|&x| x.max(0.0)).sum();
        let sum_b: f64 = b.iter().map(|&x| x.max(0.0)).sum();

        if sum_a == 0.0 || sum_b == 0.0 {
            return Ok(1.0);
        }

        let mut sum_sqrt_products = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let p = x.max(0.0) / sum_a;
            let q = y.max(0.0) / sum_b;
            sum_sqrt_products += (p * q).sqrt();
        }

        let hellinger = (1.0 - sum_sqrt_products).sqrt();
        Ok(hellinger)
    }

    /// Compute Bhattacharyya distance
    fn bhattacharyya_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        // Normalize to probability distributions
        let sum_a: f64 = a.iter().map(|&x| x.max(0.0)).sum();
        let sum_b: f64 = b.iter().map(|&x| x.max(0.0)).sum();

        if sum_a == 0.0 || sum_b == 0.0 {
            return Ok(f64::INFINITY);
        }

        let mut bc = 0.0; // Bhattacharyya coefficient
        for (&x, &y) in a.iter().zip(b.iter()) {
            let p = x.max(0.0) / sum_a;
            let q = y.max(0.0) / sum_b;
            bc += (p * q).sqrt();
        }

        if bc <= 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(-bc.ln())
        }
    }

    /// Update centroids
    fn update_centroids(
        &self,
        centroids: &mut Array2<f64>,
        data: ArrayView2<f64>,
        labels: ArrayView1<usize>,
    ) -> Result<()> {
        let k = centroids.nrows();
        let n_features = centroids.ncols();

        // Reset centroids
        centroids.fill(0.0);
        let mut cluster_sizes = vec![0; k];

        // Accumulate points for each cluster
        for (i, &label) in labels.iter().enumerate() {
            if label < k {
                for j in 0..n_features {
                    centroids[[label, j]] += data[[i, j]];
                }
                cluster_sizes[label] += 1;
            }
        }

        // Average to get centroids
        for i in 0..k {
            if cluster_sizes[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] /= cluster_sizes[i] as f64;
                }
            }
        }

        Ok(())
    }

    /// Predict cluster assignments for new text data
    pub fn predict(&self, text_repr: &TextRepresentation) -> Result<Array1<usize>> {
        let vectors = self.extract_vectors(text_repr)?;
        let preprocessed = self.preprocess_vectors(vectors)?;

        if let Some(ref centroids) = self.centroids {
            let mut labels = Array1::zeros(preprocessed.nrows());

            for (i, sample) in preprocessed.rows().into_iter().enumerate() {
                let (best_cluster, _distance) =
                    self.find_closest_centroid(sample, centroids.view())?;
                labels[i] = best_cluster;
            }

            Ok(labels)
        } else {
            Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ))
        }
    }

    /// Get cluster centroids
    pub fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }

    /// Get inertia (sum of distances to centroids)
    pub fn inertia(&self) -> Option<f64> {
        self.inertia
    }

    /// Get number of iterations performed
    pub fn n_iterations(&self) -> Option<usize> {
        self.n_iterations
    }
}

/// Hierarchical clustering for text with semantic similarity
pub struct SemanticHierarchical {
    config: SemanticClusteringConfig,
    linkage_matrix: Option<Array2<f64>>,
    n_clusters: Option<usize>,
}

impl SemanticHierarchical {
    /// Create a new semantic hierarchical clusterer
    pub fn new(config: SemanticClusteringConfig) -> Self {
        Self {
            config,
            linkage_matrix: None,
            n_clusters: None,
        }
    }

    /// Fit hierarchical clustering to text data
    pub fn fit(&mut self, text_repr: &TextRepresentation) -> Result<()> {
        let vectors = self.extract_vectors(text_repr)?;
        let preprocessed = self.preprocess_vectors(vectors)?;

        self.fit_hierarchical(preprocessed.view())?;
        Ok(())
    }

    /// Extract and preprocess vectors (same as SemanticKMeans)
    fn extract_vectors(&self, text_repr: &TextRepresentation) -> Result<Array2<f64>> {
        match text_repr {
            TextRepresentation::TfIdf { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::WordEmbeddings { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::ContextualEmbeddings { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::DocumentTerm { matrix, .. } => Ok(matrix.clone()),
        }
    }

    /// Preprocess vectors (simplified version)
    fn preprocess_vectors(&self, vectors: Array2<f64>) -> Result<Array2<f64>> {
        // Simplified preprocessing for hierarchical clustering
        if self.config.preprocessing.normalize_vectors {
            let mut normalized = vectors.clone();
            for mut row in normalized.rows_mut() {
                let norm = (row.iter().map(|&x| x * x).sum::<f64>()).sqrt();
                if norm > 1e-10 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
            Ok(normalized)
        } else {
            Ok(vectors)
        }
    }

    /// Fit hierarchical clustering using single linkage
    fn fit_hierarchical(&mut self, data: ArrayView2<f64>) -> Result<()> {
        let n_samples = data.nrows();

        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 samples for hierarchical clustering".to_string(),
            ));
        }

        // Compute distance matrix
        let mut distance_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = self.compute_distance(data.row(i), data.row(j))?;
                distance_matrix[[i, j]] = distance;
                distance_matrix[[j, i]] = distance;
            }
        }

        // Perform single linkage clustering (simplified)
        let mut clusters: Vec<HashSet<usize>> = (0..n_samples)
            .map(|i| {
                let mut set = HashSet::new();
                set.insert(i);
                set
            })
            .collect();

        let mut linkage_steps = Vec::new();

        while clusters.len() > 1 {
            let mut min_distance = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 1;

            // Find closest clusters
            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let distance =
                        self.cluster_distance(&clusters[i], &clusters[j], &distance_matrix);
                    if distance < min_distance {
                        min_distance = distance;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Record linkage step
            linkage_steps.push([
                merge_i as f64,
                merge_j as f64,
                min_distance,
                (clusters[merge_i].len() + clusters[merge_j].len()) as f64,
            ]);

            // Merge clusters
            let cluster_j = clusters.remove(merge_j);
            clusters[merge_i].extend(cluster_j);
        }

        // Convert to linkage matrix
        let linkage_matrix = Array2::from_shape_vec(
            (linkage_steps.len(), 4),
            linkage_steps.into_iter().flatten().collect(),
        )
        .unwrap();

        self.linkage_matrix = Some(linkage_matrix);
        Ok(())
    }

    /// Compute distance between clusters (single linkage)
    fn cluster_distance(
        &self,
        cluster_a: &HashSet<usize>,
        cluster_b: &HashSet<usize>,
        distance_matrix: &Array2<f64>,
    ) -> f64 {
        let mut min_distance = f64::INFINITY;

        for &i in cluster_a {
            for &j in cluster_b {
                let distance = distance_matrix[[i, j]];
                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        min_distance
    }

    /// Compute distance based on similarity metric
    fn compute_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        match self.config.similarity_metric {
            SemanticSimilarity::Cosine => {
                let dot_product = a.dot(&b);
                let norm_a = (a.dot(&a)).sqrt();
                let norm_b = (b.dot(&b)).sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    Ok(1.0)
                } else {
                    let similarity = dot_product / (norm_a * norm_b);
                    Ok(1.0 - similarity)
                }
            }
            SemanticSimilarity::Euclidean => Ok(euclidean_distance(a, b)),
            _ => {
                // For other metrics, use Euclidean as fallback
                Ok(euclidean_distance(a, b))
            }
        }
    }

    /// Get linkage matrix
    pub fn linkage_matrix(&self) -> Option<&Array2<f64>> {
        self.linkage_matrix.as_ref()
    }
}

/// Topic modeling-based clustering using semantic similarity
pub struct TopicBasedClustering {
    config: SemanticClusteringConfig,
    topics: Option<Array2<f64>>,
    document_topic_distributions: Option<Array2<f64>>,
    n_topics: usize,
}

impl TopicBasedClustering {
    /// Create a new topic-based clusterer
    pub fn new(config: SemanticClusteringConfig, n_topics: usize) -> Self {
        Self {
            config,
            topics: None,
            document_topic_distributions: None,
            n_topics,
        }
    }

    /// Fit topic-based clustering (simplified NMF-like approach)
    pub fn fit(&mut self, text_repr: &TextRepresentation) -> Result<()> {
        let vectors = self.extract_vectors(text_repr)?;
        let preprocessed = self.preprocess_vectors(vectors)?;

        self.fit_topics(preprocessed.view())?;
        Ok(())
    }

    /// Extract vectors from text representation
    fn extract_vectors(&self, text_repr: &TextRepresentation) -> Result<Array2<f64>> {
        match text_repr {
            TextRepresentation::TfIdf { vectors, .. } => Ok(vectors.clone()),
            TextRepresentation::DocumentTerm { matrix, .. } => Ok(matrix.clone()),
            _ => Err(ClusteringError::InvalidInput(
                "Topic modeling requires TF-IDF or document-term matrix".to_string(),
            )),
        }
    }

    /// Preprocess vectors
    fn preprocess_vectors(&self, vectors: Array2<f64>) -> Result<Array2<f64>> {
        // Ensure non-negative values for topic modeling
        let mut processed = vectors.mapv(|x| x.max(0.0));

        // Normalize documents
        for mut row in processed.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 1e-10 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(processed)
    }

    /// Fit topic model using simplified NMF
    fn fit_topics(&mut self, data: ArrayView2<f64>) -> Result<()> {
        let (n_docs, n_terms) = data.dim();

        // Initialize topics and document-topic distributions randomly
        let mut topics = Array2::from_elem((self.n_topics, n_terms), 1.0 / n_terms as f64);
        let mut doc_topics = Array2::from_elem((n_docs, self.n_topics), 1.0 / self.n_topics as f64);

        // Simplified NMF iterations
        for _iter in 0..self.config.max_iterations {
            // Update document-topic distributions
            for doc_idx in 0..n_docs {
                for topic_idx in 0..self.n_topics {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;

                    for term_idx in 0..n_terms {
                        let observed = data[[doc_idx, term_idx]];
                        let expected = topics[[topic_idx, term_idx]];

                        if expected > 1e-10 {
                            numerator += observed * expected;
                            denominator += expected;
                        }
                    }

                    if denominator > 1e-10 {
                        doc_topics[[doc_idx, topic_idx]] = numerator / denominator;
                    }
                }

                // Normalize document-topic distribution
                let sum: f64 = doc_topics.row(doc_idx).sum();
                if sum > 1e-10 {
                    for topic_idx in 0..self.n_topics {
                        doc_topics[[doc_idx, topic_idx]] /= sum;
                    }
                }
            }

            // Update topics
            for topic_idx in 0..self.n_topics {
                for term_idx in 0..n_terms {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;

                    for doc_idx in 0..n_docs {
                        let observed = data[[doc_idx, term_idx]];
                        let doc_topic_weight = doc_topics[[doc_idx, topic_idx]];

                        numerator += observed * doc_topic_weight;
                        denominator += doc_topic_weight;
                    }

                    if denominator > 1e-10 {
                        topics[[topic_idx, term_idx]] = numerator / denominator;
                    }
                }

                // Normalize topic distribution
                let sum: f64 = topics.row(topic_idx).sum();
                if sum > 1e-10 {
                    for term_idx in 0..n_terms {
                        topics[[topic_idx, term_idx]] /= sum;
                    }
                }
            }
        }

        self.topics = Some(topics);
        self.document_topic_distributions = Some(doc_topics);
        Ok(())
    }

    /// Get cluster assignments based on dominant topics
    pub fn predict(&self, text_repr: &TextRepresentation) -> Result<Array1<usize>> {
        if let Some(ref doc_topics) = self.document_topic_distributions {
            let mut labels = Array1::zeros(doc_topics.nrows());

            for (doc_idx, doc_topic_dist) in doc_topics.rows().into_iter().enumerate() {
                let mut max_prob = 0.0;
                let mut best_topic = 0;

                for (topic_idx, &prob) in doc_topic_dist.iter().enumerate() {
                    if prob > max_prob {
                        max_prob = prob;
                        best_topic = topic_idx;
                    }
                }

                labels[doc_idx] = best_topic;
            }

            Ok(labels)
        } else {
            Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ))
        }
    }

    /// Get topics (term distributions)
    pub fn topics(&self) -> Option<&Array2<f64>> {
        self.topics.as_ref()
    }

    /// Get document-topic distributions
    pub fn document_topic_distributions(&self) -> Option<&Array2<f64>> {
        self.document_topic_distributions.as_ref()
    }
}

/// Convenience functions for text clustering

/// Perform semantic K-means clustering on text data
#[allow(dead_code)]
pub fn semantic_kmeans(
    text_repr: &TextRepresentation,
    n_clusters: usize,
    similarity_metric: SemanticSimilarity,
) -> Result<(Array2<f64>, Array1<usize>)> {
    let config = SemanticClusteringConfig {
        n_clusters: Some(n_clusters),
        similarity_metric,
        ..Default::default()
    };

    let mut clusterer = SemanticKMeans::new(config);
    clusterer.fit(text_repr)?;

    let centers = clusterer
        .cluster_centers()
        .ok_or_else(|| {
            ClusteringError::ComputationError("Failed to get cluster centers".to_string())
        })?
        .clone();

    let labels = clusterer.predict(text_repr)?;

    Ok((centers, labels))
}

/// Perform hierarchical clustering on text data
#[allow(dead_code)]
pub fn semantic_hierarchical(
    text_repr: &TextRepresentation,
    similarity_metric: SemanticSimilarity,
) -> Result<Array2<f64>> {
    let config = SemanticClusteringConfig {
        similarity_metric,
        ..Default::default()
    };

    let mut clusterer = SemanticHierarchical::new(config);
    clusterer.fit(text_repr)?;

    clusterer
        .linkage_matrix()
        .ok_or_else(|| {
            ClusteringError::ComputationError("Failed to get linkage matrix".to_string())
        })
        .map(|m| m.clone())
}

/// Perform topic-based clustering on text data
#[allow(dead_code)]
pub fn topic_clustering(
    text_repr: &TextRepresentation,
    n_topics: usize,
) -> Result<(Array2<f64>, Array2<f64>, Array1<usize>)> {
    let config = SemanticClusteringConfig::default();
    let mut clusterer = TopicBasedClustering::new(config, n_topics);

    clusterer.fit(text_repr)?;

    let _topics = clusterer
        .topics()
        .ok_or_else(|| ClusteringError::ComputationError("Failed to get topics".to_string()))?
        .clone();

    let doc_topics = clusterer
        .document_topic_distributions()
        .ok_or_else(|| {
            ClusteringError::ComputationError(
                "Failed to get document-topic distributions".to_string(),
            )
        })?
        .clone();

    let labels = clusterer.predict(text_repr)?;

    Ok((_topics, doc_topics, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_semantic_kmeans_basic() {
        // Create sample TF-IDF vectors
        let vectors = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.9],
        )
        .unwrap();

        let text_repr = TextRepresentation::TfIdf {
            vectors,
            vocabulary: HashMap::new(),
        };

        let result = semantic_kmeans(&text_repr, 2, SemanticSimilarity::Cosine);
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_similarity_metrics() {
        let a = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = ndarray::Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let config = SemanticClusteringConfig::default();
        let clusterer = SemanticKMeans::new(config);

        // Test cosine similarity
        let cosine_sim = clusterer.cosine_similarity(a.view(), b.view()).unwrap();
        assert_eq!(cosine_sim, 0.0); // Orthogonal vectors

        // Test Manhattan distance
        let manhattan = clusterer.manhattan_distance(a.view(), b.view()).unwrap();
        assert_eq!(manhattan, 2.0);
    }

    #[test]
    fn testtext_preprocessing() {
        let config = SemanticClusteringConfig::default();
        let clusterer = SemanticKMeans::new(config);

        let matrix = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 1.0, 2.0, 2.0]).unwrap();

        let normalized = clusterer.normalize_vectors(matrix).unwrap();

        // Check that vectors are normalized
        for row in normalized.rows() {
            let norm = (row.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_topic_clustering_basic() {
        // Create sample document-term matrix
        let matrix = Array2::from_shape_vec(
            (3, 4),
            vec![2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        )
        .unwrap();

        let text_repr = TextRepresentation::DocumentTerm {
            matrix,
            vocabulary: vec![
                "word1".to_string(),
                "word2".to_string(),
                "word3".to_string(),
                "word4".to_string(),
            ],
        };

        let result = topic_clustering(&text_repr, 2);
        assert!(result.is_ok());

        let (topics, doc_topics, labels) = result.unwrap();
        assert_eq!(topics.nrows(), 2);
        assert_eq!(doc_topics.nrows(), 3);
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_semantic_similarity_enum() {
        assert_eq!(SemanticSimilarity::Cosine, SemanticSimilarity::Cosine);
        assert_ne!(SemanticSimilarity::Cosine, SemanticSimilarity::Euclidean);
    }
}
