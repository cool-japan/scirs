//! Vector Operations for Hyperdimensional Computing
//!
//! This module implements the core mathematical operations for hypervectors,
//! including creation, similarity computation, bundling, binding, and other
//! fundamental operations that form the basis of HDC computations.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::{NdimageError, NdimageResult};
use crate::hyperdimensional_computing::types::{HDCConfig, Hypervector};

impl Hypervector {
    /// Create a new random hypervector
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the hypervector
    /// * `sparsity` - Fraction of dimensions that are non-zero
    ///
    /// # Returns
    ///
    /// A new random hypervector with the specified dimensionality and sparsity
    pub fn random(dim: usize, sparsity: f64) -> Self {
        let num_nonzero = (dim as f64 * sparsity) as usize;
        let mut sparse_data = Vec::new();
        let mut rng = scirs2_core::random::rng();
        let mut used_indices = HashSet::new();

        while sparse_data.len() < num_nonzero {
            let idx = rng.gen_range(0..dim);
            if !used_indices.contains(&idx) {
                used_indices.insert(idx);
                let value = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                sparse_data.push((idx, value));
            }
        }

        sparse_data.sort_by_key(|&(idx_, _)| idx_);

        let norm = (sparse_data.len() as f64).sqrt();

        Self {
            sparse_data,
            dimension: dim,
            norm,
        }
    }

    /// Create hypervector from dense array
    ///
    /// # Arguments
    ///
    /// * `data` - Dense array to convert
    /// * `sparsity` - Target sparsity level
    ///
    /// # Returns
    ///
    /// A sparse hypervector representation of the dense data
    pub fn from_dense(data: &Array1<f64>, sparsity: f64) -> Self {
        let mut sparse_data = Vec::new();
        let threshold = sparsity * data.iter().map(|x| x.abs()).sum::<f64>() / data.len() as f64;

        for (i, &value) in data.iter().enumerate() {
            if value.abs() > threshold {
                sparse_data.push((i, value));
            }
        }

        let norm = sparse_data.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();

        Self {
            sparse_data,
            dimension: data.len(),
            norm,
        }
    }

    /// Convert to dense representation
    ///
    /// # Returns
    ///
    /// Dense array representation of the hypervector
    pub fn to_dense(&self) -> Array1<f64> {
        let mut dense = Array1::zeros(self.dimension);
        for &(idx, value) in &self.sparse_data {
            dense[idx] = value;
        }
        dense
    }

    /// Compute cosine similarity with another hypervector
    ///
    /// # Arguments
    ///
    /// * `other` - The other hypervector to compare with
    ///
    /// # Returns
    ///
    /// Cosine similarity score between 0.0 and 1.0
    pub fn similarity(&self, other: &Self) -> f64 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.sparse_data.len() && j < other.sparse_data.len() {
            let (self_idx, self_val) = self.sparse_data[i];
            let (other_idx, other_val) = other.sparse_data[j];

            if self_idx == other_idx {
                dot_product += self_val * other_val;
                i += 1;
                j += 1;
            } else if self_idx < other_idx {
                i += 1;
            } else {
                j += 1;
            }
        }

        if self.norm > 0.0 && other.norm > 0.0 {
            dot_product / (self.norm * other.norm)
        } else {
            0.0
        }
    }

    /// Bundle (superposition) with another hypervector
    ///
    /// Bundling creates a representation that is similar to both input vectors
    /// and can be used to represent sets or unions of concepts.
    ///
    /// # Arguments
    ///
    /// * `other` - The other hypervector to bundle with
    ///
    /// # Returns
    ///
    /// A new hypervector representing the bundle of both inputs
    pub fn bundle(&self, other: &Self) -> NdimageResult<Self> {
        if self.dimension != other.dimension {
            return Err(NdimageError::InvalidInput(format!(
                "Dimension mismatch: {} vs {}",
                self.dimension, other.dimension
            )));
        }

        let mut result_map = BTreeMap::new();

        // Add self values
        for &(idx, value) in &self.sparse_data {
            *result_map.entry(idx).or_insert(0.0) += value;
        }

        // Add other values
        for &(idx, value) in &other.sparse_data {
            *result_map.entry(idx).or_insert(0.0) += value;
        }

        let sparse_data: Vec<(usize, f64)> = result_map
            .into_iter()
            .filter(|&(_, v)| v.abs() > 1e-10)
            .collect();

        let norm = sparse_data.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();

        Ok(Self {
            sparse_data,
            dimension: self.dimension,
            norm,
        })
    }

    /// Bind (convolution) with another hypervector
    ///
    /// Binding creates a representation that is dissimilar to both inputs
    /// and is used to encode associations and relationships between concepts.
    ///
    /// # Arguments
    ///
    /// * `other` - The other hypervector to bind with
    ///
    /// # Returns
    ///
    /// A new hypervector representing the binding of both inputs
    pub fn bind(&self, other: &Self) -> NdimageResult<Self> {
        if self.dimension != other.dimension {
            return Err(NdimageError::InvalidInput(format!(
                "Dimension mismatch: {} vs {}",
                self.dimension, other.dimension
            )));
        }

        // For sparse binding, we use circular convolution approximation
        let mut result_map = BTreeMap::new();

        for &(self_idx, self_val) in &self.sparse_data {
            for &(other_idx, other_val) in &other.sparse_data {
                let result_idx = (self_idx + other_idx) % self.dimension;
                *result_map.entry(result_idx).or_insert(0.0) += self_val * other_val;
            }
        }

        let sparse_data: Vec<(usize, f64)> = result_map
            .into_iter()
            .filter(|&(_, v)| v.abs() > 1e-10)
            .collect();

        let norm = sparse_data.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();

        Ok(Self {
            sparse_data,
            dimension: self.dimension,
            norm,
        })
    }

    /// Unbind (inverse binding) with another hypervector
    ///
    /// Unbinding recovers the original vector that was bound with the given vector.
    /// If C = A ⊛ B, then A ≈ C ⊛ B⁻¹
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to unbind from this one
    ///
    /// # Returns
    ///
    /// A new hypervector representing the unbinding result
    pub fn unbind(&self, other: &Self) -> NdimageResult<Self> {
        if self.dimension != other.dimension {
            return Err(NdimageError::InvalidInput(format!(
                "Dimension mismatch: {} vs {}",
                self.dimension, other.dimension
            )));
        }

        // Create inverse of other vector (circular shift by dimension - index)
        let mut other_inverse_data = Vec::new();
        for &(idx, val) in &other.sparse_data {
            let inv_idx = if idx == 0 { 0 } else { self.dimension - idx };
            other_inverse_data.push((inv_idx, val));
        }

        let other_inverse = Self {
            sparse_data: other_inverse_data,
            dimension: other.dimension,
            norm: other.norm,
        };

        // Bind with the inverse
        self.bind(&other_inverse)
    }

    /// Normalize the hypervector
    ///
    /// # Returns
    ///
    /// A new normalized hypervector
    pub fn normalize(&self) -> Self {
        if self.norm <= 0.0 {
            return self.clone();
        }

        let normalized_data = self
            .sparse_data
            .iter()
            .map(|&(idx, val)| (idx, val / self.norm))
            .collect();

        Self {
            sparse_data: normalized_data,
            dimension: self.dimension,
            norm: 1.0,
        }
    }

    /// Scale the hypervector by a constant
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor
    ///
    /// # Returns
    ///
    /// A new scaled hypervector
    pub fn scale(&self, factor: f64) -> Self {
        let scaled_data = self
            .sparse_data
            .iter()
            .map(|&(idx, val)| (idx, val * factor))
            .collect();

        Self {
            sparse_data: scaled_data,
            dimension: self.dimension,
            norm: self.norm * factor.abs(),
        }
    }

    /// Threshold the hypervector, keeping only values above the threshold
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum absolute value to keep
    ///
    /// # Returns
    ///
    /// A new thresholded hypervector
    pub fn threshold(&self, threshold: f64) -> Self {
        let thresholded_data: Vec<(usize, f64)> = self
            .sparse_data
            .iter()
            .filter_map(|&(idx, val)| {
                if val.abs() > threshold {
                    Some((idx, val))
                } else {
                    None
                }
            })
            .collect();

        let norm = thresholded_data
            .iter()
            .map(|&(_, v)| v * v)
            .sum::<f64>()
            .sqrt();

        Self {
            sparse_data: thresholded_data,
            dimension: self.dimension,
            norm,
        }
    }

    /// Get the sparsity level of the hypervector
    ///
    /// # Returns
    ///
    /// Fraction of non-zero dimensions
    pub fn sparsity(&self) -> f64 {
        self.sparse_data.len() as f64 / self.dimension as f64
    }

    /// Check if the hypervector is empty (all zeros)
    ///
    /// # Returns
    ///
    /// True if the hypervector has no non-zero elements
    pub fn is_empty(&self) -> bool {
        self.sparse_data.is_empty()
    }

    /// Get the L2 norm of the hypervector
    ///
    /// # Returns
    ///
    /// The L2 norm value
    pub fn l2_norm(&self) -> f64 {
        self.norm
    }

    /// Create a zero hypervector
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the zero vector
    ///
    /// # Returns
    ///
    /// A hypervector with all zeros
    pub fn zeros(dim: usize) -> Self {
        Self {
            sparse_data: Vec::new(),
            dimension: dim,
            norm: 0.0,
        }
    }
}

/// Advanced vector operations
impl Hypervector {
    /// Compute Hamming distance with another hypervector
    ///
    /// # Arguments
    ///
    /// * `other` - The other hypervector to compare with
    ///
    /// # Returns
    ///
    /// Hamming distance (number of differing dimensions)
    pub fn hamming_distance(&self, other: &Self) -> usize {
        if self.dimension != other.dimension {
            return self.dimension; // Maximum distance
        }

        let mut distance = 0;
        let mut i = 0;
        let mut j = 0;

        // Count positions where both have values but they differ
        while i < self.sparse_data.len() && j < other.sparse_data.len() {
            let (self_idx, self_val) = self.sparse_data[i];
            let (other_idx, other_val) = other.sparse_data[j];

            if self_idx == other_idx {
                if (self_val > 0.0) != (other_val > 0.0) {
                    distance += 1;
                }
                i += 1;
                j += 1;
            } else if self_idx < other_idx {
                distance += 1; // Self has value, other doesn't
                i += 1;
            } else {
                distance += 1; // Other has value, self doesn't
                j += 1;
            }
        }

        // Count remaining elements
        distance += (self.sparse_data.len() - i) + (other.sparse_data.len() - j);

        distance
    }

    /// Compute overlap (intersection) with another hypervector
    ///
    /// # Arguments
    ///
    /// * `other` - The other hypervector to compute overlap with
    ///
    /// # Returns
    ///
    /// Number of dimensions where both vectors have non-zero values
    pub fn overlap(&self, other: &Self) -> usize {
        if self.dimension != other.dimension {
            return 0;
        }

        let mut overlap_count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < self.sparse_data.len() && j < other.sparse_data.len() {
            let (self_idx, _) = self.sparse_data[i];
            let (other_idx, _) = other.sparse_data[j];

            if self_idx == other_idx {
                overlap_count += 1;
                i += 1;
                j += 1;
            } else if self_idx < other_idx {
                i += 1;
            } else {
                j += 1;
            }
        }

        overlap_count
    }

    /// Create a permuted version of the hypervector
    ///
    /// # Arguments
    ///
    /// * `permutation` - Permutation mapping
    ///
    /// # Returns
    ///
    /// A new permuted hypervector
    pub fn permute(&self, permutation: &[usize]) -> NdimageResult<Self> {
        if permutation.len() != self.dimension {
            return Err(NdimageError::InvalidInput(
                "Permutation size must match vector dimension".into(),
            ));
        }

        let mut permuted_data = Vec::new();
        for &(idx, val) in &self.sparse_data {
            if idx < permutation.len() {
                permuted_data.push((permutation[idx], val));
            }
        }

        permuted_data.sort_by_key(|&(idx, _)| idx);

        Ok(Self {
            sparse_data: permuted_data,
            dimension: self.dimension,
            norm: self.norm,
        })
    }
}

/// Utility functions for vector operations
pub mod vector_utils {
    use super::*;

    /// Create a random permutation of given size
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the permutation
    ///
    /// # Returns
    ///
    /// A random permutation vector
    pub fn random_permutation(size: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..size).collect();
        let mut rng = scirs2_core::random::rng();
        perm.shuffle(&mut rng);
        perm
    }

    /// Bundle multiple hypervectors together
    ///
    /// # Arguments
    ///
    /// * `vectors` - Vector of hypervectors to bundle
    ///
    /// # Returns
    ///
    /// A new hypervector representing the bundle of all inputs
    pub fn bundle_multiple(vectors: &[Hypervector]) -> NdimageResult<Hypervector> {
        if vectors.is_empty() {
            return Err(NdimageError::InvalidInput("No vectors to bundle".into()));
        }

        let mut result = vectors[0].clone();
        for vector in vectors.iter().skip(1) {
            result = result.bundle(vector)?;
        }

        Ok(result)
    }

    /// Compute the centroid of multiple hypervectors
    ///
    /// # Arguments
    ///
    /// * `vectors` - Vector of hypervectors
    ///
    /// # Returns
    ///
    /// The centroid hypervector
    pub fn centroid(vectors: &[Hypervector]) -> NdimageResult<Hypervector> {
        if vectors.is_empty() {
            return Err(NdimageError::InvalidInput("No vectors provided".into()));
        }

        let bundled = bundle_multiple(vectors)?;
        let scale_factor = 1.0 / vectors.len() as f64;
        Ok(bundled.scale(scale_factor))
    }

    /// Weight a hypervector by a given factor
    ///
    /// # Arguments
    ///
    /// * `hv` - The hypervector to weight
    /// * `weight` - The weighting factor
    ///
    /// # Returns
    ///
    /// A new weighted hypervector
    pub fn weight_hypervector(hv: &Hypervector, weight: f64) -> Hypervector {
        hv.scale(weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hypervector_random_creation() {
        let hv = Hypervector::random(1000, 0.1);
        assert_eq!(hv.dimension, 1000);
        assert!(hv.sparse_data.len() > 90); // Around 10% sparsity
        assert!(hv.sparse_data.len() < 110);
        assert!(hv.norm > 0.0);
    }

    #[test]
    fn test_hypervector_from_dense() {
        let dense = Array1::from(vec![1.0, 0.0, -1.0, 0.5, 0.0]);
        let hv = Hypervector::from_dense(&dense, 0.1);

        assert_eq!(hv.dimension, 5);
        assert!(!hv.sparse_data.is_empty());
        assert!(hv.norm > 0.0);
    }

    #[test]
    fn test_hypervector_to_dense() {
        let hv = Hypervector {
            sparse_data: vec![(0, 1.0), (2, -1.0), (4, 0.5)],
            dimension: 5,
            norm: 1.5,
        };

        let dense = hv.to_dense();
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[1], 0.0);
        assert_eq!(dense[2], -1.0);
        assert_eq!(dense[3], 0.0);
        assert_eq!(dense[4], 0.5);
    }

    #[test]
    fn test_hypervector_similarity() {
        let hv1 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let hv2 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let hv3 = Hypervector {
            sparse_data: vec![(2, 1.0), (3, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        assert_abs_diff_eq!(hv1.similarity(&hv2), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hv1.similarity(&hv3), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypervector_bundle() {
        let hv1 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let hv2 = Hypervector {
            sparse_data: vec![(1, 1.0), (2, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let bundled = hv1.bundle(&hv2).unwrap();
        assert_eq!(bundled.dimension, 10);
        assert_eq!(bundled.sparse_data.len(), 3); // indices 0, 1, 2

        // Check specific values
        assert_eq!(bundled.sparse_data[0], (0, 1.0)); // Only from hv1
        assert_eq!(bundled.sparse_data[1], (1, 2.0)); // From both
        assert_eq!(bundled.sparse_data[2], (2, 1.0)); // Only from hv2
    }

    #[test]
    fn test_hypervector_bind() {
        let hv1 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let hv2 = Hypervector {
            sparse_data: vec![(1, 1.0), (2, 1.0)],
            dimension: 10,
            norm: 2.0_f64.sqrt(),
        };

        let bound = hv1.bind(&hv2).unwrap();
        assert_eq!(bound.dimension, 10);
        assert!(!bound.sparse_data.is_empty());
        assert!(bound.norm > 0.0);
    }

    #[test]
    fn test_hypervector_normalize() {
        let hv = Hypervector {
            sparse_data: vec![(0, 2.0), (1, 2.0)],
            dimension: 10,
            norm: 8.0_f64.sqrt(),
        };

        let normalized = hv.normalize();
        assert_abs_diff_eq!(normalized.norm, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            normalized.sparse_data[0].1,
            2.0 / 8.0_f64.sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_hypervector_scale() {
        let hv = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 2.0)],
            dimension: 10,
            norm: 5.0_f64.sqrt(),
        };

        let scaled = hv.scale(2.0);
        assert_eq!(scaled.sparse_data[0].1, 2.0);
        assert_eq!(scaled.sparse_data[1].1, 4.0);
        assert_abs_diff_eq!(scaled.norm, 2.0 * 5.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_hypervector_threshold() {
        let hv = Hypervector {
            sparse_data: vec![(0, 0.1), (1, 0.5), (2, 1.0)],
            dimension: 10,
            norm: 1.0,
        };

        let thresholded = hv.threshold(0.3);
        assert_eq!(thresholded.sparse_data.len(), 2); // Only values >= 0.3
        assert_eq!(thresholded.sparse_data[0], (1, 0.5));
        assert_eq!(thresholded.sparse_data[1], (2, 1.0));
    }

    #[test]
    fn test_hypervector_sparsity() {
        let hv = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0)],
            dimension: 10,
            norm: 1.0,
        };

        assert_abs_diff_eq!(hv.sparsity(), 0.2, epsilon = 1e-10); // 2/10
    }

    #[test]
    fn test_hypervector_hamming_distance() {
        let hv1 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0), (2, -1.0)],
            dimension: 10,
            norm: 1.0,
        };

        let hv2 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, -1.0), (3, 1.0)],
            dimension: 10,
            norm: 1.0,
        };

        let distance = hv1.hamming_distance(&hv2);
        assert_eq!(distance, 3); // Different at positions 1, 2, 3
    }

    #[test]
    fn test_hypervector_overlap() {
        let hv1 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, 1.0), (2, -1.0)],
            dimension: 10,
            norm: 1.0,
        };

        let hv2 = Hypervector {
            sparse_data: vec![(0, 1.0), (1, -1.0), (3, 1.0)],
            dimension: 10,
            norm: 1.0,
        };

        let overlap = hv1.overlap(&hv2);
        assert_eq!(overlap, 2); // Both have values at positions 0 and 1
    }

    #[test]
    fn test_bundle_multiple() {
        use vector_utils::*;

        let hv1 = Hypervector::random(100, 0.1);
        let hv2 = Hypervector::random(100, 0.1);
        let hv3 = Hypervector::random(100, 0.1);

        let vectors = vec![hv1, hv2, hv3];
        let bundled = bundle_multiple(&vectors).unwrap();

        assert_eq!(bundled.dimension, 100);
        assert!(bundled.norm > 0.0);
    }

    #[test]
    fn test_centroid() {
        use vector_utils::*;

        let hv1 = Hypervector::random(100, 0.1);
        let hv2 = Hypervector::random(100, 0.1);

        let vectors = vec![hv1, hv2];
        let centroid_hv = centroid(&vectors).unwrap();

        assert_eq!(centroid_hv.dimension, 100);
        assert!(centroid_hv.norm > 0.0);
    }
}
