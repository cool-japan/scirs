//! Data partitioning strategies for distributed clustering
//!
//! This module provides various strategies for partitioning data across
//! multiple worker nodes in distributed clustering algorithms.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::random::seq::SliceRandom;
use std::fmt::Debug;

use super::fault_tolerance::DataPartition;
use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Data partitioning coordinator for distributed clustering
#[derive(Debug)]
pub struct DataPartitioner<F: Float> {
    pub config: PartitioningConfig,
    pub partitions: Vec<DataPartition<F>>,
    pub partition_stats: PartitioningStatistics,
}

/// Configuration for data partitioning
#[derive(Debug, Clone)]
pub struct PartitioningConfig {
    pub n_workers: usize,
    pub strategy: PartitioningStrategy,
    pub balance_threshold: f64,
    pub enable_load_balancing: bool,
    pub min_partition_size: usize,
    pub max_partition_size: Option<usize>,
    pub preserve_locality: bool,
    pub random_seed: Option<u64>,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            strategy: PartitioningStrategy::Random,
            balance_threshold: 0.1,
            enable_load_balancing: true,
            min_partition_size: 100,
            max_partition_size: None,
            preserve_locality: false,
            random_seed: None,
        }
    }
}

/// Available partitioning strategies
#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    /// Random partitioning
    Random,
    /// Round-robin partitioning
    RoundRobin,
    /// Stratified partitioning based on preliminary clustering
    Stratified { n_strata: usize },
    /// Hash-based partitioning
    Hash,
    /// Range-based partitioning
    Range { feature_index: usize },
    /// Locality-preserving partitioning
    LocalityPreserving { similarity_threshold: f64 },
    /// Custom partitioning with user-defined function
    Custom,
}

/// Statistics about the partitioning
#[derive(Debug, Default)]
pub struct PartitioningStatistics {
    pub partition_sizes: Vec<usize>,
    pub load_balance_score: f64,
    pub locality_score: f64,
    pub partitioning_time_ms: u64,
    pub memory_usage_bytes: usize,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> DataPartitioner<F> {
    /// Create new data partitioner
    pub fn new(config: PartitioningConfig) -> Self {
        Self {
            config,
            partitions: Vec::new(),
            partition_stats: PartitioningStatistics::default(),
        }
    }

    /// Partition data according to the configured strategy
    pub fn partition_data(&mut self, data: ArrayView2<F>) -> Result<Vec<DataPartition<F>>> {
        let start_time = std::time::Instant::now();

        // Calculate target partition sizes
        let partition_sizes = self.calculate_partition_sizes(data.nrows())?;

        // Apply partitioning strategy
        let partitions = match &self.config.strategy {
            PartitioningStrategy::Random => self.random_partition(data, &partition_sizes),
            PartitioningStrategy::RoundRobin => self.round_robin_partition(data, &partition_sizes),
            PartitioningStrategy::Stratified { n_strata } => {
                self.stratified_partition(data, &partition_sizes, *n_strata)
            }
            PartitioningStrategy::Hash => self.hash_partition(data, &partition_sizes),
            PartitioningStrategy::Range { feature_index } => {
                self.range_partition(data, &partition_sizes, *feature_index)
            }
            PartitioningStrategy::LocalityPreserving {
                similarity_threshold,
            } => self.locality_preserving_partition(data, &partition_sizes, *similarity_threshold),
            PartitioningStrategy::Custom => self.custom_partition(data, &partition_sizes),
        }?;

        // Update statistics
        let partitioning_time = start_time.elapsed().as_millis() as u64;
        self.update_statistics(&partitions, partitioning_time);

        self.partitions = partitions.clone();
        Ok(partitions)
    }

    /// Calculate target sizes for each partition
    fn calculate_partition_sizes(&self, totalsize: usize) -> Result<Vec<usize>> {
        if self.config.n_workers == 0 {
            return Err(ClusteringError::InvalidInput(
                "Number of workers must be greater than 0".to_string(),
            ));
        }

        let base_size = totalsize / self.config.n_workers;
        let remainder = totalsize % self.config.n_workers;

        let mut sizes = vec![base_size; self.config.n_workers];

        // Distribute remainder across first few workers
        for i in 0..remainder {
            sizes[i] += 1;
        }

        // Adjust for minimum partition size constraints
        // If total is less than n_workers * min_partition_size, we can't satisfy the constraint
        // so we use the calculated sizes instead
        let effective_min_size = self
            .config
            .min_partition_size
            .min(totalsize / self.config.n_workers + 1);

        for size in &mut sizes {
            if *size < effective_min_size {
                *size = effective_min_size;
            }
            if let Some(max_size) = self.config.max_partition_size {
                if *size > max_size {
                    *size = max_size;
                }
            }
        }

        // Ensure the total doesn't exceed the original totalsize
        let current_total: usize = sizes.iter().sum();
        if current_total > totalsize {
            // Redistribute to match totalsize exactly
            let mut sizes = vec![totalsize / self.config.n_workers; self.config.n_workers];
            let remainder = totalsize % self.config.n_workers;
            for i in 0..remainder {
                sizes[i] += 1;
            }
            return Ok(sizes);
        }

        Ok(sizes)
    }

    /// Random partitioning strategy
    fn random_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_samples = data.nrows();
        let n_workers = self.config.n_workers;

        // Create random permutation of indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = scirs2_core::random::rng();
        indices.shuffle(&mut rng);

        let mut partitions = Vec::new();
        let mut start_idx = 0;

        for (worker_id, &partition_size) in partition_sizes.iter().enumerate() {
            let end_idx = (start_idx + partition_size).min(n_samples);

            if start_idx < end_idx {
                let mut partition_data = Array2::zeros((end_idx - start_idx, data.ncols()));

                for (i, &data_idx) in indices[start_idx..end_idx].iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(data_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }

            start_idx = end_idx;
            if start_idx >= n_samples {
                break;
            }
        }

        Ok(partitions)
    }

    /// Stratified partitioning using preliminary clustering
    fn stratified_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
        n_strata: usize,
    ) -> Result<Vec<DataPartition<F>>> {
        let n_samples = data.nrows();

        if n_samples < n_strata {
            // Fall back to random if not enough data
            return self.random_partition(data, partition_sizes);
        }

        // Step 1: Perform preliminary clustering to identify strata
        let strata_assignments = self.identify_strata(data, n_strata)?;

        // Step 2: Group data points by stratum
        let mut strata_groups: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
        for (point_idx, &stratum_id) in strata_assignments.iter().enumerate() {
            strata_groups[stratum_id].push(point_idx);
        }

        // Step 3: Distribute strata points proportionally to workers
        let mut worker_assignments: Vec<Vec<usize>> = vec![Vec::new(); self.config.n_workers];

        for stratum_points in strata_groups.iter() {
            if stratum_points.is_empty() {
                continue;
            }

            // Calculate how many points each worker should get from this stratum
            let total_points = stratum_points.len();
            let mut distributed = 0;

            for worker_id in 0..self.config.n_workers {
                let target_size = partition_sizes[worker_id];
                let current_size = worker_assignments[worker_id].len();
                let remaining_capacity = target_size.saturating_sub(current_size);

                // Proportional allocation with remaining capacity constraint
                let total_remaining_capacity: usize = worker_assignments
                    .iter()
                    .enumerate()
                    .skip(worker_id)
                    .map(|(i, assignments)| partition_sizes[i].saturating_sub(assignments.len()))
                    .sum();

                let points_for_worker = if total_remaining_capacity == 0 {
                    0
                } else {
                    let proportion = remaining_capacity as f64 / total_remaining_capacity as f64;
                    let remaining_points = total_points - distributed;
                    ((remaining_points as f64 * proportion).round() as usize)
                        .min(remaining_points)
                        .min(remaining_capacity)
                };

                // Assign points to this worker
                let start_idx = distributed;
                let end_idx = (start_idx + points_for_worker).min(total_points);

                for &point_idx in &stratum_points[start_idx..end_idx] {
                    worker_assignments[worker_id].push(point_idx);
                }

                distributed = end_idx;

                if distributed >= total_points {
                    break;
                }
            }
        }

        // Step 4: Create partitions from worker assignments
        let mut partitions = Vec::new();
        for (worker_id, point_indices) in worker_assignments.into_iter().enumerate() {
            if !point_indices.is_empty() {
                let mut partition_data = Array2::zeros((point_indices.len(), data.ncols()));

                for (i, &point_idx) in point_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(point_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }
        }

        Ok(partitions)
    }

    /// Identify strata using simple K-means clustering
    fn identify_strata(&self, data: ArrayView2<F>, nstrata: usize) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Initialize centroids randomly
        let mut rng = scirs2_core::random::rng();
        let mut point_indices: Vec<usize> = (0..n_samples).collect();
        point_indices.shuffle(&mut rng);

        let mut centroids = Array2::zeros((nstrata, n_features));
        for (i, &point_idx) in point_indices.iter().take(nstrata).enumerate() {
            centroids.row_mut(i).assign(&data.row(point_idx));
        }

        let mut assignments = Array1::zeros(n_samples);
        let max_iterations = 10; // Quick preliminary clustering

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for (point_idx, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                let mut best_centroid = 0;

                for (centroid_idx, centroid) in centroids.rows().into_iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_centroid = centroid_idx;
                    }
                }

                if assignments[point_idx] != best_centroid {
                    assignments[point_idx] = best_centroid;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            centroids.fill(F::zero());
            let mut counts = vec![0; nstrata];

            for (point_idx, point) in data.rows().into_iter().enumerate() {
                let cluster_id = assignments[point_idx];
                for (j, &value) in point.iter().enumerate() {
                    centroids[[cluster_id, j]] = centroids[[cluster_id, j]] + value;
                }
                counts[cluster_id] += 1;
            }

            // Compute averages
            for i in 0..nstrata {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        centroids[[i, j]] = centroids[[i, j]] / F::from(counts[i]).unwrap();
                    }
                }
            }
        }

        Ok(assignments)
    }

    /// Round-robin partitioning
    fn round_robin_partition(
        &self,
        data: ArrayView2<F>,
        _partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_workers = self.config.n_workers;
        let mut worker_data: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Assign points in round-robin fashion
        for (row_idx, _) in data.rows().into_iter().enumerate() {
            let worker_id = row_idx % n_workers;
            worker_data[worker_id].push(row_idx);
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_data.into_iter().enumerate() {
            if !row_indices.is_empty() {
                let mut partition_data = Array2::zeros((row_indices.len(), data.ncols()));

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    workerid: worker_id,
                    weight: row_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Hash-based partitioning using feature hash
    fn hash_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_workers = self.config.n_workers;
        let mut worker_assignments: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Hash each data point to a worker
        for (row_idx, row) in data.rows().into_iter().enumerate() {
            // Simple hash based on first feature (can be improved)
            let hash_value = if !row.is_empty() {
                (row[0].to_f64().unwrap_or(0.0) * 1000.0) as u64
            } else {
                row_idx as u64
            };

            let worker_id = (hash_value % n_workers as u64) as usize;
            worker_assignments[worker_id].push(row_idx);
        }

        // Create partitions with size balancing
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_assignments.into_iter().enumerate() {
            // Limit partition size if needed
            let max_size = partition_sizes
                .get(worker_id)
                .copied()
                .unwrap_or(row_indices.len());
            let actual_indices = if row_indices.len() > max_size {
                &row_indices[..max_size]
            } else {
                &row_indices
            };

            if !actual_indices.is_empty() {
                let mut partition_data = Array2::zeros((actual_indices.len(), data.ncols()));

                for (i, &row_idx) in actual_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }
        }

        Ok(partitions)
    }

    /// Range-based partitioning on a specific feature
    fn range_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
        feature_index: usize,
    ) -> Result<Vec<DataPartition<F>>> {
        if feature_index >= data.ncols() {
            return Err(ClusteringError::InvalidInput(
                "Feature index out of bounds".to_string(),
            ));
        }

        // Extract feature values and sort indices
        let mut indexed_values: Vec<(usize, F)> = data
            .column(feature_index)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Partition based on sorted order
        let mut partitions = Vec::new();
        let mut start_idx = 0;

        for (worker_id, &partition_size) in partition_sizes.iter().enumerate() {
            let end_idx = (start_idx + partition_size).min(indexed_values.len());

            if start_idx < end_idx {
                let mut partition_data = Array2::zeros((end_idx - start_idx, data.ncols()));

                for (i, &(original_idx, _)) in indexed_values[start_idx..end_idx].iter().enumerate()
                {
                    partition_data.row_mut(i).assign(&data.row(original_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }

            start_idx = end_idx;
            if start_idx >= indexed_values.len() {
                break;
            }
        }

        Ok(partitions)
    }

    /// Locality-preserving partitioning based on similarity
    fn locality_preserving_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
        similarity_threshold: f64,
    ) -> Result<Vec<DataPartition<F>>> {
        let n_samples = data.nrows();
        let mut assigned: Vec<bool> = vec![false; n_samples];
        let mut worker_assignments: Vec<Vec<usize>> = vec![Vec::new(); self.config.n_workers];

        let mut current_worker = 0;
        let mut unassigned_points: Vec<usize> = (0..n_samples).collect();

        while !unassigned_points.is_empty() && current_worker < self.config.n_workers {
            let target_size = partition_sizes[current_worker];
            let mut current_partition = Vec::new();

            // Start with a random unassigned point
            if let Some(seed_idx) = unassigned_points.first().copied() {
                current_partition.push(seed_idx);
                assigned[seed_idx] = true;
                unassigned_points.retain(|&x| x != seed_idx);

                // Grow partition by adding similar points
                while current_partition.len() < target_size && !unassigned_points.is_empty() {
                    let mut best_similarity = 0.0;
                    let mut best_candidate = None;

                    // Find most similar unassigned point to any point in current partition
                    for &candidate_idx in &unassigned_points {
                        let candidate_point = data.row(candidate_idx);

                        for &partition_point_idx in &current_partition {
                            let partition_point = data.row(partition_point_idx);
                            let distance = euclidean_distance(candidate_point, partition_point)
                                .to_f64()
                                .unwrap_or(f64::INFINITY);
                            let similarity = 1.0 / (1.0 + distance); // Convert distance to similarity

                            if similarity > best_similarity && similarity >= similarity_threshold {
                                best_similarity = similarity;
                                best_candidate = Some(candidate_idx);
                            }
                        }
                    }

                    if let Some(best_idx) = best_candidate {
                        current_partition.push(best_idx);
                        assigned[best_idx] = true;
                        unassigned_points.retain(|&x| x != best_idx);
                    } else {
                        // No similar points found, add random points to fill partition
                        while current_partition.len() < target_size && !unassigned_points.is_empty()
                        {
                            let random_idx = unassigned_points.remove(0);
                            current_partition.push(random_idx);
                            assigned[random_idx] = true;
                        }
                        break;
                    }
                }

                worker_assignments[current_worker] = current_partition;
            }

            current_worker += 1;
        }

        // Assign any remaining points to workers with space
        for remaining_idx in unassigned_points {
            for worker_id in 0..self.config.n_workers {
                if worker_assignments[worker_id].len() < partition_sizes[worker_id] {
                    worker_assignments[worker_id].push(remaining_idx);
                    break;
                }
            }
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, point_indices) in worker_assignments.into_iter().enumerate() {
            if !point_indices.is_empty() {
                let mut partition_data = Array2::zeros((point_indices.len(), data.ncols()));

                for (i, &point_idx) in point_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(point_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }
        }

        Ok(partitions)
    }

    /// Custom partitioning (placeholder for user-defined strategies)
    fn custom_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        // Default to random partitioning for custom strategy
        // In a real implementation, this would allow user-defined partitioning functions
        self.random_partition(data, partition_sizes)
    }

    /// Update partitioning statistics
    fn update_statistics(&mut self, partitions: &[DataPartition<F>], partitioning_timems: u64) {
        self.partition_stats.partition_sizes = partitions.iter().map(|p| p.data.nrows()).collect();
        self.partition_stats.partitioning_time_ms = partitioning_timems;

        // Calculate load balance score (1.0 = perfectly balanced, 0.0 = completely imbalanced)
        if !self.partition_stats.partition_sizes.is_empty() {
            let avg_size = self.partition_stats.partition_sizes.iter().sum::<usize>() as f64
                / self.partition_stats.partition_sizes.len() as f64;
            let variance = self
                .partition_stats
                .partition_sizes
                .iter()
                .map(|&size| (size as f64 - avg_size).powi(2))
                .sum::<f64>()
                / self.partition_stats.partition_sizes.len() as f64;

            self.partition_stats.load_balance_score = if avg_size > 0.0 {
                1.0 - (variance.sqrt() / avg_size).min(1.0)
            } else {
                0.0
            };
        }

        // Calculate memory usage (approximate)
        self.partition_stats.memory_usage_bytes = partitions
            .iter()
            .map(|p| p.data.len() * std::mem::size_of::<F>())
            .sum();
    }

    /// Get partitioning statistics
    pub fn get_statistics(&self) -> &PartitioningStatistics {
        &self.partition_stats
    }

    /// Get current partitions
    pub fn get_partitions(&self) -> &[DataPartition<F>] {
        &self.partitions
    }

    /// Validate partition balance
    pub fn validate_partition_balance(&self) -> bool {
        self.partition_stats.load_balance_score >= (1.0 - self.config.balance_threshold)
    }

    /// Rebalance partitions if needed
    pub fn rebalance_if_needed(&mut self, data: ArrayView2<F>) -> Result<bool> {
        if !self.config.enable_load_balancing || self.validate_partition_balance() {
            return Ok(false);
        }

        // Re-partition the data
        self.partition_data(data)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_data_partitioner_creation() {
        let config = PartitioningConfig::default();
        let partitioner = DataPartitioner::<f64>::new(config);

        assert_eq!(partitioner.config.n_workers, 4);
        assert!(partitioner.partitions.is_empty());
    }

    #[test]
    fn test_calculate_partition_sizes() {
        let config = PartitioningConfig {
            n_workers: 3,
            min_partition_size: 1, // Set a reasonable min_partition_size for the test
            ..Default::default()
        };
        let partitioner = DataPartitioner::<f64>::new(config);

        let sizes = partitioner.calculate_partition_sizes(100).unwrap();
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes.iter().sum::<usize>(), 100);

        // Should be approximately balanced
        let max_diff = sizes.iter().max().unwrap() - sizes.iter().min().unwrap();
        assert!(max_diff <= 1);
    }

    #[test]
    fn test_random_partitioning() {
        let config = PartitioningConfig {
            n_workers: 2,
            strategy: PartitioningStrategy::Random,
            min_partition_size: 1, // Set a reasonable min_partition_size for the test
            ..Default::default()
        };
        let mut partitioner = DataPartitioner::new(config);

        let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect()).unwrap();
        let partitions = partitioner.partition_data(data.view()).unwrap();

        assert_eq!(partitions.len(), 2);
        assert!(partitions.iter().all(|p| p.data.nrows() > 0));

        let total_points: usize = partitions.iter().map(|p| p.data.nrows()).sum();
        assert_eq!(total_points, 100);
    }

    #[test]
    fn test_round_robin_partitioning() {
        let config = PartitioningConfig {
            n_workers: 3,
            strategy: PartitioningStrategy::RoundRobin,
            ..Default::default()
        };
        let mut partitioner = DataPartitioner::new(config);

        let data = Array2::from_shape_vec((99, 2), (0..198).map(|x| x as f64).collect()).unwrap();
        let partitions = partitioner.partition_data(data.view()).unwrap();

        assert_eq!(partitions.len(), 3);
        assert_eq!(partitions[0].data.nrows(), 33);
        assert_eq!(partitions[1].data.nrows(), 33);
        assert_eq!(partitions[2].data.nrows(), 33);
    }

    #[test]
    fn test_load_balance_score() {
        let config = PartitioningConfig::default();
        let mut partitioner = DataPartitioner::<f64>::new(config);

        // Perfect balance - create mock partitions
        let balanced_partitions: Vec<DataPartition<f64>> = (0..4)
            .map(|i| DataPartition::new(i, Array2::zeros((25, 2)), i))
            .collect();
        partitioner.update_statistics(&balanced_partitions, 0);
        assert!((partitioner.partition_stats.load_balance_score - 1.0).abs() < 0.01);

        // Imbalanced - create imbalanced mock partitions
        let imbalanced_partitions = vec![
            DataPartition::new(0, Array2::zeros((10, 2)), 0),
            DataPartition::new(1, Array2::zeros((90, 2)), 1),
        ];
        partitioner.update_statistics(&imbalanced_partitions, 0);
        assert!(partitioner.partition_stats.load_balance_score < 0.5);
    }

    #[test]
    fn test_partition_size_constraints() {
        let config = PartitioningConfig {
            n_workers: 3,
            min_partition_size: 10,
            max_partition_size: Some(50),
            ..Default::default()
        };
        let partitioner = DataPartitioner::<f64>::new(config);

        let sizes = partitioner.calculate_partition_sizes(120).unwrap();
        assert!(sizes.iter().all(|&size| size >= 10 && size <= 50));
    }
}
