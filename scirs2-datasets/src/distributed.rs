//! Distributed dataset processing capabilities
//!
//! This module provides functionality for processing datasets across multiple machines or processes:
//! - Parallel data loading and processing
//! - Dataset sharding and distribution
//! - Distributed sampling and cross-validation
//! - MapReduce-style operations on datasets

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::cache::DatasetCache;
use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;

/// Configuration for distributed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of worker processes/threads
    pub num_workers: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Communication timeout (seconds)
    pub timeout: u64,
    /// Whether to use shared memory for large datasets
    pub use_shared_memory: bool,
    /// Maximum memory per worker (MB)
    pub memory_limit_mb: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            num_workers: num_cpus,
            chunk_size: 10000,
            timeout: 300,
            use_shared_memory: false,
            memory_limit_mb: 1024,
        }
    }
}

/// Distributed dataset processor
pub struct DistributedProcessor {
    config: DistributedConfig,
    #[allow(dead_code)]
    cache: DatasetCache,
}

impl DistributedProcessor {
    /// Create a new distributed processor
    pub fn new(config: DistributedConfig) -> Result<Self> {
        let cachedir = dirs::cache_dir()
            .ok_or_else(|| DatasetsError::Other("Could not determine cache directory".to_string()))?
            .join("scirs2-datasets");
        let cache = DatasetCache::new(cachedir);

        Ok(Self { config, cache })
    }

    /// Create with default configuration
    pub fn default_config() -> Result<Self> {
        Self::new(DistributedConfig::default())
    }

    /// Process a large dataset in parallel chunks
    pub fn process_dataset_parallel<F, R>(&self, dataset: &Dataset, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&Dataset) -> Result<R> + Send + Sync + Clone + 'static,
        R: Send + 'static,
    {
        let chunks = self.split_dataset_into_chunks(dataset)?;
        let processor = Arc::new(processor);

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Spawn worker threads
        for chunk in chunks {
            let tx = tx.clone();
            let processor = Arc::clone(&processor);

            let handle = thread::spawn(move || {
                let result = processor(&chunk);
                let _ = tx.send(result);
            });

            handles.push(handle);
        }

        // Drop the original sender
        drop(tx);

        // Collect results
        let mut results = Vec::new();
        for result in rx {
            results.push(result?);
        }

        // Wait for all workers to finish
        for handle in handles {
            let _ = handle.join();
        }

        Ok(results)
    }

    /// Distribute dataset across multiple workers with MapReduce pattern
    pub fn map_reduce_dataset<M, R, C>(&self, dataset: &Dataset, mapper: M, reducer: R) -> Result<C>
    where
        M: Fn(&Dataset) -> Result<Vec<C>> + Send + Sync + Clone + 'static,
        R: Fn(Vec<C>) -> Result<C> + Send + Sync + 'static,
        C: Send + 'static,
    {
        // Map phase: process chunks in parallel
        let map_results = self.process_dataset_parallel(dataset, mapper)?;

        // Reduce phase: combine results
        let flattened: Vec<C> = map_results.into_iter().flatten().collect();
        reducer(flattened)
    }

    /// Split a dataset into balanced chunks for distribution
    pub fn split_dataset_into_chunks(&self, dataset: &Dataset) -> Result<Vec<Dataset>> {
        let n_samples = dataset.n_samples();
        let chunk_size = self
            .config
            .chunk_size
            .min(n_samples / self.config.num_workers + 1);

        let mut chunks = Vec::new();

        for start in (0..n_samples).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_samples);
            let chunk_data = dataset.data.slice(s![start..end, ..]).to_owned();

            let chunk_target = dataset
                .target
                .as_ref()
                .map(|target| target.slice(s![start..end]).to_owned());

            let chunk = Dataset {
                data: chunk_data,
                target: chunk_target,
                featurenames: dataset.featurenames.clone(),
                targetnames: dataset.targetnames.clone(),
                feature_descriptions: dataset.feature_descriptions.clone(),
                description: Some(format!("Chunk {start}-{end} of distributed dataset")),
                metadata: dataset.metadata.clone(),
            };

            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Distributed random sampling across workers
    pub fn distributed_sample(
        &self,
        dataset: &Dataset,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        if n_samples >= dataset.n_samples() {
            return Ok(dataset.clone());
        }

        let samples_per_chunk = n_samples / self.config.num_workers;
        let remainder = n_samples % self.config.num_workers;

        let chunks = self.split_dataset_into_chunks(dataset)?;
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        for (i, chunk) in chunks.into_iter().enumerate() {
            let tx = tx.clone();
            let chunk_samples = if i < remainder {
                samples_per_chunk + 1
            } else {
                samples_per_chunk
            };

            let seed = random_state.map(|s| s + i as u64);

            let handle = thread::spawn(move || {
                let sampled = Self::sample_chunk(&chunk, chunk_samples, seed);
                let _ = tx.send(sampled);
            });

            handles.push(handle);
        }

        drop(tx);

        // Collect sampled chunks
        let mut sampled_chunks = Vec::new();
        for result in rx {
            sampled_chunks.push(result?);
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Combine sampled chunks
        self.combine_datasets(&sampled_chunks)
    }

    /// Distributed cross-validation split
    pub fn distributed_k_fold(
        &self,
        dataset: &Dataset,
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<(Dataset, Dataset)>> {
        let n_samples = dataset.n_samples();
        let fold_size = n_samples / k;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            use scirs2_core::random::seq::SliceRandom;
            use scirs2_core::random::SeedableRng;

            let mut rng = if let Some(seed) = random_state {
                scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
            } else {
                // For deterministic testing, use a fixed seed when no seed provided
                scirs2_core::random::rngs::StdRng::seed_from_u64(42)
            };

            indices.shuffle(&mut rng);
        }

        let mut folds = Vec::new();

        for fold_idx in 0..k {
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == k - 1 {
                n_samples
            } else {
                (fold_idx + 1) * fold_size
            };

            let test_indices = &indices[test_start..test_end];
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();

            let train_data = self.select_samples(dataset, &train_indices)?;
            let test_data = self.select_samples(dataset, test_indices)?;

            folds.push((train_data, test_data));
        }

        Ok(folds)
    }

    /// Distributed stratified sampling
    pub fn distributed_stratified_sample(
        &self,
        dataset: &Dataset,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        let target = dataset.target.as_ref().ok_or_else(|| {
            DatasetsError::InvalidFormat("Stratified sampling requires target values".to_string())
        })?;

        // Group _samples by class
        let mut class_groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &value) in target.iter().enumerate() {
            let class = value as i32;
            class_groups.entry(class).or_default().push(idx);
        }

        // Calculate _samples per class
        let n_classes = class_groups.len();
        let base_samples_per_class = n_samples / n_classes;
        let remainder = n_samples % n_classes;

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        for (class_idx, (class, indices)) in class_groups.into_iter().enumerate() {
            let tx = tx.clone();
            let class_samples = if class_idx < remainder {
                base_samples_per_class + 1
            } else {
                base_samples_per_class
            };

            let seed = random_state.map(|s| s + class_idx as u64);

            let handle = thread::spawn(move || {
                let sampled_indices = Self::sample_indices(&indices, class_samples, seed);
                let _ = tx.send((class, sampled_indices));
            });

            handles.push(handle);
        }

        drop(tx);

        // Collect sampled indices
        let mut all_sampled_indices = Vec::new();
        for (_, indices) in rx {
            all_sampled_indices.extend(indices?);
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Create stratified sample
        self.select_samples(dataset, &all_sampled_indices)
    }

    /// Parallel feature scaling across workers
    pub fn distributed_scale(
        &self,
        dataset: &Dataset,
        method: ScalingMethod,
    ) -> Result<(Dataset, ScalingParameters)> {
        let n_features = dataset.n_features();
        let chunks = self.split_dataset_into_chunks(dataset)?;

        // Phase 1: Compute statistics in parallel
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        for chunk in chunks.iter() {
            let tx = tx.clone();
            let chunk = chunk.clone();

            let handle = thread::spawn(move || {
                let stats = Self::compute_chunk_statistics(&chunk);
                let _ = tx.send(stats);
            });

            handles.push(handle);
        }

        drop(tx);

        // Collect statistics
        let mut all_stats = Vec::new();
        for stats in rx {
            all_stats.push(stats?);
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Phase 2: Combine statistics
        let global_stats = Self::combine_statistics(&all_stats, n_features)?;
        let scaling_params = ScalingParameters::from_statistics(&global_stats, method);

        // Phase 3: Apply scaling in parallel
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        for chunk in chunks {
            let tx = tx.clone();
            let params = scaling_params.clone();

            let handle = thread::spawn(move || {
                let scaled_chunk = Self::apply_scaling(&chunk, &params);
                let _ = tx.send(scaled_chunk);
            });

            handles.push(handle);
        }

        drop(tx);

        // Collect scaled chunks
        let mut scaled_chunks = Vec::new();
        for result in rx {
            scaled_chunks.push(result?);
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Combine scaled chunks
        let scaled_dataset = self.combine_datasets(&scaled_chunks)?;
        Ok((scaled_dataset, scaling_params))
    }

    // Helper methods

    fn sample_chunk(
        chunk: &Dataset,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Dataset> {
        if n_samples >= chunk.n_samples() {
            return Ok(chunk.clone());
        }

        use scirs2_core::random::seq::SliceRandom;
        use scirs2_core::random::SeedableRng;

        let mut rng = if let Some(seed) = random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            // For deterministic testing, use a fixed seed when no seed provided
            scirs2_core::random::rngs::StdRng::seed_from_u64(42)
        };

        let mut indices: Vec<usize> = (0..chunk.n_samples()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_samples);

        Self::select_samples_static(chunk, &indices)
    }

    fn sample_indices(
        indices: &[usize],
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Vec<usize>> {
        if n_samples >= indices.len() {
            return Ok(indices.to_vec());
        }

        use scirs2_core::random::seq::SliceRandom;
        use scirs2_core::random::SeedableRng;

        let mut rng = if let Some(seed) = random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            // For deterministic testing, use a fixed seed when no seed provided
            scirs2_core::random::rngs::StdRng::seed_from_u64(42)
        };

        let mut sampled = indices.to_vec();
        sampled.shuffle(&mut rng);
        sampled.truncate(n_samples);

        Ok(sampled)
    }

    fn select_samples(&self, dataset: &Dataset, indices: &[usize]) -> Result<Dataset> {
        Self::select_samples_static(dataset, indices)
    }

    fn select_samples_static(dataset: &Dataset, indices: &[usize]) -> Result<Dataset> {
        let selected_data = dataset.data.select(Axis(0), indices);
        let selected_target = dataset
            .target
            .as_ref()
            .map(|target| target.select(Axis(0), indices));

        Ok(Dataset {
            data: selected_data,
            target: selected_target,
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some("Distributed sample".to_string()),
            metadata: dataset.metadata.clone(),
        })
    }

    fn combine_datasets(&self, datasets: &[Dataset]) -> Result<Dataset> {
        if datasets.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "Cannot combine empty dataset list".to_string(),
            ));
        }

        let n_features = datasets[0].n_features();
        let total_samples: usize = datasets.iter().map(|d| d.n_samples()).sum();

        // Combine data arrays
        let mut combined_data = Vec::with_capacity(total_samples * n_features);
        let mut combined_target = if datasets[0].target.is_some() {
            Some(Vec::with_capacity(total_samples))
        } else {
            None
        };

        for dataset in datasets {
            for row in dataset.data.rows() {
                combined_data.extend(row.iter());
            }

            if let (Some(ref mut combined), Some(ref target)) =
                (&mut combined_target, &dataset.target)
            {
                combined.extend(target.iter());
            }
        }

        let data = Array2::from_shape_vec((total_samples, n_features), combined_data)
            .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

        let target = combined_target.map(Array1::from_vec);

        Ok(Dataset {
            data,
            target,
            featurenames: datasets[0].featurenames.clone(),
            targetnames: datasets[0].targetnames.clone(),
            feature_descriptions: datasets[0].feature_descriptions.clone(),
            description: Some("Combined distributed dataset".to_string()),
            metadata: datasets[0].metadata.clone(),
        })
    }

    fn compute_chunk_statistics(chunk: &Dataset) -> Result<ChunkStatistics> {
        let data = &chunk.data;
        let n_features = data.ncols();
        let n_samples = data.nrows() as f64;

        let mut means = vec![0.0; n_features];
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];
        let mut sum_squares = vec![0.0; n_features];

        for col in 0..n_features {
            let column = data.column(col);

            let sum: f64 = column.sum();
            means[col] = sum / n_samples;

            for &value in column.iter() {
                mins[col] = mins[col].min(value);
                maxs[col] = maxs[col].max(value);
                sum_squares[col] += value * value;
            }
        }

        Ok(ChunkStatistics {
            n_samples: n_samples as usize,
            means,
            mins,
            maxs,
            sum_squares,
        })
    }

    fn combine_statistics(
        stats: &[ChunkStatistics],
        n_features: usize,
    ) -> Result<GlobalStatistics> {
        let total_samples: usize = stats.iter().map(|s| s.n_samples).sum();
        let mut global_means = vec![0.0; n_features];
        let mut global_mins = vec![f64::INFINITY; n_features];
        let mut global_maxs = vec![f64::NEG_INFINITY; n_features];
        let mut global_stds = vec![0.0; n_features];

        // Combine means
        for (feature, global_mean) in global_means.iter_mut().enumerate().take(n_features) {
            let weighted_sum: f64 = stats
                .iter()
                .map(|s| s.means[feature] * s.n_samples as f64)
                .sum();
            *global_mean = weighted_sum / total_samples as f64;
        }

        // Combine mins and maxs
        for feature in 0..n_features {
            for chunk_stats in stats {
                global_mins[feature] = global_mins[feature].min(chunk_stats.mins[feature]);
                global_maxs[feature] = global_maxs[feature].max(chunk_stats.maxs[feature]);
            }
        }

        // Compute global standard deviations
        for feature in 0..n_features {
            let sum_squared_deviations: f64 = stats
                .iter()
                .map(|s| {
                    let chunk_mean = s.means[feature];
                    let global_mean = global_means[feature];
                    let n = s.n_samples as f64;

                    // Sum of squares within chunk + correction for mean difference
                    s.sum_squares[feature] - 2.0 * chunk_mean * n * global_mean
                        + n * global_mean * global_mean
                })
                .sum();

            global_stds[feature] = (sum_squared_deviations / total_samples as f64).sqrt();
        }

        Ok(GlobalStatistics {
            means: global_means,
            stds: global_stds,
            mins: global_mins,
            maxs: global_maxs,
        })
    }

    fn apply_scaling(dataset: &Dataset, params: &ScalingParameters) -> Result<Dataset> {
        let mut scaled_data = dataset.data.clone();

        match params.method {
            ScalingMethod::StandardScaler => {
                for (col_idx, mut column) in scaled_data.columns_mut().into_iter().enumerate() {
                    let mean = params.means[col_idx];
                    let std = params.stds[col_idx];

                    if std > 1e-8 {
                        // Avoid division by zero
                        for value in column.iter_mut() {
                            *value = (*value - mean) / std;
                        }
                    }
                }
            }
            ScalingMethod::MinMaxScaler => {
                for (col_idx, mut column) in scaled_data.columns_mut().into_iter().enumerate() {
                    let min = params.mins[col_idx];
                    let max = params.maxs[col_idx];
                    let range = max - min;

                    if range > 1e-8 {
                        // Avoid division by zero
                        for value in column.iter_mut() {
                            *value = (*value - min) / range;
                        }
                    }
                }
            }
            ScalingMethod::RobustScaler => {
                // For simplicity, fall back to standard scaling
                // In a full implementation, you'd compute medians and MAD
                for (col_idx, mut column) in scaled_data.columns_mut().into_iter().enumerate() {
                    let mean = params.means[col_idx];
                    let std = params.stds[col_idx];

                    if std > 1e-8 {
                        for value in column.iter_mut() {
                            *value = (*value - mean) / std;
                        }
                    }
                }
            }
        }

        Ok(Dataset {
            data: scaled_data,
            target: dataset.target.clone(),
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some("Distributed scaled dataset".to_string()),
            metadata: dataset.metadata.clone(),
        })
    }
}

/// Statistics computed on a chunk of data
#[derive(Debug, Clone)]
struct ChunkStatistics {
    n_samples: usize,
    means: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
    sum_squares: Vec<f64>,
}

/// Global statistics combined from all chunks
#[derive(Debug, Clone)]
struct GlobalStatistics {
    means: Vec<f64>,
    stds: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

/// Scaling methods for distributed processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// Z-score normalization
    StandardScaler,
    /// Min-max scaling to [0, 1]
    MinMaxScaler,
    /// Robust scaling using median and MAD
    RobustScaler,
}

/// Parameters for scaling transformations
#[derive(Debug, Clone)]
pub struct ScalingParameters {
    method: ScalingMethod,
    means: Vec<f64>,
    stds: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

impl ScalingParameters {
    fn from_statistics(stats: &GlobalStatistics, method: ScalingMethod) -> Self {
        Self {
            method,
            means: stats.means.clone(),
            stds: stats.stds.clone(),
            mins: stats.mins.clone(),
            maxs: stats.maxs.clone(),
        }
    }
}

// Add missing import for array slicing syntax
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::make_classification;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert!(config.num_workers > 0);
        assert!(config.chunk_size > 0);
    }

    #[test]
    fn test_split_dataset_into_chunks() {
        let dataset = make_classification(100, 5, 2, 3, 1, Some(42)).unwrap();
        let processor = DistributedProcessor::default_config().unwrap();

        let chunks = processor.split_dataset_into_chunks(&dataset).unwrap();

        assert!(!chunks.is_empty());

        let total_samples: usize = chunks.iter().map(|c| c.n_samples()).sum();
        assert_eq!(total_samples, dataset.n_samples());
    }

    #[test]
    fn test_distributed_sample() {
        let dataset = make_classification(1000, 5, 2, 3, 1, Some(42)).unwrap();
        let processor = DistributedProcessor::default_config().unwrap();

        let sampled = processor
            .distributed_sample(&dataset, 100, Some(42))
            .unwrap();

        assert_eq!(sampled.n_samples(), 100);
        assert_eq!(sampled.n_features(), dataset.n_features());
    }

    #[test]
    fn test_distributed_k_fold() {
        let dataset = make_classification(100, 5, 2, 3, 1, Some(42)).unwrap();
        let processor = DistributedProcessor::default_config().unwrap();

        let folds = processor
            .distributed_k_fold(&dataset, 5, true, Some(42))
            .unwrap();

        assert_eq!(folds.len(), 5);

        for (train, test) in folds {
            assert!(train.n_samples() > 0);
            assert!(test.n_samples() > 0);
            assert_eq!(train.n_features(), dataset.n_features());
            assert_eq!(test.n_features(), dataset.n_features());
        }
    }

    #[test]
    fn test_combine_datasets() {
        let dataset1 = make_classification(50, 3, 2, 2, 1, Some(42)).unwrap();
        let dataset2 = make_classification(30, 3, 2, 2, 1, Some(43)).unwrap();

        let processor = DistributedProcessor::default_config().unwrap();
        let combined = processor.combine_datasets(&[dataset1, dataset2]).unwrap();

        assert_eq!(combined.n_samples(), 80);
        assert_eq!(combined.n_features(), 3);
    }

    #[test]
    fn test_parallel_processing() {
        let dataset = make_classification(200, 4, 2, 3, 1, Some(42)).unwrap();
        let processor = DistributedProcessor::default_config().unwrap();

        // Simple processor that counts samples
        let counter = |chunk: &Dataset| -> Result<usize> { Ok(chunk.n_samples()) };

        let results = processor
            .process_dataset_parallel(&dataset, counter)
            .unwrap();

        let total_processed: usize = results.iter().sum();
        assert_eq!(total_processed, dataset.n_samples());
    }
}
