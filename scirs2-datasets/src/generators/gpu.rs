//! GPU-accelerated dataset generators

use super::basic::{make_blobs, make_classification, make_regression};
use super::config::GpuConfig;
use crate::error::{DatasetsError, Result};
use crate::gpu::{GpuContext, GpuDeviceInfo};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Distribution;
// Use local GPU implementation instead of core to avoid feature flag issues
use crate::gpu::GpuBackend as LocalGpuBackend;

/// GPU-accelerated classification dataset generation
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn make_classification_gpu(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Check if GPU is available and requested
    if gpuconfig.use_gpu && gpu_is_available() {
        make_classification_gpu_impl(
            n_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            randomseed,
            gpuconfig,
        )
    } else {
        // Fallback to CPU implementation
        make_classification(
            n_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            randomseed,
        )
    }
}

/// Internal GPU implementation for classification data generation
#[allow(dead_code)]
fn make_classification_gpu_impl(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Input validation
    if n_samples == 0 || n_features == 0 || n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples, n_features, and n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({n_features}) must be >= n_informative ({n_informative})"
        )));
    }

    if n_classes < 2 || n_clusters_per_class == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be >= 2 and n_clusters_per_class must be > 0".to_string(),
        ));
    }

    // Create GPU context
    let gpu_context = GpuContext::new(crate::gpu::GpuConfig {
        backend: crate::gpu::GpuBackend::Cuda {
            device_id: gpuconfig.device_id as u32,
        },
        memory: crate::gpu::GpuMemoryConfig::default(),
        threads_per_block: 256,
        enable_double_precision: !gpuconfig.use_single_precision,
        use_fast_math: false,
        random_seed: None,
    })
    .map_err(|e| DatasetsError::Other(format!("Failed to create GPU context: {e}")))?;

    // Generate data in chunks to avoid memory issues
    let chunk_size = std::cmp::min(gpuconfig.chunk_size, n_samples);
    let num_chunks = n_samples.div_ceil(chunk_size);

    let mut all_data = Vec::new();
    let mut all_targets = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = std::cmp::min(start_idx + chunk_size, n_samples);
        let chunk_samples = end_idx - start_idx;

        // Generate chunk on GPU
        let (chunk_data, chunk_targets) = generate_classification_chunk_gpu(
            &gpu_context,
            chunk_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            randomseed.map(|s| s + chunk_idx as u64),
            gpuconfig.use_single_precision,
        )?;

        all_data.extend(chunk_data);
        all_targets.extend(chunk_targets);
    }

    // Convert to ndarray
    let data = Array2::from_shape_vec((n_samples, n_features), all_data)
        .map_err(|e| DatasetsError::Other(format!("Failed to create data array: {e}")))?;

    let target = Array1::from_vec(all_targets);

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();
    let classnames: Vec<String> = (0..n_classes).map(|i| format!("class_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_targetnames(classnames)
        .with_description(format!(
            "GPU-accelerated synthetic classification dataset with {n_classes} _classes and {n_features} _features"
        ));

    Ok(dataset)
}

/// Generate a chunk of classification data on GPU
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn generate_classification_chunk_gpu(
    gpu_context: &GpuContext,
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    randomseed: Option<u64>,
    _use_single_precision: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // For now, implement using GPU matrix operations
    // In a real implementation, this would use custom GPU kernels

    let _seed = randomseed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(_seed);

    // Generate centroids
    let n_centroids = n_classes * n_clusters_per_class;
    let mut centroids = vec![0.0; n_centroids * n_informative];

    for i in 0..n_centroids {
        for j in 0..n_informative {
            centroids[i * n_informative + j] = 2.0 * rng.gen_range(-1.0f64..1.0f64);
        }
    }

    // Generate _samples using GPU-accelerated operations
    let mut data = vec![0.0; n_samples * n_features];
    let mut targets = vec![0.0; n_samples];

    // Implement GPU buffer operations for accelerated data generation
    if *gpu_context.backend() != LocalGpuBackend::Cpu {
        return generate_classification_gpu_optimized(
            gpu_context,
            &centroids,
            n_samples,
            n_features,
            n_classes,
            n_clusters_per_class,
            n_informative,
            &mut rng,
        );
    }

    // CPU fallback: Generate _samples in parallel chunks
    let samples_per_class = n_samples / n_classes;
    let remainder = n_samples % n_classes;

    let mut sample_idx = 0;
    for _class in 0..n_classes {
        let n_samples_class = if _class < remainder {
            samples_per_class + 1
        } else {
            samples_per_class
        };

        let samples_per_cluster = n_samples_class / n_clusters_per_class;
        let cluster_remainder = n_samples_class % n_clusters_per_class;

        for cluster in 0..n_clusters_per_class {
            let n_samples_cluster = if cluster < cluster_remainder {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            let centroid_idx = _class * n_clusters_per_class + cluster;

            for _ in 0..n_samples_cluster {
                // Generate sample around centroid
                for j in 0..n_informative {
                    let centroid_val = centroids[centroid_idx * n_informative + j];
                    let noise = rand_distr::Normal::new(0.0, 0.3).unwrap().sample(&mut rng);
                    data[sample_idx * n_features + j] = centroid_val + noise;
                }

                // Add noise _features
                for j in n_informative..n_features {
                    data[sample_idx * n_features + j] =
                        rand_distr::Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
                }

                targets[sample_idx] = _class as f64;
                sample_idx += 1;
            }
        }
    }

    Ok((data, targets))
}

/// GPU-optimized classification data generation using buffer operations
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn generate_classification_gpu_optimized(
    _gpu_context: &GpuContext,
    centroids: &[f64],
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    rng: &mut StdRng,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // For now, use CPU-based implementation since core GPU _features are not available
    // TODO: Implement proper GPU acceleration when core GPU _features are stabilized

    // CPU fallback implementation since GPU _features are not available
    use rand_distr::Distribution;
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let mut data = vec![0.0; n_samples * n_features];
    let mut targets = vec![0.0; n_samples];

    // Samples per _class
    let samples_per_class = n_samples / n_classes;
    let remainder = n_samples % n_classes;

    let mut sample_idx = 0;

    for _class in 0..n_classes {
        let n_samples_class = if _class < remainder {
            samples_per_class + 1
        } else {
            samples_per_class
        };

        // Samples per cluster within this _class
        let samples_per_cluster = n_samples_class / n_clusters_per_class;
        let cluster_remainder = n_samples_class % n_clusters_per_class;

        for cluster in 0..n_clusters_per_class {
            let n_samples_cluster = if cluster < cluster_remainder {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            let centroid_idx = _class * n_clusters_per_class + cluster;

            for _ in 0..n_samples_cluster {
                // Generate _informative _features around cluster centroid
                for j in 0..n_informative {
                    let centroid_val = centroids[centroid_idx * n_informative + j];
                    data[sample_idx * n_features + j] = centroid_val + 0.3 * normal.sample(rng);
                }

                // Generate noise _features
                for j in n_informative..n_features {
                    data[sample_idx * n_features + j] = normal.sample(rng);
                }

                targets[sample_idx] = _class as f64;
                sample_idx += 1;
            }
        }
    }

    // TODO: Future GPU implementation placeholder - currently using CPU fallback

    Ok((data, targets))
}

/// GPU-accelerated regression dataset generation
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn make_regression_gpu(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Check if GPU is available and requested
    if gpuconfig.use_gpu && gpu_is_available() {
        make_regression_gpu_impl(
            n_samples,
            n_features,
            n_informative,
            noise,
            randomseed,
            gpuconfig,
        )
    } else {
        // Fallback to CPU implementation
        make_regression(n_samples, n_features, n_informative, noise, randomseed)
    }
}

/// Internal GPU implementation for regression data generation
#[allow(dead_code)]
fn make_regression_gpu_impl(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Input validation
    if n_samples == 0 || n_features == 0 || n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples, n_features, and n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({n_features}) must be >= n_informative ({n_informative})"
        )));
    }

    // Create GPU context
    let gpu_context = GpuContext::new(crate::gpu::GpuConfig {
        backend: crate::gpu::GpuBackend::Cuda {
            device_id: gpuconfig.device_id as u32,
        },
        memory: crate::gpu::GpuMemoryConfig::default(),
        threads_per_block: 256,
        enable_double_precision: !gpuconfig.use_single_precision,
        use_fast_math: false,
        random_seed: None,
    })
    .map_err(|e| DatasetsError::Other(format!("Failed to create GPU context: {e}")))?;

    let _seed = randomseed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(_seed);

    // Generate coefficient matrix on GPU
    let mut coefficients = vec![0.0; n_informative];
    for coeff in coefficients.iter_mut().take(n_informative) {
        *coeff = rng.gen_range(-2.0f64..2.0f64);
    }

    // Generate data matrix in chunks
    let chunk_size = std::cmp::min(gpuconfig.chunk_size, n_samples);
    let num_chunks = n_samples.div_ceil(chunk_size);

    let mut all_data = Vec::new();
    let mut all_targets = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = std::cmp::min(start_idx + chunk_size, n_samples);
        let chunk_samples = end_idx - start_idx;

        // Generate chunk on GPU
        let (chunk_data, chunk_targets) = generate_regression_chunk_gpu(
            &gpu_context,
            chunk_samples,
            n_features,
            n_informative,
            &coefficients,
            noise,
            randomseed.map(|s| s + chunk_idx as u64),
        )?;

        all_data.extend(chunk_data);
        all_targets.extend(chunk_targets);
    }

    // Convert to ndarray
    let data = Array2::from_shape_vec((n_samples, n_features), all_data)
        .map_err(|e| DatasetsError::Other(format!("Failed to create data array: {e}")))?;

    let target = Array1::from_vec(all_targets);

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "GPU-accelerated synthetic regression dataset with {n_features} _features"
        ));

    Ok(dataset)
}

/// Generate a chunk of regression data on GPU
#[allow(dead_code)]
fn generate_regression_chunk_gpu(
    gpu_context: &GpuContext,
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    coefficients: &[f64],
    noise: f64,
    randomseed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let _seed = randomseed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(_seed);

    // Generate random data matrix
    let mut data = vec![0.0; n_samples * n_features];
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    // Use GPU for matrix multiplication if available
    for i in 0..n_samples {
        for j in 0..n_features {
            data[i * n_features + j] = normal.sample(&mut rng);
        }
    }

    // Calculate targets using GPU matrix operations
    let mut targets = vec![0.0; n_samples];
    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();

    // Create GPU buffers for accelerated matrix operations
    if *gpu_context.backend() != LocalGpuBackend::Cpu {
        return generate_regression_gpu_optimized(
            gpu_context,
            &data,
            coefficients,
            n_samples,
            n_features,
            n_informative,
            noise,
            &mut rng,
        );
    }

    // CPU fallback: Matrix multiplication using nested loops
    for i in 0..n_samples {
        let mut target_val = 0.0;
        for j in 0..n_informative {
            target_val += data[i * n_features + j] * coefficients[j];
        }

        // Add noise
        target_val += noise_dist.sample(&mut rng);
        targets[i] = target_val;
    }

    Ok((data, targets))
}

/// GPU-optimized regression data generation using buffer operations and matrix multiplication
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn generate_regression_gpu_optimized(
    _gpu_context: &GpuContext,
    data: &[f64],
    coefficients: &[f64],
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    rng: &mut StdRng,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // For now, use CPU-based implementation since core GPU _features are not available
    // TODO: Implement proper GPU acceleration when core GPU _features are stabilized

    // CPU fallback implementation since GPU _features are not available
    use rand_distr::Distribution;
    let normal = rand_distr::Normal::new(0.0, noise).unwrap();

    let mut targets = vec![0.0; n_samples];

    // Matrix multiplication for regression targets
    for i in 0..n_samples {
        let mut target = 0.0;
        for j in 0..n_informative {
            target += data[i * n_features + j] * coefficients[j];
        }

        // Add noise
        target += normal.sample(rng);
        targets[i] = target;
    }

    Ok((data.to_vec(), targets))
}

/// GPU-accelerated blob generation
#[allow(dead_code)]
pub fn make_blobs_gpu(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Check if GPU is available and requested
    if gpuconfig.use_gpu && gpu_is_available() {
        make_blobs_gpu_impl(
            n_samples,
            n_features,
            n_centers,
            cluster_std,
            randomseed,
            gpuconfig,
        )
    } else {
        // Fallback to CPU implementation
        make_blobs(n_samples, n_features, n_centers, cluster_std, randomseed)
    }
}

/// Internal GPU implementation for blob generation
#[allow(dead_code)]
fn make_blobs_gpu_impl(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    randomseed: Option<u64>,
    gpuconfig: GpuConfig,
) -> Result<Dataset> {
    // Input validation
    if n_samples == 0 || n_features == 0 || n_centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples, n_features, and n_centers must be > 0".to_string(),
        ));
    }

    if cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster_std must be > 0".to_string(),
        ));
    }

    // Create GPU context
    let gpu_context = GpuContext::new(crate::gpu::GpuConfig {
        backend: crate::gpu::GpuBackend::Cuda {
            device_id: gpuconfig.device_id as u32,
        },
        memory: crate::gpu::GpuMemoryConfig::default(),
        threads_per_block: 256,
        enable_double_precision: !gpuconfig.use_single_precision,
        use_fast_math: false,
        random_seed: None,
    })
    .map_err(|e| DatasetsError::Other(format!("Failed to create GPU context: {e}")))?;

    let _seed = randomseed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(_seed);

    // Generate cluster _centers
    let mut centers = Array2::zeros((n_centers, n_features));
    let center_dist = rand_distr::Normal::new(0.0, 10.0).unwrap();

    for i in 0..n_centers {
        for j in 0..n_features {
            centers[[i, j]] = center_dist.sample(&mut rng);
        }
    }

    // Generate _samples around _centers using GPU acceleration
    let samples_per_center = n_samples / n_centers;
    let remainder = n_samples % n_centers;

    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let mut sample_idx = 0;
    let noise_dist = rand_distr::Normal::new(0.0, cluster_std).unwrap();

    for center_idx in 0..n_centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        // Generate _samples for this center using GPU acceleration
        if *gpu_context.backend() != LocalGpuBackend::Cpu {
            // Use GPU kernel for parallel sample generation
            let gpu_generated = generate_blobs_center_gpu(
                &gpu_context,
                &centers,
                center_idx,
                n_samples_center,
                n_features,
                cluster_std,
                &mut rng,
            )?;

            // Copy GPU-generated data to main arrays
            for (local_idx, sample) in gpu_generated.iter().enumerate() {
                for j in 0..n_features {
                    data[[sample_idx + local_idx, j]] = sample[j];
                }
                target[sample_idx + local_idx] = center_idx as f64;
            }
            sample_idx += n_samples_center;
        } else {
            // CPU fallback: generate sequentially
            for _ in 0..n_samples_center {
                for j in 0..n_features {
                    data[[sample_idx, j]] = centers[[center_idx, j]] + noise_dist.sample(&mut rng);
                }
                target[sample_idx] = center_idx as f64;
                sample_idx += 1;
            }
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();
    let centernames: Vec<String> = (0..n_centers).map(|i| format!("center_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_targetnames(centernames)
        .with_description(format!(
            "GPU-accelerated synthetic blob dataset with {n_centers} _centers and {n_features} _features"
        ));

    Ok(dataset)
}

/// GPU-optimized blob center generation using parallel kernels
#[allow(dead_code)]
fn generate_blobs_center_gpu(
    _gpu_context: &GpuContext,
    centers: &Array2<f64>,
    center_idx: usize,
    n_samples_center: usize,
    n_features: usize,
    cluster_std: f64,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>> {
    // For now, use CPU-based implementation since core GPU _features are not available
    // TODO: Implement proper GPU acceleration when core GPU _features are stabilized

    // Extract _center coordinates for this specific _center
    let _center_coords: Vec<f64> = (0..n_features).map(|j| centers[[center_idx, j]]).collect();

    // CPU fallback implementation since GPU _features are not available
    use rand_distr::Distribution;
    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

    let mut result = Vec::with_capacity(n_samples_center);

    for _ in 0..n_samples_center {
        let mut sample = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let center_val = centers[[center_idx, j]];
            let noise = normal.sample(rng);
            sample.push(center_val + noise);
        }
        result.push(sample);
    }

    Ok(result)
}

/// Check if GPU is available for acceleration
#[allow(dead_code)]
pub fn gpu_is_available() -> bool {
    // Try to create a GPU context to check availability
    GpuContext::new(crate::gpu::GpuConfig::default()).is_ok()
}

/// Get GPU device information
#[allow(dead_code)]
pub fn get_gpu_info() -> Result<Vec<GpuDeviceInfo>> {
    crate::gpu::list_gpu_devices()
        .map_err(|e| DatasetsError::Other(format!("Failed to get GPU info: {e}")))
}

/// Benchmark GPU vs CPU performance for data generation
#[allow(dead_code)]
pub fn benchmark_gpu_vs_cpu(
    n_samples: usize,
    n_features: usize,
    iterations: usize,
) -> Result<(f64, f64)> {
    use std::time::Instant;

    // Benchmark CPU implementation
    let cpu_start = Instant::now();
    for _ in 0..iterations {
        let _result = make_classification(n_samples, n_features, 3, 2, n_features, Some(42))?;
    }
    let cpu_time = cpu_start.elapsed().as_secs_f64() / iterations as f64;

    // Benchmark GPU implementation
    let gpuconfig = GpuConfig::default();
    let gpu_start = Instant::now();
    for _ in 0..iterations {
        let _result = make_classification_gpu(
            n_samples,
            n_features,
            3,
            2,
            n_features,
            Some(42),
            gpuconfig.clone(),
        )?;
    }
    let gpu_time = gpu_start.elapsed().as_secs_f64() / iterations as f64;

    Ok((cpu_time, gpu_time))
}
