//! GPU-accelerated dataset generators

use super::basic::{make_blobs, make_classification, make_regression};
use super::config::GpuConfig;
use super::gpu_dispatch;
use crate::error::{DatasetsError, Result};
use crate::gpu::{GpuContext, GpuDeviceInfo};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Distribution;
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
            centroids[i * n_informative + j] = 2.0 * rng.random_range(-1.0f64..1.0f64);
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
                    let noise = scirs2_core::random::Normal::new(0.0, 0.3)
                        .expect("Operation failed")
                        .sample(&mut rng);
                    data[sample_idx * n_features + j] = centroid_val + noise;
                }

                // Add noise _features
                for j in n_informative..n_features {
                    data[sample_idx * n_features + j] = scirs2_core::random::Normal::new(0.0, 1.0)
                        .expect("Operation failed")
                        .sample(&mut rng);
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
    use scirs2_core::random::Distribution;
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("Operation failed");

    let mut data = vec![0.0; n_samples * n_features];
    let mut targets = vec![0.0; n_samples];

    // Per-sample broadcast centroid coordinates for the informative columns,
    // laid out contiguously as [n_samples, n_informative]. Built alongside the
    // raw noise so the GPU can compute `informative = centroids + 0.3 * noise`.
    let mut centroid_broadcast = vec![0.0; n_samples * n_informative];

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
                // Draw raw informative noise into `data` (RNG order preserved) and
                // record the matching centroid coordinate for the GPU offset step.
                for j in 0..n_informative {
                    data[sample_idx * n_features + j] = normal.sample(rng);
                    centroid_broadcast[sample_idx * n_informative + j] =
                        centroids[centroid_idx * n_informative + j];
                }

                // Generate noise _features (no GPU benefit; raw draws are final).
                for j in n_informative..n_features {
                    data[sample_idx * n_features + j] = normal.sample(rng);
                }

                targets[sample_idx] = _class as f64;
                sample_idx += 1;
            }
        }
    }

    // Heavy step: informative_features = centroid_broadcast + 0.3 * noise.
    // Dispatch the large elementwise transform to the GPU (threshold-gated,
    // f32), falling back to a CPU pass on any GPU error or below the threshold.
    let informative_flat: Vec<f64> = (0..n_samples)
        .flat_map(|i| (0..n_informative).map(move |j| (i, j)))
        .map(|(i, j)| data[i * n_features + j])
        .collect();

    let transformed = match gpu_dispatch::try_classification_informative_gpu(
        &centroid_broadcast,
        &informative_flat,
        n_samples,
        n_informative,
    ) {
        gpu_dispatch::GpuDispatch::Done(gpu_informative) => gpu_informative,
        gpu_dispatch::GpuDispatch::FallbackToCpu => centroid_broadcast
            .iter()
            .zip(informative_flat.iter())
            .map(|(&c, &noise)| c + 0.3 * noise)
            .collect(),
    };

    // Scatter the informative results back into the strided data layout.
    for i in 0..n_samples {
        for j in 0..n_informative {
            data[i * n_features + j] = transformed[i * n_informative + j];
        }
    }

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
        *coeff = rng.random_range(-2.0f64..2.0f64);
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
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("Operation failed");

    // Use GPU for matrix multiplication if available
    for i in 0..n_samples {
        for j in 0..n_features {
            data[i * n_features + j] = normal.sample(&mut rng);
        }
    }

    // Calculate targets using GPU matrix operations
    let mut targets = vec![0.0; n_samples];
    let noise_dist = scirs2_core::random::Normal::new(0.0, noise).expect("Operation failed");

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
    use scirs2_core::random::Distribution;

    // Heavy step: y_noise_free = X[:, :n_informative] · coefficients.
    // Dispatch the matmul to the GPU (threshold-gated, f32), falling back to the
    // CPU nested-loop reduction on any GPU error or below the threshold.
    let mut targets = match gpu_dispatch::try_regression_targets_gpu(
        data,
        coefficients,
        n_samples,
        n_features,
        n_informative,
    ) {
        gpu_dispatch::GpuDispatch::Done(gpu_targets) => gpu_targets,
        gpu_dispatch::GpuDispatch::FallbackToCpu => {
            // CPU fallback: nested-loop matrix multiplication.
            let mut t = vec![0.0; n_samples];
            for (i, slot) in t.iter_mut().enumerate() {
                let mut acc = 0.0;
                for j in 0..n_informative {
                    acc += data[i * n_features + j] * coefficients[j];
                }
                *slot = acc;
            }
            t
        }
    };

    // Add per-sample Gaussian noise on the CPU so the RNG sequence is identical
    // regardless of whether the matmul ran on GPU or CPU.
    let normal = scirs2_core::random::Normal::new(0.0, noise).expect("Operation failed");
    for target in targets.iter_mut() {
        *target += normal.sample(rng);
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
    let center_dist = scirs2_core::random::Normal::new(0.0, 10.0).expect("Operation failed");

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
    let noise_dist = scirs2_core::random::Normal::new(0.0, cluster_std).expect("Operation failed");

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
    use scirs2_core::random::Distribution;
    let normal = scirs2_core::random::Normal::new(0.0, cluster_std).expect("Operation failed");

    // Draw all noise in canonical (sample, feature) order into a flat buffer and
    // build the matching broadcast of the center coordinates. The GPU then
    // computes `sample = center_broadcast + noise` elementwise.
    let total = n_samples_center * n_features;
    let mut noise_flat = vec![0.0; total];
    let mut center_broadcast = vec![0.0; total];
    for s in 0..n_samples_center {
        for j in 0..n_features {
            noise_flat[s * n_features + j] = normal.sample(rng);
            center_broadcast[s * n_features + j] = centers[[center_idx, j]];
        }
    }

    // Heavy step: dispatch the broadcast-add to the GPU (threshold-gated, f32),
    // falling back to a CPU pass on any GPU error or below the threshold.
    let combined = match gpu_dispatch::try_blobs_center_gpu(
        &center_broadcast,
        &noise_flat,
        n_samples_center,
        n_features,
    ) {
        gpu_dispatch::GpuDispatch::Done(gpu_samples) => gpu_samples,
        gpu_dispatch::GpuDispatch::FallbackToCpu => center_broadcast
            .iter()
            .zip(noise_flat.iter())
            .map(|(&c, &noise)| c + noise)
            .collect(),
    };

    // Reshape the flat [n_samples_center, n_features] result into rows.
    let mut result = Vec::with_capacity(n_samples_center);
    for s in 0..n_samples_center {
        let start = s * n_features;
        result.push(combined[start..start + n_features].to_vec());
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

/// Smoke tests for the real `GpuNdarray` dispatch paths.
///
/// Each test forces a problem above [`gpu_dispatch::GPU_DATASET_THRESHOLD`] and
/// then either:
/// - confirms the GPU result agrees with the CPU reference within `f32`
///   tolerance (when a wgpu adapter is present), or
/// - skips gracefully when the dispatch falls back to CPU (no adapter).
///
/// They only run with the `gpu_wgpu` feature enabled so the default build is
/// unaffected.
#[cfg(all(test, feature = "gpu_wgpu"))]
mod gpu_smoke_tests {
    use super::gpu_dispatch::{
        try_blobs_center_gpu, try_classification_informative_gpu, try_regression_targets_gpu,
        GpuDispatch, GPU_DATASET_THRESHOLD,
    };

    const TOL: f64 = 1e-3;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn regression_targets_gpu_matches_cpu_or_skips() {
        let n_samples = 512usize;
        let n_features = 16usize;
        let n_informative = 8usize;
        assert!(n_samples * n_informative >= GPU_DATASET_THRESHOLD);

        // Deterministic design matrix and coefficients (no RNG needed).
        let data: Vec<f64> = (0..n_samples * n_features)
            .map(|k| ((k % 7) as f64) * 0.1 - 0.3)
            .collect();
        let coef: Vec<f64> = (0..n_informative).map(|j| 0.5 + 0.25 * j as f64).collect();

        // CPU reference: y = X[:, :k] · coef.
        let cpu: Vec<f64> = (0..n_samples)
            .map(|i| {
                (0..n_informative)
                    .map(|j| data[i * n_features + j] * coef[j])
                    .sum()
            })
            .collect();

        match try_regression_targets_gpu(&data, &coef, n_samples, n_features, n_informative) {
            GpuDispatch::Done(gpu) => {
                assert_eq!(gpu.len(), cpu.len());
                let diff = max_abs_diff(&gpu, &cpu);
                assert!(
                    diff < TOL,
                    "GPU regression matmul diverged from CPU by {diff:.2e} (> {TOL:.0e})"
                );
                println!("regression GPU matmul matched CPU (max diff {diff:.2e})");
            }
            GpuDispatch::FallbackToCpu => {
                println!("no wgpu adapter — regression GPU test skipped (fell back to CPU)");
            }
        }
    }

    #[test]
    fn classification_informative_gpu_matches_cpu_or_skips() {
        let n_samples = 1024usize;
        let n_informative = 8usize;
        assert!(n_samples * n_informative >= GPU_DATASET_THRESHOLD);

        let centroids: Vec<f64> = (0..n_samples * n_informative)
            .map(|k| ((k % 5) as f64) - 2.0)
            .collect();
        let noise: Vec<f64> = (0..n_samples * n_informative)
            .map(|k| 0.01 * ((k % 11) as f64) - 0.05)
            .collect();

        // CPU reference: informative = centroids + 0.3 * noise.
        let cpu: Vec<f64> = centroids
            .iter()
            .zip(noise.iter())
            .map(|(&c, &n)| c + 0.3 * n)
            .collect();

        match try_classification_informative_gpu(&centroids, &noise, n_samples, n_informative) {
            GpuDispatch::Done(gpu) => {
                assert_eq!(gpu.len(), cpu.len());
                let diff = max_abs_diff(&gpu, &cpu);
                assert!(
                    diff < TOL,
                    "GPU classification offset diverged from CPU by {diff:.2e} (> {TOL:.0e})"
                );
                println!("classification GPU offset matched CPU (max diff {diff:.2e})");
            }
            GpuDispatch::FallbackToCpu => {
                println!("no wgpu adapter — classification GPU test skipped (fell back to CPU)");
            }
        }
    }

    #[test]
    fn blobs_center_gpu_matches_cpu_or_skips() {
        let n_samples_center = 1024usize;
        let n_features = 8usize;
        assert!(n_samples_center * n_features >= GPU_DATASET_THRESHOLD);

        let center_broadcast: Vec<f64> = (0..n_samples_center * n_features)
            .map(|k| ((k % n_features) as f64) * 1.5 - 3.0)
            .collect();
        let noise: Vec<f64> = (0..n_samples_center * n_features)
            .map(|k| 0.02 * ((k % 13) as f64) - 0.1)
            .collect();

        // CPU reference: sample = center + noise.
        let cpu: Vec<f64> = center_broadcast
            .iter()
            .zip(noise.iter())
            .map(|(&c, &n)| c + n)
            .collect();

        match try_blobs_center_gpu(&center_broadcast, &noise, n_samples_center, n_features) {
            GpuDispatch::Done(gpu) => {
                assert_eq!(gpu.len(), cpu.len());
                let diff = max_abs_diff(&gpu, &cpu);
                assert!(
                    diff < TOL,
                    "GPU blobs broadcast-add diverged from CPU by {diff:.2e} (> {TOL:.0e})"
                );
                println!("blobs GPU broadcast-add matched CPU (max diff {diff:.2e})");
            }
            GpuDispatch::FallbackToCpu => {
                println!("no wgpu adapter — blobs GPU test skipped (fell back to CPU)");
            }
        }
    }

    #[test]
    fn below_threshold_falls_back_to_cpu() {
        // A tiny problem must never dispatch to the GPU.
        let n_samples = 4usize;
        let n_features = 4usize;
        let n_informative = 2usize;
        let data = vec![1.0f64; n_samples * n_features];
        let coef = vec![1.0f64; n_informative];
        assert!(matches!(
            try_regression_targets_gpu(&data, &coef, n_samples, n_features, n_informative),
            GpuDispatch::FallbackToCpu
        ));
    }
}
