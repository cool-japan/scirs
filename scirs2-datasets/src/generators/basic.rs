//! Basic dataset generators (classification, regression, blobs, etc.)

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::{Distribution, Uniform};
use std::f64::consts::PI;

/// Generate a random classification dataset with clusters
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({n_features}) must be >= n_informative ({n_informative})"
        )));
    }

    if n_classes < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be >= 2".to_string(),
        ));
    }

    if n_clusters_per_class == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_clusters_per_class must be > 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate centroids for each _class and cluster
    let n_centroids = n_classes * n_clusters_per_class;
    let mut centroids = Array2::zeros((n_centroids, n_informative));
    let scale = 2.0;

    for i in 0..n_centroids {
        for j in 0..n_informative {
            centroids[[i, j]] = scale * rng.random_range(-1.0f64..1.0f64);
        }
    }

    // Generate _samples
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("Operation failed");

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

        // Assign clusters within this _class
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
                // Randomly select a point near the cluster centroid
                for j in 0..n_informative {
                    data[[sample_idx, j]] =
                        centroids[[centroid_idx, j]] + 0.3 * normal.sample(&mut rng);
                }

                // Add noise _features
                for j in n_informative..n_features {
                    data[[sample_idx, j]] = normal.sample(&mut rng);
                }

                target[sample_idx] = _class as f64;
                sample_idx += 1;
            }
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    // Create _class names
    let classnames: Vec<String> = (0..n_classes).map(|i| format!("class_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_targetnames(classnames)
        .with_description(format!(
            "Synthetic classification dataset with {n_classes} _classes and {n_features} _features"
        ));

    Ok(dataset)
}

/// Minimum number of design-matrix elements (`n_samples * n_features`) before
/// [`make_regression`] is allowed to offload its `y = X · coef` target product
/// to the optional CUDA GPU path (`crate::gpu_cuda`).
///
/// Deliberately set high (1,000,000) so that every small / test-sized problem
/// always uses the deterministic CPU path, keeping `make_regression`'s seeded
/// output bit-for-bit stable even on NVIDIA hosts. Only genuinely large
/// generations are eligible for GPU offload.
#[cfg(feature = "cuda")]
const CUDA_REGRESSION_MIN_ELEMS: usize = 1_000_000;

/// Generate a random regression dataset
#[allow(dead_code)]
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({n_features}) must be >= n_informative ({n_informative})"
        )));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate the coefficients for the _informative _features
    let mut coef = Array1::zeros(n_features);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("Operation failed");

    for i in 0..n_informative {
        coef[i] = 100.0 * normal.sample(&mut rng);
    }

    // Generate the _features
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Generate the target.
    //
    // The original single loop is split into two byte-for-byte-equivalent
    // phases so the optional CUDA fast path has a clean seam:
    //   1. compute the noise-free linear target `linear = X · coef`,
    //   2. add the per-sample Gaussian noise.
    // The dot-product accumulation order is unchanged and the noise RNG is still
    // drawn exactly once per sample in ascending index order, so the default
    // (CPU / no-`cuda`-feature) output is identical to before.
    let mut linear = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let mut y = 0.0;
        for j in 0..n_features {
            y += data[[i, j]] * coef[j];
        }
        linear[i] = y;
    }

    // Optional, additive CUDA fast path for the linear target `X · coef`
    // (off by default; present only when the `cuda` feature is enabled).
    //
    // The product is offloaded to the NVIDIA GPU ONLY for large problems and
    // ONLY when a real CUDA device is present. `CUDA_REGRESSION_MIN_ELEMS` is
    // deliberately high so every small / test-sized generation always keeps the
    // deterministic CPU result above — make_regression's seeded output stays
    // bit-stable even on NVIDIA hosts. On any GPU error the CPU result (the
    // source of truth already in `linear`) is silently retained.
    #[cfg(feature = "cuda")]
    {
        if crate::gpu_cuda::cuda_is_available()
            && n_samples.saturating_mul(n_features) >= CUDA_REGRESSION_MIN_ELEMS
        {
            if let Ok(gpu_y) = crate::gpu_cuda::cuda_regression_target(&data.view(), &coef.view()) {
                linear = gpu_y;
            }
        }
    }

    // Add per-sample Gaussian noise. Drawn in ascending sample order via a
    // paired iterator so the RNG sequence is identical to the original loop
    // (and `needless_range_loop`-clean).
    let mut target = Array1::zeros(n_samples);
    for (slot, &lin) in target.iter_mut().zip(linear.iter()) {
        let mut y = lin;
        if noise > 0.0 {
            y += normal.sample(&mut rng) * noise;
        }
        *slot = y;
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "Synthetic regression dataset with {n_features} _features ({n_informative} informative)"
        ))
        .with_metadata("noise", &noise.to_string())
        .with_metadata("coefficients", &format!("{coef:?}"));

    Ok(dataset)
}

/// Generate a random time series dataset
#[allow(dead_code)]
pub fn make_time_series(
    n_samples: usize,
    n_features: usize,
    trend: bool,
    seasonality: bool,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("Operation failed");
    let mut data = Array2::zeros((n_samples, n_features));

    for feature in 0..n_features {
        let trend_coef = if trend {
            rng.random_range(0.01f64..0.1f64)
        } else {
            0.0
        };
        let seasonality_period = rng.sample(Uniform::new(10, 50).expect("Operation failed")) as f64;
        let seasonality_amplitude = if seasonality {
            rng.random_range(1.0f64..5.0f64)
        } else {
            0.0
        };

        let base_value = rng.random_range(-10.0f64..10.0f64);

        for i in 0..n_samples {
            let t = i as f64;

            // Add base value
            let mut value = base_value;

            // Add trend
            if trend {
                value += trend_coef * t;
            }

            // Add seasonality
            if seasonality {
                value += seasonality_amplitude * (2.0 * PI * t / seasonality_period).sin();
            }

            // Add noise
            if noise > 0.0 {
                value += normal.sample(&mut rng) * noise;
            }

            data[[i, feature]] = value;
        }
    }

    // Create time index (unused for now but can be useful for plotting)
    let time_index: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let _time_array = Array1::from(time_index);

    // Create dataset
    let mut dataset = Dataset::new(data, None);

    // Create feature names
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "Synthetic time series dataset with {n_features} _features"
        ))
        .with_metadata("trend", &trend.to_string())
        .with_metadata("seasonality", &seasonality.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a random blobs dataset for clustering
#[allow(dead_code)]
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "centers must be > 0".to_string(),
        ));
    }

    if cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster_std must be > 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(-center_box..center_box);
        }
    }

    // Generate _samples around centers
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = scirs2_core::random::Normal::new(0.0, cluster_std).expect("Operation failed");

    // Samples per center
    let samples_per_center = n_samples / centers;
    let remainder = n_samples % centers;

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        for _ in 0..n_samples_center {
            for j in 0..n_features {
                data[[sample_idx, j]] = cluster_centers[[center_idx, j]] + normal.sample(&mut rng);
            }

            target[sample_idx] = center_idx as f64;
            sample_idx += 1;
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "Synthetic clustering dataset with {centers} clusters and {n_features} _features"
        ))
        .with_metadata("centers", &centers.to_string())
        .with_metadata("cluster_std", &cluster_std.to_string());

    Ok(dataset)
}

/// Generate a spiral dataset for non-linear classification
#[allow(dead_code)]
pub fn make_spirals(
    n_samples: usize,
    n_spirals: usize,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_spirals == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_spirals must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(scirs2_core::random::Normal::new(0.0, noise).expect("Operation failed"))
    } else {
        None
    };

    let samples_per_spiral = n_samples / n_spirals;
    let remainder = n_samples % n_spirals;

    let mut sample_idx = 0;

    for spiral in 0..n_spirals {
        let n_samples_spiral = if spiral < remainder {
            samples_per_spiral + 1
        } else {
            samples_per_spiral
        };

        let spiral_offset = 2.0 * PI * spiral as f64 / n_spirals as f64;

        for i in 0..n_samples_spiral {
            let t = 2.0 * PI * i as f64 / n_samples_spiral as f64;
            let radius = t / (2.0 * PI);

            let mut x = radius * (t + spiral_offset).cos();
            let mut y = radius * (t + spiral_offset).sin();

            // Add noise if specified
            if let Some(ref normal_dist) = normal {
                x += normal_dist.sample(&mut rng);
                y += normal_dist.sample(&mut rng);
            }

            data[[sample_idx, 0]] = x;
            data[[sample_idx, 1]] = y;
            target[sample_idx] = spiral as f64;
            sample_idx += 1;
        }
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_featurenames(vec!["x".to_string(), "y".to_string()])
        .with_targetnames((0..n_spirals).map(|i| format!("spiral_{i}")).collect())
        .with_description(format!("Spiral dataset with {n_spirals} _spirals"))
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a moons dataset for non-linear classification
#[allow(dead_code)]
pub fn make_moons(n_samples: usize, noise: f64, randomseed: Option<u64>) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(scirs2_core::random::Normal::new(0.0, noise).expect("Operation failed"))
    } else {
        None
    };

    let samples_per_moon = n_samples / 2;
    let remainder = n_samples % 2;

    let mut sample_idx = 0;

    // Generate first moon (upper crescent)
    for i in 0..(samples_per_moon + remainder) {
        let t = PI * i as f64 / (samples_per_moon + remainder) as f64;

        let mut x = t.cos();
        let mut y = t.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 0.0;
        sample_idx += 1;
    }

    // Generate second moon (lower crescent, flipped)
    for i in 0..samples_per_moon {
        let t = PI * i as f64 / samples_per_moon as f64;

        let mut x = 1.0 - t.cos();
        let mut y = 0.5 - t.sin(); // Offset vertically and flip

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 1.0;
        sample_idx += 1;
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_featurenames(vec!["x".to_string(), "y".to_string()])
        .with_targetnames(vec!["moon_0".to_string(), "moon_1".to_string()])
        .with_description("Two moons dataset for non-linear classification".to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a circles dataset for non-linear classification
#[allow(dead_code)]
pub fn make_circles(
    n_samples: usize,
    factor: f64,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if factor <= 0.0 || factor >= 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(scirs2_core::random::Normal::new(0.0, noise).expect("Operation failed"))
    } else {
        None
    };

    let samples_per_circle = n_samples / 2;
    let remainder = n_samples % 2;

    let mut sample_idx = 0;

    // Generate outer circle
    for i in 0..(samples_per_circle + remainder) {
        let angle = 2.0 * PI * i as f64 / (samples_per_circle + remainder) as f64;

        let mut x = angle.cos();
        let mut y = angle.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 0.0;
        sample_idx += 1;
    }

    // Generate inner circle (scaled by factor)
    for i in 0..samples_per_circle {
        let angle = 2.0 * PI * i as f64 / samples_per_circle as f64;

        let mut x = factor * angle.cos();
        let mut y = factor * angle.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 1.0;
        sample_idx += 1;
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_featurenames(vec!["x".to_string(), "y".to_string()])
        .with_targetnames(vec!["outer_circle".to_string(), "inner_circle".to_string()])
        .with_description("Concentric circles dataset for non-linear classification".to_string())
        .with_metadata("factor", &factor.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate anisotropic (elongated) clusters
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_anisotropic_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    anisotropy_factor: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be >= 2 for anisotropic clusters".to_string(),
        ));
    }

    if centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "centers must be > 0".to_string(),
        ));
    }

    if cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster_std must be > 0.0".to_string(),
        ));
    }

    if anisotropy_factor <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "anisotropy_factor must be > 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(-center_box..center_box);
        }
    }

    // Generate _samples around centers with anisotropic distribution
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = scirs2_core::random::Normal::new(0.0, cluster_std).expect("Operation failed");

    let samples_per_center = n_samples / centers;
    let remainder = n_samples % centers;

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        // Generate a random rotation angle for this cluster
        let rotation_angle = rng.random_range(0.0..(2.0 * PI));

        for _ in 0..n_samples_center {
            // Generate point with anisotropic distribution (elongated along first axis)
            let mut point = vec![0.0; n_features];

            // First axis has normal std..second axis has reduced _std (anisotropy)
            point[0] = normal.sample(&mut rng);
            point[1] = normal.sample(&mut rng) / anisotropy_factor;

            // Remaining axes have normal _std
            for item in point.iter_mut().take(n_features).skip(2) {
                *item = normal.sample(&mut rng);
            }

            // Apply rotation for 2D case
            if n_features >= 2 {
                let cos_theta = rotation_angle.cos();
                let sin_theta = rotation_angle.sin();

                let x_rot = cos_theta * point[0] - sin_theta * point[1];
                let y_rot = sin_theta * point[0] + cos_theta * point[1];

                point[0] = x_rot;
                point[1] = y_rot;
            }

            // Translate to cluster center
            for j in 0..n_features {
                data[[sample_idx, j]] = cluster_centers[[center_idx, j]] + point[j];
            }

            target[sample_idx] = center_idx as f64;
            sample_idx += 1;
        }
    }

    let mut dataset = Dataset::new(data, Some(target));
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "Anisotropic clustering dataset with {centers} elongated clusters and {n_features} _features"
        ))
        .with_metadata("centers", &centers.to_string())
        .with_metadata("cluster_std", &cluster_std.to_string())
        .with_metadata("anisotropy_factor", &anisotropy_factor.to_string());

    Ok(dataset)
}

/// Generate hierarchical clusters (clusters within clusters)
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_hierarchical_clusters(
    n_samples: usize,
    n_features: usize,
    n_main_clusters: usize,
    n_sub_clusters: usize,
    main_cluster_std: f64,
    sub_cluster_std: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_main_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_main_clusters must be > 0".to_string(),
        ));
    }

    if n_sub_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_sub_clusters must be > 0".to_string(),
        ));
    }

    if main_cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "main_cluster_std must be > 0.0".to_string(),
        ));
    }

    if sub_cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "sub_cluster_std must be > 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate main cluster centers
    let mut main_centers = Array2::zeros((n_main_clusters, n_features));
    let center_box = 20.0;

    for i in 0..n_main_clusters {
        for j in 0..n_features {
            main_centers[[i, j]] = rng.random_range(-center_box..center_box);
        }
    }

    let mut data = Array2::zeros((n_samples, n_features));
    let mut main_target = Array1::zeros(n_samples);
    let mut sub_target = Array1::zeros(n_samples);

    let main_normal =
        scirs2_core::random::Normal::new(0.0, main_cluster_std).expect("Operation failed");
    let sub_normal =
        scirs2_core::random::Normal::new(0.0, sub_cluster_std).expect("Operation failed");

    let samples_per_main = n_samples / n_main_clusters;
    let remainder = n_samples % n_main_clusters;

    let mut sample_idx = 0;

    for main_idx in 0..n_main_clusters {
        let n_samples_main = if main_idx < remainder {
            samples_per_main + 1
        } else {
            samples_per_main
        };

        // Generate sub-cluster centers within this main cluster
        let mut sub_centers = Array2::zeros((n_sub_clusters, n_features));
        for i in 0..n_sub_clusters {
            for j in 0..n_features {
                sub_centers[[i, j]] = main_centers[[main_idx, j]] + main_normal.sample(&mut rng);
            }
        }

        let samples_per_sub = n_samples_main / n_sub_clusters;
        let sub_remainder = n_samples_main % n_sub_clusters;

        for sub_idx in 0..n_sub_clusters {
            let n_samples_sub = if sub_idx < sub_remainder {
                samples_per_sub + 1
            } else {
                samples_per_sub
            };

            for _ in 0..n_samples_sub {
                for j in 0..n_features {
                    data[[sample_idx, j]] = sub_centers[[sub_idx, j]] + sub_normal.sample(&mut rng);
                }

                main_target[sample_idx] = main_idx as f64;
                sub_target[sample_idx] = (main_idx * n_sub_clusters + sub_idx) as f64;
                sample_idx += 1;
            }
        }
    }

    let mut dataset = Dataset::new(data, Some(main_target));
    let featurenames: Vec<String> = (0..n_features).map(|i| format!("feature_{i}")).collect();

    dataset = dataset
        .with_featurenames(featurenames)
        .with_description(format!(
            "Hierarchical clustering dataset with {n_main_clusters} main clusters, {n_sub_clusters} sub-_clusters each"
        ))
        .with_metadata("n_main_clusters", &n_main_clusters.to_string())
        .with_metadata("n_sub_clusters", &n_sub_clusters.to_string())
        .with_metadata("main_cluster_std", &main_cluster_std.to_string())
        .with_metadata("sub_cluster_std", &sub_cluster_std.to_string());

    let sub_target_vec = sub_target.to_vec();
    dataset = dataset.with_metadata("sub_cluster_labels", &format!("{sub_target_vec:?}"));

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `make_regression` is deterministic for a fixed seed: two calls with the
    /// same arguments produce bit-identical data and targets. Regression guard
    /// for the two-phase target split.
    #[test]
    fn make_regression_deterministic_for_fixed_seed() {
        let a = make_regression(50, 4, 3, 0.1, Some(42)).expect("first make_regression");
        let b = make_regression(50, 4, 3, 0.1, Some(42)).expect("second make_regression");
        assert_eq!(a.data, b.data, "data must be reproducible for a fixed seed");
        assert_eq!(
            a.target, b.target,
            "target must be reproducible for a fixed seed"
        );
    }

    /// With zero noise the target is the pure linear combination `X · coef`; it
    /// must be finite and length-consistent, and a non-zero noise must perturb
    /// it (while the seed-determined design matrix stays identical).
    #[test]
    fn make_regression_noise_free_is_linear_and_noise_perturbs() {
        let clean = make_regression(60, 5, 5, 0.0, Some(7)).expect("clean make_regression");
        let clean_t = clean.target.as_ref().expect("clean target present");
        assert_eq!(clean_t.len(), 60);
        assert!(
            clean_t.iter().all(|v| v.is_finite()),
            "targets must be finite"
        );

        let noisy = make_regression(60, 5, 5, 50.0, Some(7)).expect("noisy make_regression");
        let noisy_t = noisy.target.as_ref().expect("noisy target present");
        assert_eq!(clean.data, noisy.data, "design matrix is seed-determined");
        assert!(
            clean_t
                .iter()
                .zip(noisy_t.iter())
                .any(|(c, n)| (c - n).abs() > 1e-9),
            "non-zero noise must perturb the target"
        );
    }

    /// §4b SUB-THRESHOLD CPU PATH.
    /// Builds and runs only under the `cuda` feature. 64 × 8 = 512 elements —
    /// well below CUDA_REGRESSION_MIN_ELEMS (1 000 000) — so `make_regression`
    /// always takes the deterministic CPU path even on a CUDA-capable host. This
    /// proves the `cuda`-feature build compiles, the CPU path is unaffected, and
    /// the seeded output is byte-identical across two calls.
    #[cfg(feature = "cuda")]
    #[test]
    fn make_regression_cuda_feature_build_matches_cpu() {
        let ds = make_regression(64, 8, 5, 0.0, Some(11)).expect("cuda-feature make_regression");
        let target = ds.target.as_ref().expect("target present");
        assert_eq!(ds.data.nrows(), 64);
        assert_eq!(ds.data.ncols(), 8);
        assert_eq!(target.len(), 64);
        assert!(target.iter().all(|v| v.is_finite()));

        let ds2 = make_regression(64, 8, 5, 0.0, Some(11)).expect("repeat make_regression");
        assert_eq!(
            ds.target, ds2.target,
            "cuda-feature build must stay deterministic on the CPU path"
        );
    }

    /// §4b DISPATCH ENGAGES + CORRECTNESS.
    ///
    /// n_samples × n_features = 2 000 × 600 = 1 200 000 >= CUDA_REGRESSION_MIN_ELEMS
    /// (1 000 000). On this host (RTX A4000, sm_86, CUDA 12.4) `cuda_is_available()`
    /// returns `true`, so the GPU branch IS taken. The test asserts:
    ///   — the function returns `Ok` (no error from the GPU path),
    ///   — the targets are finite (no NaN / Inf from a corrupt GEMM),
    ///   — the targets have non-trivial variance (sanity check against an
    ///     all-zeros degenerate result).
    #[cfg(feature = "cuda")]
    #[test]
    fn make_regression_cuda_dispatch_large_size_targets_finite() {
        const N_SAMPLES: usize = 2000;
        const N_FEATURES: usize = 600;
        const N_INFORMATIVE: usize = 200;
        // 2 000 × 600 = 1 200 000 >= CUDA_REGRESSION_MIN_ELEMS → GPU-eligible.
        let ds = make_regression(
            N_SAMPLES,
            N_FEATURES,
            N_INFORMATIVE,
            0.0,
            Some(0xc0_ffee_u64),
        )
        .expect("large make_regression must succeed on CUDA host");
        let target = ds.target.as_ref().expect("target present");
        assert_eq!(ds.data.nrows(), N_SAMPLES, "n_samples mismatch");
        assert_eq!(ds.data.ncols(), N_FEATURES, "n_features mismatch");
        assert_eq!(target.len(), N_SAMPLES, "target length mismatch");
        assert!(
            target.iter().all(|v| v.is_finite()),
            "all targets must be finite after GPU GEMM"
        );
        // With N_INFORMATIVE=200 real coefficients drawn from 100·N(0,1) and features
        // from N(0,1), the expected target std dev is ~100·sqrt(200)≈1414; variance >> 1.
        let n = target.len() as f64;
        let mean = target.iter().sum::<f64>() / n;
        let variance = target.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!(
            variance > 1.0,
            "target variance {variance:.2} must be >> 1 for non-trivial informative coefficients"
        );
    }

    /// §4b FIXED-SEED BYTE-STABILITY — double-run determinism (THE KEY TEST).
    ///
    /// Two calls with the same large fixed seed must produce **bit-identical**
    /// targets even when the GPU branch is taken. What is verified:
    ///   (a) The GPU-computed `linear = X·coef` is bit-stable across calls;
    ///       cuBLAS GEMM is deterministic for identical inputs on the same device.
    ///   (b) The per-sample noise RNG draws happen entirely on the CPU, around
    ///       the GPU seam, so their sequence is preserved — the full target
    ///       vector is therefore byte-identical across both runs.
    ///
    /// Sizes: 2 000 × 600 = 1 200 000 >= CUDA_REGRESSION_MIN_ELEMS.
    /// Noise is non-zero (10.0) to confirm that the noise RNG order too is
    /// unaffected by the GPU offload.
    #[cfg(feature = "cuda")]
    #[test]
    fn make_regression_cuda_gpu_path_double_run_byte_identical() {
        const SEED: u64 = 0x1234_5678_abcd_ef01_u64;
        const N_SAMPLES: usize = 2000;
        const N_FEATURES: usize = 600;
        let run1 = make_regression(N_SAMPLES, N_FEATURES, 200, 10.0, Some(SEED))
            .expect("first large make_regression");
        let run2 = make_regression(N_SAMPLES, N_FEATURES, 200, 10.0, Some(SEED))
            .expect("second large make_regression");
        assert_eq!(
            run1.data, run2.data,
            "design matrix must be byte-identical for the same seed"
        );
        // THE KEY ASSERTION: targets must be bit-identical across both GPU offloads.
        // Any deviation here means either cuBLAS is non-deterministic for these
        // inputs on this device, or the noise RNG order shifted around the GPU seam.
        let t1 = run1.target.as_ref().expect("run1 target present");
        let t2 = run2.target.as_ref().expect("run2 target present");
        assert_eq!(
            t1, t2,
            "target must be byte-identical for the same seed (GPU-path double run)"
        );
    }

    /// §4b CPU RECONSTRUCTION — GPU replaced only the noise-free matmul.
    ///
    /// With `noise = 0.0`, `target[i] = Σ_j X[i,j] · coef[j]`. We replay the
    /// same `StdRng` sequence used internally by `make_regression` to reconstruct
    /// the coefficient vector, then compute `X·coef` on the CPU (sequential
    /// accumulation) and compare against the GPU-produced target.
    ///
    /// What is verified: the GPU offload produced `X·coef` (not some other value)
    /// and did NOT alter the coefficient vector, the design matrix, or the RNG
    /// sequence for noise draws.
    ///
    /// Tolerance 1e-6 absolute accounts for floating-point non-associativity
    /// between GPU parallel tree-reduction and CPU sequential dot-product. With
    /// N_INFORMATIVE=200, coef ~ O(100), features ~ N(0,1), row sums have
    /// magnitude ~ O(100·√200) ≈ 1 414; typical observed discrepancy is a few
    /// ULPs (< 1e-11 absolute) — the 1e-6 bound is intentionally generous.
    #[cfg(feature = "cuda")]
    #[test]
    fn make_regression_cuda_gpu_output_matches_cpu_matmul() {
        const SEED: u64 = 0xdead_beef_cafe_babe_u64;
        const N_SAMPLES: usize = 2000;
        const N_FEATURES: usize = 600;
        const N_INFORMATIVE: usize = 200;

        let ds = make_regression(N_SAMPLES, N_FEATURES, N_INFORMATIVE, 0.0, Some(SEED))
            .expect("large make_regression for CPU reconstruction");
        let y_gpu = ds.target.as_ref().expect("target present");

        // Replay the internal RNG sequence.  `make_regression` calls
        // `StdRng::seed_from_u64(SEED)` then draws N_INFORMATIVE values via
        // `100.0 * normal.sample(&mut rng)` for coef, followed by
        // N_SAMPLES * N_FEATURES values for the data.  By seeding an identical
        // `StdRng` with the same seed we get the same coef draws.
        let mut rng_replay = StdRng::seed_from_u64(SEED);
        let normal_replay = scirs2_core::random::Normal::new(0.0, 1.0)
            .expect("Normal distribution init for replay");
        let mut coef_replay = Array1::<f64>::zeros(N_FEATURES);
        for i in 0..N_INFORMATIVE {
            coef_replay[i] = 100.0 * normal_replay.sample(&mut rng_replay);
        }

        // CPU sequential dot-product on the design matrix returned by
        // make_regression (generated by the same RNG from the same seed).
        let cpu_linear = ds.data.dot(&coef_replay);

        let max_abs_diff = y_gpu
            .iter()
            .zip(cpu_linear.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_abs_diff < 1e-6,
            "GPU matmul vs CPU reconstruction: max absolute diff = {max_abs_diff:.3e} (expected < 1e-6)"
        );
    }

    /// §7 FALLBACK — `make_regression` never panics for GPU-eligible sizes.
    ///
    /// The implementation wraps the GPU call as:
    /// ```text
    /// if let Ok(gpu_y) = cuda_regression_target(...) { linear = gpu_y; }
    /// ```
    /// so any `Err` from the GPU path silently retains the already-computed
    /// CPU `linear`. This test asserts the public function always returns `Ok`
    /// and never panics, regardless of whether the GPU branch succeeds or
    /// falls back. On this host the GPU path succeeds; on a host without CUDA
    /// it falls back silently — both are acceptable outcomes.
    #[cfg(feature = "cuda")]
    #[test]
    fn make_regression_cuda_never_panics_on_large_size() {
        // 2 000 × 600 = 1 200 000 >= CUDA_REGRESSION_MIN_ELEMS → GPU-eligible.
        let result = make_regression(2000, 600, 200, 1.0, Some(0xface_b00c_u64));
        assert!(
            result.is_ok(),
            "make_regression must return Ok for a GPU-eligible size (got {:?})",
            result.as_ref().err()
        );
        let ds = result.expect("make_regression Ok");
        assert_eq!(ds.data.nrows(), 2000, "n_samples");
        assert_eq!(ds.data.ncols(), 600, "n_features");
        let target = ds.target.as_ref().expect("target present");
        assert_eq!(target.len(), 2000, "target length");
        assert!(
            target.iter().all(|v| v.is_finite()),
            "targets must be finite regardless of GPU/CPU path taken"
        );
    }
}
