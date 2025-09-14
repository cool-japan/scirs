//! Dataset generators

use crate::error::{DatasetsError, Result};
use crate::gpu::{GpuContext, GpuDeviceInfo};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Uniform;
// Use local GPU implementation instead of core to avoid feature flag issues
use crate::gpu::GpuBackend as LocalGpuBackend;
// Parallel operations will be added as needed
// #[cfg(feature = "parallel")]
// use scirs2_core::parallel_ops::*;
use rand::seq::SliceRandom;
use scirs2_core::rng;
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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate centroids for each _class and cluster
    let n_centroids = n_classes * n_clusters_per_class;
    let mut centroids = Array2::zeros((n_centroids, n_informative));
    let scale = 2.0;

    for i in 0..n_centroids {
        for j in 0..n_informative {
            centroids[[i, j]] = scale * rng.gen_range(-1.0f64..1.0f64);
        }
    }

    // Generate _samples
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate the coefficients for the _informative _features
    let mut coef = Array1::zeros(n_features);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

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

    // Generate the target
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut y = 0.0;
        for j in 0..n_features {
            y += data[[i, j]] * coef[j];
        }

        // Add noise
        if noise > 0.0 {
            y += normal.sample(&mut rng) * noise;
        }

        target[i] = y;
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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::zeros((n_samples, n_features));

    for feature in 0..n_features {
        let trend_coef = if trend {
            rng.gen_range(0.01f64..0.1f64)
        } else {
            0.0
        };
        let seasonality_period = rng.sample(Uniform::new(10, 50).unwrap()) as f64;
        let seasonality_amplitude = if seasonality {
            rng.gen_range(1.0f64..5.0f64)
        } else {
            0.0
        };

        let base_value = rng.gen_range(-10.0f64..10.0f64);

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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.gen_range(-center_box..center_box);
        }
    }

    // Generate _samples around centers
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
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

/// Generate a Swiss roll dataset for dimensionality reduction
#[allow(dead_code)]
pub fn make_swiss_roll(n_samples: usize, noise: f64, randomseed: Option<u64>) -> Result<Dataset> {
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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples); // Color parameter for visualization

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
    } else {
        None
    };

    for i in 0..n_samples {
        // Parameter along the roll
        let t = 1.5 * PI * (1.0 + 2.0 * i as f64 / n_samples as f64);

        // Height parameter
        let height = 21.0 * i as f64 / n_samples as f64;

        let mut x = t * t.cos();
        let mut y = height;
        let mut z = t * t.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
            z += normal_dist.sample(&mut rng);
        }

        data[[i, 0]] = x;
        data[[i, 1]] = y;
        data[[i, 2]] = z;
        color[i] = t; // Color based on parameter for visualization
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_featurenames(vec!["x".to_string(), "y".to_string(), "z".to_string()])
        .with_description("Swiss roll manifold dataset for dimensionality reduction".to_string())
        .with_metadata("noise", &noise.to_string())
        .with_metadata("dimensions", "3")
        .with_metadata("manifold_dim", "2");

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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.gen_range(-center_box..center_box);
        }
    }

    // Generate _samples around centers with anisotropic distribution
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

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
        let rotation_angle = rng.gen_range(0.0..(2.0 * PI));

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
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate main cluster centers
    let mut main_centers = Array2::zeros((n_main_clusters, n_features));
    let center_box = 20.0;

    for i in 0..n_main_clusters {
        for j in 0..n_features {
            main_centers[[i, j]] = rng.gen_range(-center_box..center_box);
        }
    }

    let mut data = Array2::zeros((n_samples, n_features));
    let mut main_target = Array1::zeros(n_samples);
    let mut sub_target = Array1::zeros(n_samples);

    let main_normal = rand_distr::Normal::new(0.0, main_cluster_std).unwrap();
    let sub_normal = rand_distr::Normal::new(0.0, sub_cluster_std).unwrap();

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

/// Missing data patterns for noise injection
#[derive(Debug, Clone, Copy)]
pub enum MissingPattern {
    /// Missing Completely at Random - uniform probability across all features
    MCAR,
    /// Missing at Random - probability depends on observed values
    MAR,
    /// Missing Not at Random - probability depends on missing values themselves
    MNAR,
    /// Block-wise missing - entire blocks of consecutive features/samples missing
    Block,
}

/// Outlier types for injection
#[derive(Debug, Clone, Copy)]
pub enum OutlierType {
    /// Point outliers - individual data points that are anomalous
    Point,
    /// Contextual outliers - points anomalous in specific contexts
    Contextual,
    /// Collective outliers - groups of points that together form an anomaly
    Collective,
}

/// Inject missing data into a dataset with realistic patterns
#[allow(dead_code)]
pub fn inject_missing_data(
    data: &mut Array2<f64>,
    missing_rate: f64,
    pattern: MissingPattern,
    randomseed: Option<u64>,
) -> Result<Array2<bool>> {
    // Validate input parameters
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(DatasetsError::InvalidFormat(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let mut missing_mask = Array2::from_elem((n_samples, n_features), false);

    match pattern {
        MissingPattern::MCAR => {
            // Missing Completely at Random - uniform probability
            for i in 0..n_samples {
                for j in 0..n_features {
                    if rng.gen_range(0.0f64..1.0) < missing_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::MAR => {
            // Missing at Random - probability depends on first feature
            for i in 0..n_samples {
                let first_feature_val = data[[i, 0]];
                let normalized_val = (first_feature_val + 10.0) / 20.0; // Normalize roughly to [0,1]
                let adjusted_rate = missing_rate * normalized_val.clamp(0.1, 2.0);

                for j in 1..n_features {
                    // Skip first feature
                    if rng.gen_range(0.0f64..1.0) < adjusted_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::MNAR => {
            // Missing Not at Random - higher values more likely to be missing
            for i in 0..n_samples {
                for j in 0..n_features {
                    let value = data[[i, j]];
                    let normalized_val = (value + 10.0) / 20.0; // Normalize roughly to [0,1]
                    let adjusted_rate = missing_rate * normalized_val.clamp(0.1, 3.0);

                    if rng.gen_range(0.0f64..1.0) < adjusted_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::Block => {
            // Block-wise missing - entire blocks are missing
            let block_size = (n_features as f64 * missing_rate).ceil() as usize;
            let n_blocks = (missing_rate * n_samples as f64).ceil() as usize;

            for _ in 0..n_blocks {
                let start_row = rng.sample(Uniform::new(0, n_samples).unwrap());
                let start_col =
                    rng.sample(Uniform::new(0, n_features.saturating_sub(block_size)).unwrap());

                for i in start_row..n_samples.min(start_row + block_size) {
                    for j in start_col..n_features.min(start_col + block_size) {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
    }

    Ok(missing_mask)
}

/// Inject outliers into a dataset
#[allow(dead_code)]
pub fn inject_outliers(
    data: &mut Array2<f64>,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    randomseed: Option<u64>,
) -> Result<Array1<bool>> {
    // Validate input parameters
    if !(0.0..=1.0).contains(&outlier_rate) {
        return Err(DatasetsError::InvalidFormat(
            "outlier_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if outlier_strength <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "outlier_strength must be > 0.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let n_outliers = (n_samples as f64 * outlier_rate).ceil() as usize;
    let mut outlier_mask = Array1::from_elem(n_samples, false);

    // Calculate data statistics for outlier generation
    let mut feature_means = vec![0.0; n_features];
    let mut feature_stds = vec![0.0; n_features];

    for j in 0..n_features {
        let column = data.column(j);
        let valid_values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

        if !valid_values.is_empty() {
            feature_means[j] = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let variance = valid_values
                .iter()
                .map(|&x| (x - feature_means[j]).powi(2))
                .sum::<f64>()
                / valid_values.len() as f64;
            feature_stds[j] = variance.sqrt().max(1.0); // Use minimum std of 1.0 to ensure outliers can be created
        }
    }

    match outlier_type {
        OutlierType::Point => {
            // Point outliers - individual anomalous points
            for _ in 0..n_outliers {
                let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                outlier_mask[outlier_idx] = true;

                // Modify each feature to be an outlier
                for j in 0..n_features {
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    data[[outlier_idx, j]] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }
            }
        }
        OutlierType::Contextual => {
            // Contextual outliers - anomalous in specific feature combinations
            for _ in 0..n_outliers {
                let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                outlier_mask[outlier_idx] = true;

                // Only modify a subset of features to create contextual anomaly
                let n_features_to_modify =
                    rng.sample(Uniform::new(1, (n_features / 2).max(1) + 1).unwrap());
                let mut features_to_modify: Vec<usize> = (0..n_features).collect();
                features_to_modify.shuffle(&mut rng);
                features_to_modify.truncate(n_features_to_modify);

                for &j in &features_to_modify {
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    data[[outlier_idx, j]] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }
            }
        }
        OutlierType::Collective => {
            // Collective outliers - groups of points that together form anomalies
            let outliers_per_group = (n_outliers / 3).max(2); // At least 2 per group
            let n_groups = (n_outliers / outliers_per_group).max(1);

            for _ in 0..n_groups {
                // Generate cluster center for this collective outlier
                let mut outlier_center = vec![0.0; n_features];
                for j in 0..n_features {
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    outlier_center[j] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }

                // Generate points around this center
                for _ in 0..outliers_per_group {
                    let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                    outlier_mask[outlier_idx] = true;

                    for j in 0..n_features {
                        let noise = rng.gen_range(-0.5f64..0.5f64) * feature_stds[j];
                        data[[outlier_idx, j]] = outlier_center[j] + noise;
                    }
                }
            }
        }
    }

    Ok(outlier_mask)
}

/// Add realistic noise patterns to time series data
#[allow(dead_code)]
pub fn add_time_series_noise(
    data: &mut Array2<f64>,
    noise_types: &[(&str, f64)], // (noise_type, strength)
    randomseed: Option<u64>,
) -> Result<()> {
    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for &(noise_type, strength) in noise_types {
        match noise_type {
            "gaussian" => {
                // Add Gaussian white noise
                for i in 0..n_samples {
                    for j in 0..n_features {
                        data[[i, j]] += strength * normal.sample(&mut rng);
                    }
                }
            }
            "spikes" => {
                // Add random spikes (impulse noise)
                let n_spikes = (n_samples as f64 * strength * 0.1).ceil() as usize;
                for _ in 0..n_spikes {
                    let spike_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                    let feature_idx = rng.sample(Uniform::new(0, n_features).unwrap());
                    let spike_magnitude = rng.gen_range(5.0..=15.0) * strength;
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };

                    data[[spike_idx, feature_idx]] += direction * spike_magnitude;
                }
            }
            "drift" => {
                // Add gradual drift over time
                for i in 0..n_samples {
                    let drift_amount = strength * (i as f64 / n_samples as f64);
                    for j in 0..n_features {
                        data[[i, j]] += drift_amount;
                    }
                }
            }
            "seasonal" => {
                // Add seasonal pattern noise
                let period = n_samples as f64 / 4.0; // 4 seasons
                for i in 0..n_samples {
                    let seasonal_component = strength * (2.0 * PI * i as f64 / period).sin();
                    for j in 0..n_features {
                        data[[i, j]] += seasonal_component;
                    }
                }
            }
            "autocorrelated" => {
                // Add autocorrelated noise (AR(1) process)
                let ar_coeff = 0.7; // Autocorrelation coefficient
                for j in 0..n_features {
                    let mut prev_noise = 0.0;
                    for i in 0..n_samples {
                        let new_noise = ar_coeff * prev_noise + strength * normal.sample(&mut rng);
                        data[[i, j]] += new_noise;
                        prev_noise = new_noise;
                    }
                }
            }
            "heteroscedastic" => {
                // Add heteroscedastic noise (variance changes over time)
                for i in 0..n_samples {
                    let variance_factor = 1.0 + strength * (i as f64 / n_samples as f64);
                    for j in 0..n_features {
                        data[[i, j]] += variance_factor * strength * normal.sample(&mut rng);
                    }
                }
            }
            _ => {
                return Err(DatasetsError::InvalidFormat(format!(
                    "Unknown noise type: {noise_type}. Supported , types: gaussian, spikes, drift, seasonal, autocorrelated, heteroscedastic"
                )));
            }
        }
    }

    Ok(())
}

/// Generate a dataset with controlled corruption patterns
#[allow(dead_code)]
pub fn make_corrupted_dataset(
    base_dataset: &Dataset,
    missing_rate: f64,
    missing_pattern: MissingPattern,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    // Validate inputs
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(DatasetsError::InvalidFormat(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&outlier_rate) {
        return Err(DatasetsError::InvalidFormat(
            "outlier_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Clone the base _dataset
    let mut corrupted_data = base_dataset.data.clone();
    let corrupted_target = base_dataset.target.clone();

    // Apply missing data
    let missing_mask = inject_missing_data(
        &mut corrupted_data,
        missing_rate,
        missing_pattern,
        randomseed,
    )?;

    // Apply outliers
    let outlier_mask = inject_outliers(
        &mut corrupted_data,
        outlier_rate,
        outlier_type,
        outlier_strength,
        randomseed,
    )?;

    // Create new _dataset with corruption metadata
    let mut corrupted_dataset = Dataset::new(corrupted_data, corrupted_target);

    if let Some(featurenames) = &base_dataset.featurenames {
        corrupted_dataset = corrupted_dataset.with_featurenames(featurenames.clone());
    }

    if let Some(targetnames) = &base_dataset.targetnames {
        corrupted_dataset = corrupted_dataset.with_targetnames(targetnames.clone());
    }

    corrupted_dataset = corrupted_dataset
        .with_description(format!(
            "Corrupted version of: {}",
            base_dataset
                .description
                .as_deref()
                .unwrap_or("Unknown _dataset")
        ))
        .with_metadata("missing_rate", &missing_rate.to_string())
        .with_metadata("missing_pattern", &format!("{missing_pattern:?}"))
        .with_metadata("outlier_rate", &outlier_rate.to_string())
        .with_metadata("outlier_type", &format!("{outlier_type:?}"))
        .with_metadata("outlier_strength", &outlier_strength.to_string())
        .with_metadata(
            "missing_count",
            &missing_mask.iter().filter(|&&x| x).count().to_string(),
        )
        .with_metadata(
            "outlier_count",
            &outlier_mask.iter().filter(|&&x| x).count().to_string(),
        );

    Ok(corrupted_dataset)
}

/// GPU-accelerated data generation configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// GPU device index (0 for default)
    pub device_id: usize,
    /// Whether to use single precision (f32) instead of double (f64)
    pub use_single_precision: bool,
    /// Chunk size for GPU operations
    pub chunk_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            device_id: 0,
            use_single_precision: false,
            chunk_size: 10000,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to use GPU
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set GPU device ID
    pub fn with_device(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set precision mode
    pub fn with_single_precision(mut self, single_precision: bool) -> Self {
        self.use_single_precision = single_precision;
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

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
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let mut targets = vec![0.0; n_samples];

    // Matrix multiplication for regression targets
    for i in 0..n_samples {
        let mut target = 0.0;
        for j in 0..n_informative {
            target += data[i * n_features + j] * coefficients[j];
        }

        // Add noise
        if noise > 0.0 {
            target += noise * normal.sample(rng);
        }

        targets[i] = target;
    }

    let data_vec = data.to_vec();

    // TODO: Future GPU implementation placeholder - currently using CPU fallback

    Ok((data_vec, targets))
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

    // TODO: GPU implementation placeholder - using CPU fallback above
    /*
    let _center_buffer = (); // gpu_context.create_buffer_from_slice(&center_coords);
    let _data_buffer = (); // gpu_context.create_buffer::<f64>(n_samples_center * n_features);

    // Generate random seeds for each sample
    let seeds: Vec<u64> = (0..n_samples_center).map(|_| rng.random::<u64>()).collect();
    let seeds_buffer = gpu_context.create_buffer_from_slice(&seeds);

    // Use GPU kernel for parallel sample generation around _center
    gpu_context
        .execute(|compiler| {
            let kernel_source = r#"
            #version 450
            layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer CenterBuffer {
                double center_coords[];
            };

            layout(set = 0, binding = 1) buffer DataBuffer {
                double data[];
            };

            layout(set = 0, binding = 2) buffer SeedsBuffer {
                uint64_t seeds[];
            };

            layout(push_constant) uniform Params {
                uint n_samples_center;
                uint n_features;
                double cluster_std;
            } params;

            // High-quality pseudo-random number generator
            uint wang_hash(uint seed) {
                seed = (seed ^ 61u) ^ (seed >> 16u);
                seed *= 9u;
                seed = seed ^ (seed >> 4u);
                seed *= 0x27d4eb2du;
                seed = seed ^ (seed >> 15u);
                return seed;
            }

            // Generate Gaussian random numbers using Box-Muller transform
            double random_normal(uint seed, uint index) {
                uint h1 = wang_hash(seed + index * 2u);
                uint h2 = wang_hash(seed + index * 2u + 1u);
                double u1 = double(h1) / double(0xFFFFFFFFu);
                double u2 = double(h2) / double(0xFFFFFFFFu);

                // Ensure u1 is not zero to avoid log(0)
                u1 = max(u1, 1e-8);

                // Box-Muller transform
                return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
            }

            void main() {
                uint sample_idx = gl_GlobalInvocationID.x;
                if (sample_idx >= params.n_samples_center) return;

                uint seed = uint(seeds[sample_idx]);

                // Generate all _features for this sample
                for (uint j = 0; j < params.n_features; j++) {
                    // Get _center coordinate for this feature
                    double center_val = center_coords[j];

                    // Generate Gaussian noise
                    double noise = random_normal(seed, j) * params.cluster_std;

                    // Set the data point
                    data[sample_idx * params.n_features + j] = center_val + noise;
                }
            }
        "#;

            let kernel = compiler.compile(kernel_source)?;

            // Set kernel parameters
            kernel.set_buffer("center_coords", &center_buffer);
            kernel.set_buffer("data", &data_buffer);
            kernel.set_buffer("seeds", &seeds_buffer);
            kernel.set_u32("n_samples_center", n_samples_center as u32);
            kernel.set_u32("n_features", n_features as u32);
            kernel.set_f64("cluster_std", cluster_std);

            // Dispatch the kernel with optimal work group size
            let work_groups = [(n_samples_center + 255) / 256, 1, 1];
            kernel.dispatch(work_groups);

            Ok(())
        })
        .map_err(|e| {
            DatasetsError::Other(format!(
                "GPU blob generation kernel execution failed: {}",
                e
            ))
        })?;

    // Copy results back to CPU and restructure as Vec<Vec<f64>>
    let flat_data = data_buffer.to_vec();
    let mut result = Vec::with_capacity(n_samples_center);

    for i in 0..n_samples_center {
        let mut sample = Vec::with_capacity(n_features);
        for j in 0..n_features {
            sample.push(flat_data[i * n_features + j]);
        }
        result.push(sample);
    }

    */
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

// Advanced manifold learning datasets

/// Generate a dataset with an S-curve manifold embedded in 3D space
#[allow(dead_code)]
pub fn make_s_curve(n_samples: usize, noise: f64, randomseed: Option<u64>) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();

    for i in 0..n_samples {
        // Parameter t ranges from 0 to 4π
        let t = 4.0 * PI * (i as f64) / (n_samples as f64 - 1.0);

        // S-curve parametric equations
        data[[i, 0]] = t.sin() + noise_dist.sample(&mut rng);
        data[[i, 1]] = 2.0 * t + noise_dist.sample(&mut rng);
        data[[i, 2]] = (t / 2.0).sin() + noise_dist.sample(&mut rng);

        // Color represents the position along the curve
        color[i] = t;
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_description("S-curve manifold embedded in 3D space".to_string());

    Ok(dataset)
}

/// Generate a dataset sampling from a Swiss roll manifold
#[allow(dead_code)]
pub fn make_swiss_roll_advanced(
    n_samples: usize,
    noise: f64,
    hole: bool,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();
    let uniform = rand_distr::Uniform::new(0.0, 1.0).unwrap();

    for i in 0..n_samples {
        // Sample parameters
        let mut t = uniform.sample(&mut rng) * 3.0 * PI / 2.0;
        let mut y = uniform.sample(&mut rng) * 20.0;

        // Create hole if requested
        if hole {
            // Create a hole by rejecting _samples in the middle region
            while t > PI / 2.0 && t < PI && y > 8.0 && y < 12.0 {
                t = uniform.sample(&mut rng) * 3.0 * PI / 2.0;
                y = uniform.sample(&mut rng) * 20.0;
            }
        }

        // Swiss roll parametric equations
        data[[i, 0]] = t * t.cos() + noise_dist.sample(&mut rng);
        data[[i, 1]] = y + noise_dist.sample(&mut rng);
        data[[i, 2]] = t * t.sin() + noise_dist.sample(&mut rng);

        // Color represents position
        color[i] = t;
    }

    let mut dataset = Dataset::new(data, Some(color));
    let description = if hole {
        "Swiss roll manifold with hole embedded in 3D space"
    } else {
        "Swiss roll manifold embedded in 3D space"
    };

    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_description(description.to_string());

    Ok(dataset)
}

/// Generate a dataset from a severed sphere (broken manifold)
#[allow(dead_code)]
pub fn make_severed_sphere(
    n_samples: usize,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();
    let uniform = rand_distr::Uniform::new(0.0, 1.0).unwrap();

    for i in 0..n_samples {
        // Sample spherical coordinates, but exclude a region to "sever" the sphere
        let mut phi = uniform.sample(&mut rng) * 2.0 * PI; // azimuthal angle
        let mut theta = uniform.sample(&mut rng) * PI; // polar angle

        // Create a severed region by excluding certain angles
        while phi > PI / 3.0 && phi < 2.0 * PI / 3.0 && theta > PI / 3.0 && theta < 2.0 * PI / 3.0 {
            phi = uniform.sample(&mut rng) * 2.0 * PI;
            theta = uniform.sample(&mut rng) * PI;
        }

        let radius = 1.0; // Unit sphere

        // Convert to Cartesian coordinates
        data[[i, 0]] = radius * theta.sin() * phi.cos() + noise_dist.sample(&mut rng);
        data[[i, 1]] = radius * theta.sin() * phi.sin() + noise_dist.sample(&mut rng);
        data[[i, 2]] = radius * theta.cos() + noise_dist.sample(&mut rng);

        // Color based on position
        color[i] = phi;
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_description("Severed sphere manifold with discontinuities".to_string());

    Ok(dataset)
}

/// Generate a dataset from a twin peaks manifold (two connected peaks)
#[allow(dead_code)]
pub fn make_twin_peaks(n_samples: usize, noise: f64, randomseed: Option<u64>) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut labels = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();
    let uniform = rand_distr::Uniform::new(-2.0, 2.0).unwrap();

    for i in 0..n_samples {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);

        // Twin peaks function: two Gaussian peaks
        let peak1 = (-(((x as f64) - 1.0).powi(2) + ((y as f64) - 1.0).powi(2))).exp();
        let peak2 = (-(((x as f64) + 1.0).powi(2) + ((y as f64) + 1.0).powi(2))).exp();
        let z = peak1 + peak2 + noise_dist.sample(&mut rng);

        data[[i, 0]] = x;
        data[[i, 1]] = y;
        data[[i, 2]] = z;

        // Label based on which peak is closer
        labels[i] = if ((x as f64) - 1.0).powi(2) + ((y as f64) - 1.0).powi(2)
            < ((x as f64) + 1.0).powi(2) + ((y as f64) + 1.0).powi(2)
        {
            0.0
        } else {
            1.0
        };
    }

    let mut dataset = Dataset::new(data, Some(labels));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_targetnames(vec!["peak_0".to_string(), "peak_1".to_string()])
        .with_description("Twin peaks manifold with two connected Gaussian peaks".to_string());

    Ok(dataset)
}

/// Generate a dataset from a helix manifold in 3D space
#[allow(dead_code)]
pub fn make_helix(
    n_samples: usize,
    n_turns: f64,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_turns <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "n_turns must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();

    for i in 0..n_samples {
        // Parameter t ranges from 0 to n_turns * 2π
        let t = n_turns * 2.0 * PI * (i as f64) / (n_samples as f64 - 1.0);

        // Helix parametric equations
        data[[i, 0]] = t.cos() + noise_dist.sample(&mut rng);
        data[[i, 1]] = t.sin() + noise_dist.sample(&mut rng);
        data[[i, 2]] = t / (n_turns * 2.0 * PI) + noise_dist.sample(&mut rng); // Normalized height

        // Color represents position along helix
        color[i] = t;
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_description(format!("Helix manifold with {n_turns} _turns in 3D space"));

    Ok(dataset)
}

/// Generate a dataset from an intersecting manifolds (two intersecting planes)
#[allow(dead_code)]
pub fn make_intersecting_manifolds(
    n_samples: usize,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let samples_per_manifold = n_samples / 2;
    let remainder = n_samples % 2;

    let mut data = Array2::zeros((n_samples, 3));
    let mut labels = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();
    let uniform = rand_distr::Uniform::new(-2.0, 2.0).unwrap();

    let mut sample_idx = 0;

    // First manifold: plane z = x
    for _ in 0..samples_per_manifold + remainder {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        let z = x + noise_dist.sample(&mut rng);

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        data[[sample_idx, 2]] = z;
        labels[sample_idx] = 0.0;
        sample_idx += 1;
    }

    // Second manifold: plane z = -x
    for _ in 0..samples_per_manifold {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        let z = -x + noise_dist.sample(&mut rng);

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        data[[sample_idx, 2]] = z;
        labels[sample_idx] = 1.0;
        sample_idx += 1;
    }

    let mut dataset = Dataset::new(data, Some(labels));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_targetnames(vec!["manifold_0".to_string(), "manifold_1".to_string()])
        .with_description("Two intersecting plane manifolds in 3D space".to_string());

    Ok(dataset)
}

/// Generate a dataset from a torus manifold in 3D space
#[allow(dead_code)]
pub fn make_torus(
    n_samples: usize,
    major_radius: f64,
    minor_radius: f64,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if major_radius <= 0.0 || minor_radius <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "major_radius and minor_radius must be > 0".to_string(),
        ));
    }

    if minor_radius >= major_radius {
        return Err(DatasetsError::InvalidFormat(
            "minor_radius must be < major_radius".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples);

    let noise_dist = rand_distr::Normal::new(0.0, noise).unwrap();
    let uniform = rand_distr::Uniform::new(0.0, 2.0 * PI).unwrap();

    for i in 0..n_samples {
        let theta = uniform.sample(&mut rng); // Major angle
        let phi = uniform.sample(&mut rng); // Minor angle

        // Torus parametric equations
        data[[i, 0]] =
            (major_radius + minor_radius * phi.cos()) * theta.cos() + noise_dist.sample(&mut rng);
        data[[i, 1]] =
            (major_radius + minor_radius * phi.cos()) * theta.sin() + noise_dist.sample(&mut rng);
        data[[i, 2]] = minor_radius * phi.sin() + noise_dist.sample(&mut rng);

        // Color based on major angle
        color[i] = theta;
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_featurenames(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])
        .with_description(format!(
            "Torus manifold with major _radius {major_radius} and minor _radius {minor_radius}"
        ));

    Ok(dataset)
}

/// Advanced manifold configuration for complex datasets
#[derive(Debug, Clone)]
pub struct ManifoldConfig {
    /// Type of manifold to generate
    pub manifold_type: ManifoldType,
    /// Number of samples
    pub n_samples: usize,
    /// Noise level
    pub noise: f64,
    /// Random seed
    pub randomseed: Option<u64>,
    /// Manifold-specific parameters
    pub parameters: std::collections::HashMap<String, f64>,
}

/// Types of manifolds that can be generated
#[derive(Debug, Clone)]
pub enum ManifoldType {
    /// S-curve manifold
    SCurve,
    /// Swiss roll (with optional hole)
    SwissRoll {
        /// Whether to create a hole in the middle
        hole: bool,
    },
    /// Severed sphere
    SeveredSphere,
    /// Twin peaks
    TwinPeaks,
    /// Helix with specified turns
    Helix {
        /// Number of turns in the helix
        n_turns: f64,
    },
    /// Intersecting manifolds
    IntersectingManifolds,
    /// Torus with major and minor radii
    Torus {
        /// Major radius of the torus
        major_radius: f64,
        /// Minor radius of the torus
        minor_radius: f64,
    },
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            manifold_type: ManifoldType::SCurve,
            n_samples: 1000,
            noise: 0.1,
            randomseed: None,
            parameters: std::collections::HashMap::new(),
        }
    }
}

impl ManifoldConfig {
    /// Create a new manifold configuration
    pub fn new(manifold_type: ManifoldType) -> Self {
        Self {
            manifold_type,
            ..Default::default()
        }
    }

    /// Set number of samples
    pub fn with_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set noise level
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.randomseed = Some(seed);
        self
    }

    /// Add a parameter
    pub fn with_parameter(mut self, name: String, value: f64) -> Self {
        self.parameters.insert(name, value);
        self
    }
}

/// Generate a manifold dataset based on configuration
#[allow(dead_code)]
pub fn make_manifold(config: ManifoldConfig) -> Result<Dataset> {
    match config.manifold_type {
        ManifoldType::SCurve => make_s_curve(config.n_samples, config.noise, config.randomseed),
        ManifoldType::SwissRoll { hole } => {
            make_swiss_roll_advanced(config.n_samples, config.noise, hole, config.randomseed)
        }
        ManifoldType::SeveredSphere => {
            make_severed_sphere(config.n_samples, config.noise, config.randomseed)
        }
        ManifoldType::TwinPeaks => {
            make_twin_peaks(config.n_samples, config.noise, config.randomseed)
        }
        ManifoldType::Helix { n_turns } => {
            make_helix(config.n_samples, n_turns, config.noise, config.randomseed)
        }
        ManifoldType::IntersectingManifolds => {
            make_intersecting_manifolds(config.n_samples, config.noise, config.randomseed)
        }
        ManifoldType::Torus {
            major_radius,
            minor_radius,
        } => make_torus(
            config.n_samples,
            major_radius,
            minor_radius,
            config.noise,
            config.randomseed,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Uniform;

    #[test]
    fn test_make_classification_invalid_params() {
        // Test zero n_samples
        assert!(make_classification(0, 5, 2, 1, 3, None).is_err());

        // Test zero n_features
        assert!(make_classification(10, 0, 2, 1, 3, None).is_err());

        // Test zero n_informative
        assert!(make_classification(10, 5, 2, 1, 0, None).is_err());

        // Test n_features < n_informative
        assert!(make_classification(10, 3, 2, 1, 5, None).is_err());

        // Test n_classes < 2
        assert!(make_classification(10, 5, 1, 1, 3, None).is_err());

        // Test zero n_clusters_per_class
        assert!(make_classification(10, 5, 2, 0, 3, None).is_err());
    }

    #[test]
    fn test_make_regression_invalid_params() {
        // Test zero n_samples
        assert!(make_regression(0, 5, 3, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_regression(10, 0, 3, 1.0, None).is_err());

        // Test zero n_informative
        assert!(make_regression(10, 5, 0, 1.0, None).is_err());

        // Test n_features < n_informative
        assert!(make_regression(10, 3, 5, 1.0, None).is_err());

        // Test negative noise
        assert!(make_regression(10, 5, 3, -1.0, None).is_err());
    }

    #[test]
    fn test_make_time_series_invalid_params() {
        // Test zero n_samples
        assert!(make_time_series(0, 3, false, false, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_time_series(10, 0, false, false, 1.0, None).is_err());

        // Test negative noise
        assert!(make_time_series(10, 3, false, false, -1.0, None).is_err());
    }

    #[test]
    fn test_make_blobs_invalid_params() {
        // Test zero n_samples
        assert!(make_blobs(0, 3, 2, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_blobs(10, 0, 2, 1.0, None).is_err());

        // Test zero centers
        assert!(make_blobs(10, 3, 0, 1.0, None).is_err());

        // Test zero or negative cluster_std
        assert!(make_blobs(10, 3, 2, 0.0, None).is_err());
        assert!(make_blobs(10, 3, 2, -1.0, None).is_err());
    }

    #[test]
    fn test_make_classification_valid_params() {
        let dataset = make_classification(20, 5, 3, 2, 4, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 20);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());
        assert!(dataset.targetnames.is_some());
    }

    #[test]
    fn test_make_regression_valid_params() {
        let dataset = make_regression(15, 4, 3, 0.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 15);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());
    }

    #[test]
    fn test_make_time_series_valid_params() {
        let dataset = make_time_series(25, 3, true, true, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 25);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.featurenames.is_some());
        // Time series doesn't have targets by default
        assert!(dataset.target.is_none());
    }

    #[test]
    fn test_make_blobs_valid_params() {
        let dataset = make_blobs(30, 4, 3, 1.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 30);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());
    }

    #[test]
    fn test_make_spirals_invalid_params() {
        // Test zero n_samples
        assert!(make_spirals(0, 2, 0.1, None).is_err());

        // Test zero n_spirals
        assert!(make_spirals(100, 0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_spirals(100, 2, -0.1, None).is_err());
    }

    #[test]
    fn test_make_spirals_valid_params() {
        let dataset = make_spirals(100, 2, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());

        // Check that we have the right number of spirals
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_moons_invalid_params() {
        // Test zero n_samples
        assert!(make_moons(0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_moons(100, -0.1, None).is_err());
    }

    #[test]
    fn test_make_moons_valid_params() {
        let dataset = make_moons(100, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());

        // Check that we have exactly 2 classes (2 moons)
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_circles_invalid_params() {
        // Test zero n_samples
        assert!(make_circles(0, 0.5, 0.1, None).is_err());

        // Test invalid factor (must be between 0 and 1)
        assert!(make_circles(100, 0.0, 0.1, None).is_err());
        assert!(make_circles(100, 1.0, 0.1, None).is_err());
        assert!(make_circles(100, 1.5, 0.1, None).is_err());

        // Test negative noise
        assert!(make_circles(100, 0.5, -0.1, None).is_err());
    }

    #[test]
    fn test_make_circles_valid_params() {
        let dataset = make_circles(100, 0.5, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());

        // Check that we have exactly 2 classes (inner and outer circle)
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_swiss_roll_invalid_params() {
        // Test zero n_samples
        assert!(make_swiss_roll(0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_swiss_roll(100, -0.1, None).is_err());
    }

    #[test]
    fn test_make_swiss_roll_valid_params() {
        let dataset = make_swiss_roll(100, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some()); // Color parameter
        assert!(dataset.featurenames.is_some());
    }

    #[test]
    fn test_make_anisotropic_blobs_invalid_params() {
        // Test zero n_samples
        assert!(make_anisotropic_blobs(0, 3, 2, 1.0, 2.0, None).is_err());

        // Test insufficient features
        assert!(make_anisotropic_blobs(100, 1, 2, 1.0, 2.0, None).is_err());

        // Test zero centers
        assert!(make_anisotropic_blobs(100, 3, 0, 1.0, 2.0, None).is_err());

        // Test invalid std
        assert!(make_anisotropic_blobs(100, 3, 2, 0.0, 2.0, None).is_err());

        // Test invalid anisotropy factor
        assert!(make_anisotropic_blobs(100, 3, 2, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_anisotropic_blobs_valid_params() {
        let dataset = make_anisotropic_blobs(100, 3, 2, 1.0, 3.0, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());

        // Check that we have the right number of clusters
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_hierarchical_clusters_invalid_params() {
        // Test zero n_samples
        assert!(make_hierarchical_clusters(0, 3, 2, 3, 1.0, 0.5, None).is_err());

        // Test zero features
        assert!(make_hierarchical_clusters(100, 0, 2, 3, 1.0, 0.5, None).is_err());

        // Test zero main clusters
        assert!(make_hierarchical_clusters(100, 3, 0, 3, 1.0, 0.5, None).is_err());

        // Test zero sub clusters
        assert!(make_hierarchical_clusters(100, 3, 2, 0, 1.0, 0.5, None).is_err());

        // Test invalid main cluster std
        assert!(make_hierarchical_clusters(100, 3, 2, 3, 0.0, 0.5, None).is_err());

        // Test invalid sub cluster std
        assert!(make_hierarchical_clusters(100, 3, 2, 3, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_hierarchical_clusters_valid_params() {
        let dataset = make_hierarchical_clusters(120, 3, 2, 3, 2.0, 0.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 120);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some());
        assert!(dataset.featurenames.is_some());

        // Check that we have the right number of main clusters
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2); // 2 main clusters
        }

        // Check metadata contains sub-cluster information
        assert!(dataset.metadata.contains_key("sub_cluster_labels"));
    }

    #[test]
    fn test_inject_missing_data_invalid_params() {
        let mut data = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();

        // Test invalid missing rate
        assert!(inject_missing_data(&mut data, -0.1, MissingPattern::MCAR, None).is_err());
        assert!(inject_missing_data(&mut data, 1.5, MissingPattern::MCAR, None).is_err());
    }

    #[test]
    fn test_inject_missing_data_mcar() {
        let mut data =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let original_data = data.clone();

        let missing_mask =
            inject_missing_data(&mut data, 0.3, MissingPattern::MCAR, Some(42)).unwrap();

        // Check that some data is missing
        let missing_count = missing_mask.iter().filter(|&&x| x).count();
        assert!(missing_count > 0);

        // Check that missing values are NaN
        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                assert!(data[[i, j]].is_nan());
            } else {
                assert_eq!(data[[i, j]], original_data[[i, j]]);
            }
        }
    }

    #[test]
    fn test_inject_outliers_invalid_params() {
        let mut data = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();

        // Test invalid outlier rate
        assert!(inject_outliers(&mut data, -0.1, OutlierType::Point, 2.0, None).is_err());
        assert!(inject_outliers(&mut data, 1.5, OutlierType::Point, 2.0, None).is_err());

        // Test invalid outlier strength
        assert!(inject_outliers(&mut data, 0.1, OutlierType::Point, 0.0, None).is_err());
        assert!(inject_outliers(&mut data, 0.1, OutlierType::Point, -1.0, None).is_err());
    }

    #[test]
    fn test_inject_outliers_point() {
        let mut data = Array2::from_shape_vec((20, 2), vec![1.0; 40]).unwrap();

        let outlier_mask =
            inject_outliers(&mut data, 0.2, OutlierType::Point, 3.0, Some(42)).unwrap();

        // Check that some outliers were created
        let outlier_count = outlier_mask.iter().filter(|&&x| x).count();
        assert!(outlier_count > 0);

        // Check that outliers are different from original values
        for (i, &is_outlier) in outlier_mask.iter().enumerate() {
            if is_outlier {
                // At least one feature should be different from 1.0
                let row = data.row(i);
                assert!(row.iter().any(|&x| (x - 1.0).abs() > 1.0));
            }
        }
    }

    #[test]
    fn test_add_time_series_noise() {
        let mut data = Array2::zeros((100, 2));

        let noise_types = [("gaussian", 0.1), ("spikes", 0.05), ("drift", 0.2)];

        let original_data = data.clone();
        add_time_series_noise(&mut data, &noise_types, Some(42)).unwrap();

        // Check that data has been modified
        assert!(!data
            .iter()
            .zip(original_data.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-10));

        // Test invalid noise type
        let invalid_noise = [("invalid_type", 0.1)];
        let mut test_data = Array2::zeros((10, 2));
        assert!(add_time_series_noise(&mut test_data, &invalid_noise, Some(42)).is_err());
    }

    #[test]
    fn test_make_corrupted_dataset() {
        let base_dataset = make_blobs(50, 3, 2, 1.0, Some(42)).unwrap();

        let corrupted = make_corrupted_dataset(
            &base_dataset,
            0.1, // 10% missing
            MissingPattern::MCAR,
            0.05, // 5% outliers
            OutlierType::Point,
            2.0, // outlier strength
            Some(42),
        )
        .unwrap();

        // Check basic properties
        assert_eq!(corrupted.n_samples(), base_dataset.n_samples());
        assert_eq!(corrupted.n_features(), base_dataset.n_features());

        // Check metadata
        assert!(corrupted.metadata.contains_key("missing_rate"));
        assert!(corrupted.metadata.contains_key("outlier_rate"));
        assert!(corrupted.metadata.contains_key("missing_count"));
        assert!(corrupted.metadata.contains_key("outlier_count"));

        // Check some data is corrupted
        let has_nan = corrupted.data.iter().any(|&x| x.is_nan());
        assert!(has_nan, "Dataset should have some missing values");
    }

    #[test]
    fn test_make_corrupted_dataset_invalid_params() {
        let base_dataset = make_blobs(20, 2, 2, 1.0, Some(42)).unwrap();

        // Test invalid missing rate
        assert!(make_corrupted_dataset(
            &base_dataset,
            -0.1,
            MissingPattern::MCAR,
            0.0,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
        assert!(make_corrupted_dataset(
            &base_dataset,
            1.5,
            MissingPattern::MCAR,
            0.0,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());

        // Test invalid outlier rate
        assert!(make_corrupted_dataset(
            &base_dataset,
            0.0,
            MissingPattern::MCAR,
            -0.1,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
        assert!(make_corrupted_dataset(
            &base_dataset,
            0.0,
            MissingPattern::MCAR,
            1.5,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
    }

    #[test]
    fn test_missing_patterns() {
        let data = Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f64).collect()).unwrap();

        // Test different missing patterns
        for pattern in [
            MissingPattern::MCAR,
            MissingPattern::MAR,
            MissingPattern::MNAR,
            MissingPattern::Block,
        ] {
            let mut test_data = data.clone();
            let missing_mask = inject_missing_data(&mut test_data, 0.2, pattern, Some(42)).unwrap();

            let missing_count = missing_mask.iter().filter(|&&x| x).count();
            assert!(
                missing_count > 0,
                "Pattern {pattern:?} should create some missing values"
            );
        }
    }

    #[test]
    fn test_outlier_types() {
        let data = Array2::ones((30, 3));

        // Test different outlier types
        for outlier_type in [
            OutlierType::Point,
            OutlierType::Contextual,
            OutlierType::Collective,
        ] {
            let mut test_data = data.clone();
            let outlier_mask =
                inject_outliers(&mut test_data, 0.2, outlier_type, 3.0, Some(42)).unwrap();

            let outlier_count = outlier_mask.iter().filter(|&&x| x).count();
            assert!(
                outlier_count > 0,
                "Outlier type {outlier_type:?} should create some outliers"
            );
        }
    }
}
