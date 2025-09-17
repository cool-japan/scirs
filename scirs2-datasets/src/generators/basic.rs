//! Basic dataset generators (classification, regression, blobs, etc.)

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Uniform;
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
