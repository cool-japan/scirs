//! Advanced manifold generators for dimensionality reduction datasets

use super::config::{ManifoldConfig, ManifoldType};
use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use scirs2_core::rng;
use std::f64::consts::PI;

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
