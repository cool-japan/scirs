use ndarray::Array2;
use scirs2_cluster::gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions};
use scirs2_cluster::metrics::silhouette_score;

fn main() {
    println!("Gaussian Mixture Models (GMM) Demo");
    println!("{}", "=".repeat(50));

    // Generate synthetic data
    let data = generate_gaussian_mixture_data();
    println!(
        "Generated {} samples from a mixture of Gaussians\n",
        data.shape()[0]
    );

    // Test different covariance types
    let covariance_types = vec![
        CovarianceType::Spherical,
        CovarianceType::Diagonal,
        CovarianceType::Full,
    ];

    for cov_type in covariance_types {
        println!("Testing with covariance type: {:?}", cov_type);

        let options = GMMOptions {
            n_components: 3,
            covariance_type: cov_type,
            max_iter: 100,
            tol: 1e-4,
            n_init: 5,
            init_method: GMMInit::KMeans,
            random_seed: Some(42),
            ..Default::default()
        };

        match gaussian_mixture(data.view(), options) {
            Ok(labels) => {
                // Calculate silhouette score
                match silhouette_score(data.view(), labels.view()) {
                    Ok(score) => println!("  Silhouette score: {:.3}", score),
                    Err(_) => println!("  Silhouette score: N/A"),
                }

                // Print cluster sizes
                let mut cluster_sizes = vec![0; 3];
                for &label in labels.iter() {
                    if label >= 0 && (label as usize) < cluster_sizes.len() {
                        cluster_sizes[label as usize] += 1;
                    }
                }
                println!("  Cluster sizes: {:?}", cluster_sizes);
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
        println!();
    }

    // Test different numbers of components
    println!("\nTesting different numbers of components:");
    println!("{}", "-".repeat(40));

    for n_components in 2..=5 {
        let options = GMMOptions {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            n_init: 3,
            random_seed: Some(42),
            ..Default::default()
        };

        match gaussian_mixture(data.view(), options) {
            Ok(labels) => {
                print!("n_components = {}: ", n_components);

                // Calculate silhouette score
                match silhouette_score(data.view(), labels.view()) {
                    Ok(score) => print!("silhouette = {:.3}, ", score),
                    Err(_) => print!("silhouette = N/A, "),
                }

                // Count actual clusters found
                let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
                println!("actual clusters = {}", unique_labels.len());
            }
            Err(e) => {
                println!("n_components = {}: Error - {}", n_components, e);
            }
        }
    }

    // Compare initialization methods
    println!("\n\nComparing initialization methods:");
    println!("{}", "-".repeat(40));

    let init_methods = vec![(GMMInit::KMeans, "K-means++"), (GMMInit::Random, "Random")];

    for (init_method, name) in init_methods {
        println!("\nInitialization: {}", name);

        let mut scores = Vec::new();

        // Run multiple times to see stability
        for run in 0..5 {
            let options = GMMOptions {
                n_components: 3,
                covariance_type: CovarianceType::Full,
                init_method,
                max_iter: 100,
                n_init: 1, // Single initialization to see variance
                random_seed: Some(run as u64),
                ..Default::default()
            };

            if let Ok(labels) = gaussian_mixture(data.view(), options) {
                if let Ok(score) = silhouette_score(data.view(), labels.view()) {
                    scores.push(score);
                }
            }
        }

        if !scores.is_empty() {
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
            let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            println!("  Mean silhouette: {:.3}", mean_score);
            println!("  Range: [{:.3}, {:.3}]", min_score, max_score);
        }
    }
}

fn generate_gaussian_mixture_data() -> Array2<f64> {
    let mut data = Vec::new();

    // Component 1: centered at (0, 0) with small variance
    for _ in 0..50 {
        let x = rand::random::<f64>() * 1.0 - 0.5;
        let y = rand::random::<f64>() * 1.0 - 0.5;
        data.push(x);
        data.push(y);
    }

    // Component 2: centered at (4, 4) with medium variance
    for _ in 0..40 {
        let x = 4.0 + (rand::random::<f64>() * 2.0 - 1.0);
        let y = 4.0 + (rand::random::<f64>() * 2.0 - 1.0);
        data.push(x);
        data.push(y);
    }

    // Component 3: centered at (0, 4) with larger variance
    for _ in 0..30 {
        let x = 0.0 + (rand::random::<f64>() * 3.0 - 1.5);
        let y = 4.0 + (rand::random::<f64>() * 3.0 - 1.5);
        data.push(x);
        data.push(y);
    }

    Array2::from_shape_vec((120, 2), data).unwrap()
}
