use ndarray::{s, Array1, Array2};
use rand::rng;
use rand_distr::{Distribution, Normal};
use scirs2_signal::{bss, SignalError, SignalResult};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> SignalResult<()> {
    println!("Blind Source Separation Examples");

    // Example 1: Independent Component Analysis (ICA)
    ica_example()?;

    // Example 2: Non-negative Matrix Factorization (NMF)
    nmf_example()?;

    // Example 3: Principal Component Analysis (PCA)
    pca_example()?;

    // Example 4: Sparse Component Analysis (SCA)
    sparse_component_analysis_example()?;

    // Example 5: Comparison of BSS methods
    compare_bss_methods()?;

    Ok(())
}

/// Generate source signals for testing
fn generate_test_signals(n_samples: usize) -> Array2<f64> {
    let mut sources = Array2::zeros((4, n_samples));
    let t = Array1::linspace(0.0, 10.0, n_samples);

    // Source 1: Sine wave
    for i in 0..n_samples {
        sources[[0, i]] = (t[i] * 2.0 * PI * 0.5).sin();
    }

    // Source 2: Square wave
    for i in 0..n_samples {
        let phase = (t[i] * 2.0 * PI * 0.2) % (2.0 * PI);
        sources[[1, i]] = if phase < PI { 1.0 } else { -1.0 };
    }

    // Source 3: Sawtooth wave
    for i in 0..n_samples {
        let phase = (t[i] * 2.0 * PI * 0.3) % (2.0 * PI);
        sources[[2, i]] = phase / PI - 1.0;
    }

    // Source 4: Random spikes (sparse signal)
    let _rng = rng();
    let threshold = 0.95;
    for i in 0..n_samples {
        if rand::random::<f64>() > threshold {
            sources[[3, i]] = 5.0 * (rand::random::<f64>() * 2.0 - 1.0);
        }
    }

    sources
}

/// Create a random mixing matrix
fn generate_mixing_matrix(n_sources: usize, n_mixtures: usize) -> Array2<f64> {
    let mut mixing = Array2::zeros((n_mixtures, n_sources));
    let mut rng = rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_mixtures {
        for j in 0..n_sources {
            mixing[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Normalize each row
    for i in 0..n_mixtures {
        let row_norm = mixing
            .slice(s![i, ..])
            .mapv(|x: f64| x.powi(2))
            .sum()
            .sqrt();
        if row_norm > 0.0 {
            for j in 0..n_sources {
                mixing[[i, j]] /= row_norm;
            }
        }
    }

    mixing
}

/// Add noise to signals
fn add_noise(signals: &Array2<f64>, noise_level: f64) -> Array2<f64> {
    let (n_signals, n_samples) = signals.dim();
    let mut noisy = signals.clone();
    let mut rng = rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    for i in 0..n_signals {
        for j in 0..n_samples {
            noisy[[i, j]] += normal.sample(&mut rng);
        }
    }

    noisy
}

/// Calculate correlation between original and recovered sources
fn calculate_correlations(original: &Array2<f64>, recovered: &Array2<f64>) -> Array2<f64> {
    let (n_orig, n_samples) = original.dim();
    let (n_rec, _) = recovered.dim();
    let n = n_orig.min(n_rec);

    let mut correlations = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            // Calculate correlation coefficient
            let orig = original.slice(s![i, ..]);
            let rec = recovered.slice(s![j, ..]);

            let orig_mean = orig.mean().unwrap();
            let rec_mean = rec.mean().unwrap();

            let mut numerator = 0.0;
            let mut orig_var = 0.0;
            let mut rec_var = 0.0;

            for k in 0..n_samples {
                let orig_centered = orig[k] - orig_mean;
                let rec_centered = rec[k] - rec_mean;

                numerator += orig_centered * rec_centered;
                orig_var += orig_centered * orig_centered;
                rec_var += rec_centered * rec_centered;
            }

            if orig_var > 0.0 && rec_var > 0.0 {
                correlations[[i, j]] = numerator / (orig_var.sqrt() * rec_var.sqrt());
            }
        }
    }

    correlations
}

/// Find best source matching and calculate overall recovery quality
fn calculate_recovery_quality(original: &Array2<f64>, recovered: &Array2<f64>) -> f64 {
    let correlations = calculate_correlations(original, recovered);
    let (n_orig, n_rec) = correlations.dim();

    // For each original source, find the best matching recovered source
    let mut total_correlation = 0.0;
    let mut used_indices = vec![false; n_rec];

    for i in 0..n_orig {
        let mut best_corr = 0.0;
        let mut best_idx = 0;

        for j in 0..n_rec {
            if !used_indices[j] {
                let corr = correlations[[i, j]].abs();
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = j;
                }
            }
        }

        used_indices[best_idx] = true;
        total_correlation += best_corr;
    }

    total_correlation / n_orig as f64
}

/// Export signals to CSV for visualization
fn export_to_csv(file_name: &str, signals: &[(&str, &Array1<f64>)]) -> SignalResult<()> {
    let mut file = File::create(file_name).map_err(|e| SignalError::Compute(e.to_string()))?;

    // Write header
    let header = signals
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<String>>()
        .join(",");
    writeln!(file, "{}", header).map_err(|e| SignalError::Compute(e.to_string()))?;

    // Find common signal length
    let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();

    // Write data
    for i in 0..min_len {
        let line = signals
            .iter()
            .map(|(_, data)| data[i].to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line).map_err(|e| SignalError::Compute(e.to_string()))?;
    }

    println!("Data exported to {}", file_name);
    Ok(())
}

/// ICA example
fn ica_example() -> SignalResult<()> {
    println!("Independent Component Analysis (ICA) Example");

    // Generate source signals
    let n_samples = 1000;
    let sources = generate_test_signals(n_samples);
    let n_sources = sources.dim().0;

    // Create mixing matrix
    let n_mixtures = 4;
    let mixing = generate_mixing_matrix(n_sources, n_mixtures);

    // Mix signals
    let mixed = mixing.dot(&sources);

    // Add some noise
    let noisy_mixed = add_noise(&mixed, 0.05);

    // ICA configuration
    let config = bss::BssConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        apply_whitening: true,
        dimension_reduction: false,
        target_dimension: None,
        variance_threshold: 0.95,
        learning_rate: 0.1,
        non_negative: false,
        regularization: 1e-4,
        random_seed: Some(42),
        use_fixed_point: true,
        parallel: false,
    };

    // Apply FastICA
    let (ica_sources, _ica_mixing) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::FastICA,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;

    // Calculate recovery quality
    let ica_quality = calculate_recovery_quality(&sources, &ica_sources);
    println!("FastICA recovery quality: {:.4}", ica_quality);

    // Apply Infomax ICA
    let (infomax_sources, _infomax_mixing) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::Infomax,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;

    // Calculate recovery quality
    let infomax_quality = calculate_recovery_quality(&sources, &infomax_sources);
    println!("Infomax ICA recovery quality: {:.4}", infomax_quality);

    // Apply JADE ICA
    let (jade_sources, _jade_mixing) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::JADE,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;

    // Calculate recovery quality
    let jade_quality = calculate_recovery_quality(&sources, &jade_sources);
    println!("JADE ICA recovery quality: {:.4}", jade_quality);

    // Export source signals for visualization
    export_to_csv(
        "ica_sources.csv",
        &[
            ("Source1", &sources.slice(s![0, ..]).to_owned()),
            ("Source2", &sources.slice(s![1, ..]).to_owned()),
            ("Source3", &sources.slice(s![2, ..]).to_owned()),
            ("Source4", &sources.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export mixed signals
    export_to_csv(
        "ica_mixed.csv",
        &[
            ("Mixed1", &mixed.slice(s![0, ..]).to_owned()),
            ("Mixed2", &mixed.slice(s![1, ..]).to_owned()),
            ("Mixed3", &mixed.slice(s![2, ..]).to_owned()),
            ("Mixed4", &mixed.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export recovered signals
    export_to_csv(
        "ica_recovered.csv",
        &[
            ("Recovered1", &ica_sources.slice(s![0, ..]).to_owned()),
            ("Recovered2", &ica_sources.slice(s![1, ..]).to_owned()),
            ("Recovered3", &ica_sources.slice(s![2, ..]).to_owned()),
            ("Recovered4", &ica_sources.slice(s![3, ..]).to_owned()),
        ],
    )?;

    Ok(())
}

/// Generate non-negative source signals for NMF testing
fn generate_non_negative_signals(n_samples: usize) -> Array2<f64> {
    let mut sources = Array2::zeros((3, n_samples));
    let t = Array1::linspace(0.0, 10.0, n_samples);

    // Source 1: Half-rectified sine wave
    for i in 0..n_samples {
        sources[[0, i]] = (t[i] * 2.0 * PI * 0.5).sin().max(0.0);
    }

    // Source 2: Gaussian pulses
    for i in 0..n_samples {
        let x = t[i] - 5.0;
        sources[[1, i]] = (-x * x / 2.0).exp() + (-((x - 3.0) * (x - 3.0)) / 1.0).exp() * 0.5;
    }

    // Source 3: Exponential decay
    for i in 0..n_samples {
        let x = t[i];
        sources[[2, i]] = (-x / 2.0).exp() + (-(x - 7.0) / 1.0).exp() * (x >= 7.0) as i32 as f64;
    }

    sources
}

/// NMF example
fn nmf_example() -> SignalResult<()> {
    println!("Non-negative Matrix Factorization (NMF) Example");

    // Generate non-negative source signals
    let n_samples = 1000;
    let sources = generate_non_negative_signals(n_samples);
    let n_sources = sources.dim().0;

    // Create non-negative mixing matrix
    let n_mixtures = 4;
    let mut mixing = Array2::zeros((n_mixtures, n_sources));

    for i in 0..n_mixtures {
        for j in 0..n_sources {
            mixing[[i, j]] = rand::random::<f64>();
        }
    }

    // Mix signals
    let mixed = mixing.dot(&sources);

    // Add some noise and ensure non-negativity
    let mut noisy_mixed = add_noise(&mixed, 0.02);
    noisy_mixed = noisy_mixed.mapv(|x| x.max(0.0));

    // NMF configuration
    let config = bss::BssConfig {
        max_iterations: 200,
        convergence_threshold: 1e-6,
        apply_whitening: false,
        dimension_reduction: false,
        target_dimension: None,
        variance_threshold: 0.95,
        learning_rate: 0.1,
        non_negative: true,
        regularization: 1e-4,
        random_seed: Some(42),
        use_fixed_point: true,
        parallel: false,
    };

    // Apply NMF
    let (nmf_sources, _nmf_mixing) = bss::nmf(&noisy_mixed, n_sources, &config)?;

    // Calculate recovery quality
    let nmf_quality = calculate_recovery_quality(&sources, &nmf_sources);
    println!("NMF recovery quality: {:.4}", nmf_quality);

    // Export source signals for visualization
    export_to_csv(
        "nmf_sources.csv",
        &[
            ("Source1", &sources.slice(s![0, ..]).to_owned()),
            ("Source2", &sources.slice(s![1, ..]).to_owned()),
            ("Source3", &sources.slice(s![2, ..]).to_owned()),
        ],
    )?;

    // Export mixed signals
    export_to_csv(
        "nmf_mixed.csv",
        &[
            ("Mixed1", &mixed.slice(s![0, ..]).to_owned()),
            ("Mixed2", &mixed.slice(s![1, ..]).to_owned()),
            ("Mixed3", &mixed.slice(s![2, ..]).to_owned()),
            ("Mixed4", &mixed.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export recovered signals
    export_to_csv(
        "nmf_recovered.csv",
        &[
            ("Recovered1", &nmf_sources.slice(s![0, ..]).to_owned()),
            ("Recovered2", &nmf_sources.slice(s![1, ..]).to_owned()),
            ("Recovered3", &nmf_sources.slice(s![2, ..]).to_owned()),
        ],
    )?;

    Ok(())
}

/// PCA example
fn pca_example() -> SignalResult<()> {
    println!("Principal Component Analysis (PCA) Example");

    // Generate source signals
    let n_samples = 1000;
    let sources = generate_test_signals(n_samples);
    let n_sources = sources.dim().0;

    // Create mixing matrix
    let n_mixtures = 6; // More mixtures than sources
    let mixing = generate_mixing_matrix(n_sources, n_mixtures);

    // Mix signals
    let mixed = mixing.dot(&sources);

    // Add some noise
    let noisy_mixed = add_noise(&mixed, 0.1);

    // PCA configuration
    let config = bss::BssConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        apply_whitening: false,
        dimension_reduction: true,
        target_dimension: Some(n_sources),
        variance_threshold: 0.95,
        learning_rate: 0.1,
        non_negative: false,
        regularization: 1e-4,
        random_seed: Some(42),
        use_fixed_point: true,
        parallel: false,
    };

    // Apply PCA
    let (pca_sources, _pca_mixing) = bss::pca(&noisy_mixed, &config)?;

    // Calculate recovery quality
    let pca_quality = calculate_recovery_quality(&sources, &pca_sources);
    println!("PCA recovery quality: {:.4}", pca_quality);

    // Calculate explained variance
    let mut variances = Vec::with_capacity(pca_sources.dim().0);
    for i in 0..pca_sources.dim().0 {
        let component = pca_sources.slice(s![i, ..]);
        let mean = component.mean().unwrap();
        let var = component.mapv(|x| (x - mean).powi(2)).sum() / (n_samples as f64 - 1.0);
        variances.push(var);
    }

    // Sort variances in descending order
    variances.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Calculate cumulative explained variance
    let total_var = variances.iter().sum::<f64>();
    let mut cum_var = 0.0;

    println!("PCA explained variance:");
    for (i, &var) in variances.iter().enumerate() {
        cum_var += var;
        println!(
            "Component {}: {:.4}% (cumulative: {:.4}%)",
            i + 1,
            var / total_var * 100.0,
            cum_var / total_var * 100.0
        );
    }

    // Export source signals for visualization
    export_to_csv(
        "pca_sources.csv",
        &[
            ("Source1", &sources.slice(s![0, ..]).to_owned()),
            ("Source2", &sources.slice(s![1, ..]).to_owned()),
            ("Source3", &sources.slice(s![2, ..]).to_owned()),
            ("Source4", &sources.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export mixed signals (first 4)
    export_to_csv(
        "pca_mixed.csv",
        &[
            ("Mixed1", &mixed.slice(s![0, ..]).to_owned()),
            ("Mixed2", &mixed.slice(s![1, ..]).to_owned()),
            ("Mixed3", &mixed.slice(s![2, ..]).to_owned()),
            ("Mixed4", &mixed.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export recovered signals
    export_to_csv(
        "pca_recovered.csv",
        &[
            ("Recovered1", &pca_sources.slice(s![0, ..]).to_owned()),
            ("Recovered2", &pca_sources.slice(s![1, ..]).to_owned()),
            ("Recovered3", &pca_sources.slice(s![2, ..]).to_owned()),
            ("Recovered4", &pca_sources.slice(s![3, ..]).to_owned()),
        ],
    )?;

    Ok(())
}

/// Generate sparse source signals for SCA testing
fn generate_sparse_signals(n_samples: usize) -> Array2<f64> {
    let mut sources = Array2::zeros((3, n_samples));
    let t = Array1::linspace(0.0, 10.0, n_samples);

    // Source 1: Sparse spikes
    let _rng = rng();
    let threshold = 0.98;
    for i in 0..n_samples {
        if rand::random::<f64>() > threshold {
            sources[[0, i]] = 3.0 * (rand::random::<f64>() * 2.0 - 1.0);
        }
    }

    // Source 2: Sparse square wave
    for i in 0..n_samples {
        let phase = (t[i] * 2.0 * PI * 0.2) % (2.0 * PI);
        if phase < PI / 4.0 || (phase > PI && phase < PI + PI / 4.0) {
            sources[[1, i]] = 1.0;
        }
    }

    // Source 3: Wavelet-like
    for i in 0..n_samples {
        let x = (t[i] - 5.0) * 2.0;
        if x.abs() < 3.0 {
            sources[[2, i]] = (-x * x / 2.0).exp() * (2.0 * PI * x).cos();
        }
    }

    sources
}

/// Sparse Component Analysis example
fn sparse_component_analysis_example() -> SignalResult<()> {
    println!("Sparse Component Analysis (SCA) Example");

    // Generate sparse source signals
    let n_samples = 1000;
    let sources = generate_sparse_signals(n_samples);
    let n_sources = sources.dim().0;

    // Create mixing matrix
    let n_mixtures = 4;
    let mixing = generate_mixing_matrix(n_sources, n_mixtures);

    // Mix signals
    let mixed = mixing.dot(&sources);

    // Add some noise
    let noisy_mixed = add_noise(&mixed, 0.05);

    // SCA configuration
    let config = bss::BssConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        apply_whitening: true,
        dimension_reduction: false,
        target_dimension: None,
        variance_threshold: 0.95,
        learning_rate: 0.1,
        non_negative: false,
        regularization: 1e-4,
        random_seed: Some(42),
        use_fixed_point: true,
        parallel: false,
    };

    // Apply Sparse Component Analysis
    let (sca_sources, _sca_mixing) = bss::sparse_component_analysis(
        &noisy_mixed,
        n_sources,
        0.1, // Sparsity parameter
        &config,
    )?;

    // Calculate recovery quality
    let sca_quality = calculate_recovery_quality(&sources, &sca_sources);
    println!("SCA recovery quality: {:.4}", sca_quality);

    // Calculate sparsity (measured as percentage of near-zero elements)
    let epsilon = 1e-3;
    let mut sparsity_original = 0;
    let mut sparsity_recovered = 0;

    for i in 0..n_sources {
        for j in 0..n_samples {
            if sources[[i, j]].abs() < epsilon {
                sparsity_original += 1;
            }
            if sca_sources[[i, j]].abs() < epsilon {
                sparsity_recovered += 1;
            }
        }
    }

    let sparsity_orig_pct = sparsity_original as f64 / (n_sources * n_samples) as f64 * 100.0;
    let sparsity_rec_pct = sparsity_recovered as f64 / (n_sources * n_samples) as f64 * 100.0;

    println!("Original signals sparsity: {:.2}%", sparsity_orig_pct);
    println!("Recovered signals sparsity: {:.2}%", sparsity_rec_pct);

    // Export source signals for visualization
    export_to_csv(
        "sca_sources.csv",
        &[
            ("Source1", &sources.slice(s![0, ..]).to_owned()),
            ("Source2", &sources.slice(s![1, ..]).to_owned()),
            ("Source3", &sources.slice(s![2, ..]).to_owned()),
        ],
    )?;

    // Export mixed signals
    export_to_csv(
        "sca_mixed.csv",
        &[
            ("Mixed1", &mixed.slice(s![0, ..]).to_owned()),
            ("Mixed2", &mixed.slice(s![1, ..]).to_owned()),
            ("Mixed3", &mixed.slice(s![2, ..]).to_owned()),
            ("Mixed4", &mixed.slice(s![3, ..]).to_owned()),
        ],
    )?;

    // Export recovered signals
    export_to_csv(
        "sca_recovered.csv",
        &[
            ("Recovered1", &sca_sources.slice(s![0, ..]).to_owned()),
            ("Recovered2", &sca_sources.slice(s![1, ..]).to_owned()),
            ("Recovered3", &sca_sources.slice(s![2, ..]).to_owned()),
        ],
    )?;

    Ok(())
}

/// Comparison of BSS methods
fn compare_bss_methods() -> SignalResult<()> {
    println!("Comparison of BSS Methods");

    // Generate source signals
    let n_samples = 1000;
    let sources = generate_test_signals(n_samples);
    let n_sources = sources.dim().0;

    // Create mixing matrix
    let n_mixtures = 4;
    let mixing = generate_mixing_matrix(n_sources, n_mixtures);

    // Mix signals
    let mixed = mixing.dot(&sources);

    // Add some noise
    let noisy_mixed = add_noise(&mixed, 0.05);

    // Common configuration
    let config = bss::BssConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        apply_whitening: true,
        dimension_reduction: false,
        target_dimension: None,
        variance_threshold: 0.95,
        learning_rate: 0.1,
        non_negative: false,
        regularization: 1e-4,
        random_seed: Some(42),
        use_fixed_point: true,
        parallel: false,
    };

    // Apply PCA
    let (pca_sources, _) = bss::pca(&noisy_mixed, &config)?;
    let pca_quality = calculate_recovery_quality(&sources, &pca_sources);

    // Apply FastICA
    let (fastica_sources, _) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::FastICA,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;
    let fastica_quality = calculate_recovery_quality(&sources, &fastica_sources);

    // Apply Infomax ICA
    let (infomax_sources, _) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::Infomax,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;
    let infomax_quality = calculate_recovery_quality(&sources, &infomax_sources);

    // Apply JADE ICA
    let (jade_sources, _) = bss::ica(
        &noisy_mixed,
        Some(n_sources),
        bss::IcaMethod::JADE,
        bss::NonlinearityFunction::Tanh,
        &config,
    )?;
    let jade_quality = calculate_recovery_quality(&sources, &jade_sources);

    // Apply Joint Diagonalization
    let (jd_sources, _) = bss::joint_diagonalization(&noisy_mixed, n_sources, &config)?;
    let jd_quality = calculate_recovery_quality(&sources, &jd_sources);

    // Print comparison
    println!("Method comparison (recovery quality):");
    println!("PCA:             {:.4}", pca_quality);
    println!("FastICA:         {:.4}", fastica_quality);
    println!("Infomax ICA:     {:.4}", infomax_quality);
    println!("JADE ICA:        {:.4}", jade_quality);
    println!("Joint Diag:      {:.4}", jd_quality);

    // Export comparison results
    let mut file =
        File::create("bss_comparison.csv").map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "Method,Quality").map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "PCA,{}", pca_quality).map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "FastICA,{}", fastica_quality)
        .map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "InfomaxICA,{}", infomax_quality)
        .map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "JADEICA,{}", jade_quality).map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(file, "JointDiag,{}", jd_quality).map_err(|e| SignalError::Compute(e.to_string()))?;

    println!("Comparison data exported to bss_comparison.csv");

    Ok(())
}
