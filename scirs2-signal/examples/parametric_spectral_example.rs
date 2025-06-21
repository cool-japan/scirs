use ndarray::Array1;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use rand::{rng, Rng};
use scirs2_signal::parametric::{
    ar_spectrum, arma_spectrum, estimate_ar, estimate_arma, select_ar_order, ARMethod,
    OrderSelection,
};
use scirs2_signal::spectral::periodogram;

fn main() {
    println!("Parametric Spectral Estimation Examples");
    println!("---------------------------------------");

    // Create test signals
    println!("Generating test signals...");
    let signal_sinusoids = generate_sinusoid_signal();
    let signal_ar = generate_ar_signal();

    // Analyze sinusoidal signal with AR methods
    println!("\nAnalyzing sinusoidal signal with different AR methods:");
    analyze_with_ar_methods(&signal_sinusoids, 1024.0);

    // Analyze AR signal with different model orders
    println!("\nAnalyzing AR signal with different model orders:");
    analyze_with_different_orders(&signal_ar, 1024.0);

    // Compare with traditional periodogram
    println!("\nComparing parametric methods with periodogram:");
    compare_with_periodogram(&signal_sinusoids, 1024.0);

    // Demonstrate ARMA model on a mixed signal
    println!("\nDemonstrating ARMA model on mixed signal:");
    demonstrate_arma_model();

    // Demonstrate automatic model order selection
    println!("\nDemonstrating automatic model order selection:");
    demonstrate_order_selection(&signal_sinusoids, 1024.0);

    println!("\nDone! Results saved to CSV files for plotting.");
}

/// Generates a test signal with two sinusoidal components and noise
fn generate_sinusoid_signal() -> Array1<f64> {
    let n_samples = 512;
    let fs = 1024.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a signal with:
    // 1. A 100 Hz sinusoid
    // 2. A 250 Hz sinusoid with lower amplitude
    // 3. White noise

    let f1 = 100.0;
    let f2 = 250.0;

    let sinusoid1 = t.mapv(|ti| (2.0 * PI * f1 * ti).sin());
    let sinusoid2 = t.mapv(|ti| 0.5 * (2.0 * PI * f2 * ti).sin());

    // Add noise
    let noise_level = 0.2;
    let mut rng = rng();
    let noise = Array1::from_iter(
        (0..n_samples).map(|_| noise_level * (2.0 * rng.random_range(0.0..1.0) - 1.0)),
    );

    &sinusoid1 + &sinusoid2 + &noise
}

/// Generates a signal from an AR(4) process
fn generate_ar_signal() -> Array1<f64> {
    let n_samples = 1024;
    let n_warmup = 1000; // Extra samples to ensure stationarity

    // AR(4) coefficients [1, a1, a2, a3, a4]
    let ar_coeffs = [1.0, -2.7607, 3.8106, -2.6535, 0.9238];

    // Define poles close to unit circle to create spectral peaks
    // Generates a spectrum with two sharp peaks

    // Generate the AR signal
    let mut signal = Vec::with_capacity(n_samples + n_warmup);
    let mut rng = rng();

    // Initialize with random values
    for _ in 0..4 {
        signal.push(rng.random_range(-0.1..0.1));
    }

    // Generate AR samples
    for i in 4..(n_samples + n_warmup) {
        let mut sample = rng.random_range(-0.1..0.1);
        for j in 1..=4 {
            sample -= ar_coeffs[j] * signal[i - j];
        }
        signal.push(sample);
    }

    // Discard warmup samples
    let signal = signal[n_warmup..].to_vec();

    Array1::from_vec(signal)
}

/// Analyzes a signal using different AR estimation methods
fn analyze_with_ar_methods(signal: &Array1<f64>, fs: f64) {
    let ar_order = 20; // Higher order to capture peaks well

    // Apply different AR estimation methods
    let methods = [
        (ARMethod::YuleWalker, "Yule-Walker".to_string()),
        (ARMethod::Burg, "Burg".to_string()),
        (ARMethod::Covariance, "Covariance".to_string()),
        (
            ARMethod::ModifiedCovariance,
            "Modified Covariance".to_string(),
        ),
        (ARMethod::LeastSquares, "Least Squares".to_string()),
    ];

    // Frequency axis for spectral estimation
    let nfft = 1024;
    let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);

    // Results for saving to CSV
    let mut psd_results = Vec::new();
    psd_results.push(freqs.clone());

    for (method, method_name) in &methods {
        // Estimate AR parameters
        match estimate_ar(signal, ar_order, *method) {
            Ok((ar_coeffs, reflection_coeffs, variance)) => {
                println!(
                    "  {}: Order {}, Variance: {:.4e}",
                    method_name, ar_order, variance
                );

                // Compute power spectral density
                match ar_spectrum(&ar_coeffs, variance, &freqs, fs) {
                    Ok(psd) => {
                        // Convert to dB for better visualization
                        let psd_db = psd.mapv(|x| 10.0 * (x).log10());
                        psd_results.push(psd_db);

                        // Print some info about reflection coefficients (for Burg method)
                        if *method == ARMethod::Burg || *method == ARMethod::YuleWalker {
                            if let Some(ref k) = reflection_coeffs {
                                let k_max = k.iter().map(|&x| x.abs()).fold(0.0, f64::max);
                                println!("    Max reflection coefficient magnitude: {:.4}", k_max);

                                // Check stability
                                let is_stable = k.iter().all(|&x| x.abs() < 1.0);
                                println!(
                                    "    AR model is {}",
                                    if is_stable { "stable" } else { "unstable" }
                                );
                            }
                        }
                    }
                    Err(e) => println!("  Error computing {} spectrum: {:?}", method_name, e),
                }
            }
            Err(e) => println!("  Error in {} estimation: {:?}", method_name, e),
        }
    }

    // Save results to CSV
    save_psd_to_csv("ar_methods_comparison.csv", &psd_results, &methods);
}

/// Analyzes a signal using different AR model orders
fn analyze_with_different_orders(signal: &Array1<f64>, fs: f64) {
    // Array of different orders to try
    let orders = [2, 4, 8, 16, 32];

    // Frequency axis for spectral estimation
    let nfft = 1024;
    let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);

    // Results for saving to CSV
    let mut psd_results = Vec::new();
    psd_results.push(freqs.clone());

    // Method to use (Burg is generally most robust)
    let method = ARMethod::Burg;

    // True AR order is 4 for our generated signal
    println!("  True model order is 4");

    for &order in &orders {
        // Estimate AR parameters
        match estimate_ar(signal, order, method) {
            Ok((ar_coeffs, _, variance)) => {
                println!("  Order {}: Variance: {:.4e}", order, variance);

                // Print AR coefficients for the true order case
                if order == 4 {
                    println!("  AR(4) coefficients:");
                    for (i, &coeff) in ar_coeffs.iter().enumerate() {
                        println!("    a{} = {:.4}", i, coeff);
                    }
                }

                // Compute power spectral density
                match ar_spectrum(&ar_coeffs, variance, &freqs, fs) {
                    Ok(psd) => {
                        // Convert to dB for better visualization
                        let psd_db = psd.mapv(|x| 10.0 * (x).log10());
                        psd_results.push(psd_db);
                    }
                    Err(e) => println!("  Error computing spectrum for order {}: {:?}", order, e),
                }
            }
            Err(e) => println!("  Error in AR({}) estimation: {:?}", order, e),
        }
    }

    // Save results to CSV
    let method_names: Vec<(ARMethod, String)> = orders
        .iter()
        .map(|&order| (method, format!("AR({})", order)))
        .collect();

    save_psd_to_csv("ar_orders_comparison.csv", &psd_results, &method_names);
}

/// Compares parametric methods with traditional periodogram
fn compare_with_periodogram(signal: &Array1<f64>, fs: f64) {
    // Compute periodogram (non-parametric)
    let (pxx_periodogram, f_periodogram) =
        periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None).unwrap();

    // Convert to dB
    let pxx_db = pxx_periodogram
        .iter()
        .map(|&x| 10.0 * x.log10())
        .collect::<Vec<_>>();

    // Compute AR spectrum with Burg method
    let ar_orders = [4, 20, 60];
    let method = ARMethod::Burg;

    // Frequency axis matching periodogram
    let freqs = Array1::from(f_periodogram.clone());

    // Results for saving to CSV
    let mut psd_results = Vec::new();
    psd_results.push(freqs.clone());
    psd_results.push(Array1::from(pxx_db));

    for &order in &ar_orders {
        // Estimate AR parameters
        match estimate_ar(signal, order, method) {
            Ok((ar_coeffs, _, variance)) => {
                println!("  AR order {}: Variance: {:.4e}", order, variance);

                // Compute power spectral density
                match ar_spectrum(&ar_coeffs, variance, &freqs, fs) {
                    Ok(psd) => {
                        // Convert to dB for better visualization
                        let psd_db = psd.mapv(|x| 10.0 * (x).log10());
                        psd_results.push(psd_db);
                    }
                    Err(e) => println!("  Error computing AR({}) spectrum: {:?}", order, e),
                }
            }
            Err(e) => println!("  Error in AR({}) estimation: {:?}", order, e),
        }
    }

    // Method names for CSV
    let mut method_names = vec![(ARMethod::YuleWalker, "Periodogram".to_string())];
    for &order in &ar_orders {
        method_names.push((method, format!("AR({})", order)));
    }

    // Save results to CSV
    save_psd_to_csv("parametric_vs_periodogram.csv", &psd_results, &method_names);
}

/// Demonstrates ARMA model on a mixed signal
fn demonstrate_arma_model() {
    // Generate an ARMA(2,2) process
    let n_samples = 1024;
    let n_warmup = 1000;
    let fs = 1024.0;

    // AR parameters: [1, -1.5, 0.8]
    // MA parameters: [1, 0.7, -0.4]

    let ar_coeffs = [1.0, -1.5, 0.8];
    let ma_coeffs = [1.0, 0.7, -0.4];

    // Generate the ARMA signal
    let mut signal = Vec::with_capacity(n_samples + n_warmup);
    let mut noise_history = Vec::with_capacity(n_samples + n_warmup);
    let mut rng = rng();

    // Initialize with random values
    for _ in 0..2 {
        signal.push(rng.random_range(-0.1..0.1));
        noise_history.push(rng.random_range(-0.1..0.1));
    }

    // Generate ARMA samples
    for i in 2..(n_samples + n_warmup) {
        // Generate white noise
        let noise = rng.random_range(-0.5..0.5);
        noise_history.push(noise);

        // AR component
        let mut sample = noise;
        for j in 1..=2 {
            sample -= ar_coeffs[j] * signal[i - j];
        }

        // MA component
        for j in 1..=2 {
            sample += ma_coeffs[j] * noise_history[i - j];
        }

        signal.push(sample);
    }

    // Discard warmup samples
    let signal = signal[n_warmup..].to_vec();
    let signal = Array1::from_vec(signal);

    // Estimate ARMA parameters
    let ar_order = 2;
    let ma_order = 2;

    match estimate_arma(&signal, ar_order, ma_order) {
        Ok((ar_coeffs, ma_coeffs, variance)) => {
            println!("  Estimated ARMA({},{}) parameters:", ar_order, ma_order);
            println!("    AR coefficients:");
            for (i, &coeff) in ar_coeffs.iter().enumerate() {
                println!("      a{} = {:.4}", i, coeff);
            }

            println!("    MA coefficients:");
            for (i, &coeff) in ma_coeffs.iter().enumerate() {
                println!("      b{} = {:.4}", i, coeff);
            }

            println!("    Variance: {:.4e}", variance);

            // Compute ARMA spectrum
            let nfft = 1024;
            let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);

            // Compare with AR-only model
            let (ar_only_coeffs, _, ar_only_variance) =
                estimate_ar(&signal, ar_order, ARMethod::Burg).unwrap();

            match arma_spectrum(&ar_coeffs, &ma_coeffs, variance, &freqs, fs) {
                Ok(arma_psd) => {
                    // Convert to dB
                    let arma_psd_db = arma_psd.mapv(|x| 10.0 * (x).log10());

                    // Compute AR-only spectrum
                    let ar_psd =
                        ar_spectrum(&ar_only_coeffs, ar_only_variance, &freqs, fs).unwrap();
                    let ar_psd_db = ar_psd.mapv(|x| 10.0 * (x).log10());

                    // Compute periodogram for comparison
                    let (pxx_periodogram, _f_periodogram) =
                        periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None)
                            .unwrap();
                    let pxx_db: Vec<f64> = pxx_periodogram[..(nfft / 2 + 1)]
                        .iter()
                        .map(|x| 10.0 * (x).log10())
                        .collect();

                    // Save results to CSV
                    let psd_results = vec![
                        freqs.clone(),
                        Array1::from(pxx_db.to_owned()),
                        ar_psd_db,
                        arma_psd_db,
                    ];

                    let method_names = vec![
                        (ARMethod::YuleWalker, "Periodogram".to_string()),
                        (ARMethod::Burg, "AR(2)".to_string()),
                        (ARMethod::Burg, "ARMA(2,2)".to_string()),
                    ];

                    save_psd_to_csv("arma_comparison.csv", &psd_results, &method_names);
                }
                Err(e) => println!("  Error computing ARMA spectrum: {:?}", e),
            }
        }
        Err(e) => println!("  Error in ARMA estimation: {:?}", e),
    }
}

/// Demonstrates automatic model order selection
fn demonstrate_order_selection(signal: &Array1<f64>, fs: f64) {
    // Maximum order to consider
    let max_order = 50;

    // Different criteria
    let criteria = [
        (OrderSelection::AIC, "AIC"),
        (OrderSelection::BIC, "BIC"),
        (OrderSelection::FPE, "FPE"),
        (OrderSelection::MDL, "MDL"),
        (OrderSelection::AICc, "AICc"),
    ];

    // Use Burg method for estimation
    let method = ARMethod::Burg;

    // Results for each criterion
    for (criterion, name) in criteria {
        match select_ar_order(signal, max_order, criterion, method) {
            Ok((opt_order, criterion_values)) => {
                println!("  {}: Optimal order = {}", name, opt_order);

                // Save criterion values to CSV
                let mut file = File::create(format!("order_selection_{}.csv", name))
                    .expect("Failed to create order selection CSV file");

                writeln!(file, "order,{}_value", name).expect("Failed to write header");
                for (order, &value) in criterion_values.iter().enumerate() {
                    writeln!(file, "{},{}", order, value).expect("Failed to write data");
                }

                // Compute spectrum with the selected order
                if opt_order > 0 {
                    let (ar_coeffs, _, variance) = estimate_ar(signal, opt_order, method).unwrap();

                    // Spectrum using optimal order
                    let nfft = 1024;
                    let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);
                    let psd = ar_spectrum(&ar_coeffs, variance, &freqs, fs).unwrap();
                    let psd_db = psd.mapv(|x| 10.0 * (x).log10());

                    // Compute periodogram for comparison
                    let (pxx_periodogram, _f_periodogram) =
                        periodogram(signal.as_slice().unwrap(), Some(fs), None, None, None, None)
                            .unwrap();
                    let pxx_db: Vec<f64> = pxx_periodogram[..(nfft / 2 + 1)]
                        .iter()
                        .map(|x| 10.0 * (x).log10())
                        .collect();

                    // Save to CSV
                    let psd_results = vec![freqs.clone(), Array1::from(pxx_db.to_owned()), psd_db];

                    let method_names = vec![
                        (ARMethod::YuleWalker, "Periodogram".to_string()),
                        (method, format!("AR({}) - {}", opt_order, name)),
                    ];

                    save_psd_to_csv(
                        &format!("optimal_order_{}.csv", name),
                        &psd_results,
                        &method_names,
                    );
                }
            }
            Err(e) => println!("  Error in {} order selection: {:?}", name, e),
        }
    }
}

/// Helper function to save PSD results to CSV
fn save_psd_to_csv(
    filename: &str,
    psd_arrays: &[Array1<f64>],
    method_names: &[(ARMethod, String)],
) {
    let mut file =
        File::create(filename).unwrap_or_else(|_| panic!("Failed to create {}", filename));

    // Write header
    write!(file, "frequency").expect("Failed to write header");
    for (_, method_name) in method_names.iter().take(psd_arrays.len() - 1) {
        write!(file, ",{}", method_name).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write data
    let freqs = &psd_arrays[0];
    for i in 0..freqs.len() {
        write!(file, "{}", freqs[i]).expect("Failed to write data");

        for psd_array in psd_arrays.iter().skip(1) {
            write!(file, ",{}", psd_array[i]).expect("Failed to write data");
        }

        writeln!(file).expect("Failed to write data");
    }
}
