use ndarray::{Array1, Array2};
// use plotters::prelude::*;
use num_complex::Complex64;
use rand::{rng, Rng};
use rand_distr::{Distribution, Normal};
use scirs2_signal::{interpolate, SignalError, SignalResult};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> SignalResult<()> {
    println!("Signal Interpolation Examples");

    // Example 1: Interpolate missing values in a simple signal
    interpolate_simple_signal()?;

    // Example 2: Compare different interpolation methods
    compare_interpolation_methods()?;

    // Example 3: Interpolate bandlimited signal
    interpolate_bandlimited_signal()?;

    // Example 4: Interpolate 2D data (image)
    interpolate_2d_data()?;

    // Example 5: Automatic method selection
    auto_interpolation_example()?;

    Ok(())
}

/// Generate a sine wave signal with some missing values
fn generate_test_signal(n_samples: usize, missing_rate: f64) -> Array1<f64> {
    let mut signal = Array1::zeros(n_samples);
    let x = Array1::linspace(0.0, 10.0, n_samples);

    // Generate sine wave
    for i in 0..n_samples {
        signal[i] = (x[i] * PI / 2.0).sin() + 0.5 * (x[i] * PI).cos();
    }

    // Randomly set some values as missing (NaN)
    let mut rng = rng();
    for i in 0..n_samples {
        if rng.random::<f64>() < missing_rate {
            signal[i] = f64::NAN;
        }
    }

    signal
}

/// Export signal data to CSV for visualization
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
            .map(|(_, data)| {
                if data[i].is_nan() {
                    "NaN".to_string()
                } else {
                    data[i].to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line).map_err(|e| SignalError::Compute(e.to_string()))?;
    }

    println!("Data exported to {}", file_name);
    Ok(())
}

/// Export 2D data to CSV for visualization
fn export_2d_to_csv(file_name: &str, data: &Array2<f64>) -> SignalResult<()> {
    let mut file = File::create(file_name).map_err(|e| SignalError::Compute(e.to_string()))?;
    let (n_rows, n_cols) = data.dim();

    for i in 0..n_rows {
        let line = (0..n_cols)
            .map(|j| {
                if data[[i, j]].is_nan() {
                    "NaN".to_string()
                } else {
                    data[[i, j]].to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line).map_err(|e| SignalError::Compute(e.to_string()))?;
    }

    println!("2D data exported to {}", file_name);
    Ok(())
}

/// Example of interpolating a simple signal with missing values
fn interpolate_simple_signal() -> SignalResult<()> {
    println!("Interpolating a Simple Signal");

    // Generate a test signal with 20% missing values
    let n_samples = 100;
    let missing_rate = 0.2;
    let signal = generate_test_signal(n_samples, missing_rate);

    // Create a full reference signal without missing values
    let mut reference = Array1::zeros(n_samples);
    let x = Array1::linspace(0.0, 10.0, n_samples);
    for i in 0..n_samples {
        reference[i] = (x[i] * PI / 2.0).sin() + 0.5 * (x[i] * PI).cos();
    }

    // Configure interpolation
    let config = interpolate::InterpolationConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        regularization: 1e-6,
        window_size: 10,
        extrapolate: true,
        monotonic: false,
        smoothing: false,
        smoothing_factor: 0.1,
        frequency_constraint: true,
        cutoff_frequency: 0.3,
    };

    // Apply linear interpolation
    let linear_result = interpolate::linear_interpolate(&signal)?;

    // Apply cubic spline interpolation
    let spline_result = interpolate::cubic_spline_interpolate(&signal, &config)?;

    // Apply spectral interpolation
    let spectral_result = interpolate::spectral_interpolate(&signal, &config)?;

    // Calculate error metrics
    let mut linear_sse = 0.0;
    let mut spline_sse = 0.0;
    let mut spectral_sse = 0.0;
    let mut count = 0;

    for i in 0..n_samples {
        if signal[i].is_nan() {
            let linear_err = linear_result[i] - reference[i];
            let spline_err = spline_result[i] - reference[i];
            let spectral_err = spectral_result[i] - reference[i];

            linear_sse += linear_err * linear_err;
            spline_sse += spline_err * spline_err;
            spectral_sse += spectral_err * spectral_err;
            count += 1;
        }
    }

    let linear_mse = linear_sse / count as f64;
    let spline_mse = spline_sse / count as f64;
    let spectral_mse = spectral_sse / count as f64;

    println!("Mean squared error (missing values only):");
    println!("  Linear: {:.6}", linear_mse);
    println!("  Cubic Spline: {:.6}", spline_mse);
    println!("  Spectral: {:.6}", spectral_mse);

    // Export data for plotting
    export_to_csv(
        "interpolation_simple.csv",
        &[
            ("Reference", &reference),
            ("Missing", &signal),
            ("Linear", &linear_result),
            ("CubicSpline", &spline_result),
            ("Spectral", &spectral_result),
        ],
    )?;

    Ok(())
}

/// Example of comparing different interpolation methods
fn compare_interpolation_methods() -> SignalResult<()> {
    println!("Comparing Interpolation Methods");

    // Generate a more complex test signal
    let n_samples = 200;

    // Create a complex signal with multiple features
    let mut reference = Array1::zeros(n_samples);
    let x = Array1::linspace(0.0, 10.0, n_samples);

    for i in 0..n_samples {
        // Multi-component signal with different frequencies
        reference[i] = (x[i] as f64).sin()
            + 0.5 * ((x[i] * 3.0) as f64).sin()
            + 0.2 * ((x[i] * 7.0) as f64).sin();

        // Add some discontinuities
        if x[i] > 3.0 && x[i] < 4.0 {
            reference[i] += 1.0;
        }
        if x[i] > 7.0 && x[i] < 8.0 {
            reference[i] -= 1.0;
        }
    }

    // Create a signal with missing values in specific regions
    let mut signal = reference.clone();

    // Remove a chunk of data
    for i in 50..70 {
        signal[i] = f64::NAN;
    }

    // Remove scattered points
    for i in 100..180 {
        if i % 5 == 0 {
            signal[i] = f64::NAN;
        }
    }

    // Configure interpolation
    let config = interpolate::InterpolationConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        regularization: 1e-6,
        window_size: 10,
        extrapolate: true,
        monotonic: false,
        smoothing: false,
        smoothing_factor: 0.1,
        frequency_constraint: true,
        cutoff_frequency: 0.3,
    };

    // Apply different interpolation methods
    let methods = [
        ("Linear", interpolate::InterpolationMethod::Linear),
        ("CubicSpline", interpolate::InterpolationMethod::CubicSpline),
        (
            "CubicHermite",
            interpolate::InterpolationMethod::CubicHermite,
        ),
        (
            "MinimumEnergy",
            interpolate::InterpolationMethod::MinimumEnergy,
        ),
        ("Sinc", interpolate::InterpolationMethod::Sinc),
        ("Spectral", interpolate::InterpolationMethod::Spectral),
        (
            "NearestNeighbor",
            interpolate::InterpolationMethod::NearestNeighbor,
        ),
    ];

    let mut results = Vec::new();
    let mut mse_values = Vec::new();

    for &(name, method) in &methods {
        let result = interpolate::interpolate(&signal, method, &config)?;

        // Calculate error metrics
        let mut sse = 0.0;
        let mut count = 0;

        for i in 0..n_samples {
            if signal[i].is_nan() {
                let err = result[i] - reference[i];
                sse += err * err;
                count += 1;
            }
        }

        let mse = sse / count as f64;
        println!("  {}: MSE = {:.6}", name, mse);

        results.push((name, result.clone()));
        mse_values.push((name, mse));
    }

    // Export data for plotting
    let mut export_data = vec![("Reference", &reference), ("Missing", &signal)];
    for (name, result) in &results {
        export_data.push((*name, result));
    }

    export_to_csv("interpolation_comparison.csv", &export_data)?;

    // Export MSE values
    let mut mse_file =
        File::create("interpolation_mse.csv").map_err(|e| SignalError::Compute(e.to_string()))?;
    writeln!(mse_file, "Method,MSE").map_err(|e| SignalError::Compute(e.to_string()))?;

    for &(name, mse) in &mse_values {
        writeln!(mse_file, "{},{}", name, mse).map_err(|e| SignalError::Compute(e.to_string()))?;
    }

    println!("MSE values exported to interpolation_mse.csv");

    Ok(())
}

/// Example of interpolating a bandlimited signal
fn interpolate_bandlimited_signal() -> SignalResult<()> {
    println!("Interpolating a Bandlimited Signal");

    // Generate a bandlimited signal
    let n_samples = 200;
    let nyquist = n_samples / 2;
    let cutoff = nyquist / 4; // Use 1/4 of Nyquist frequency

    // Create frequency domain representation
    let mut spectrum = vec![Complex64::new(0.0, 0.0); n_samples];

    // Add some frequency components
    let mut rng = rng();
    let normal = Normal::<f64>::new(0.0, 1.0).unwrap();

    for i in 1..cutoff {
        let amplitude: f64 = normal.sample(&mut rng).abs();
        let phase = rand::random::<f64>() * 2.0 * PI;

        // Symmetric for real signal
        spectrum[i] = Complex64::new(amplitude * phase.cos(), amplitude * phase.sin());
        spectrum[n_samples - i] = Complex64::new(amplitude * phase.cos(), -amplitude * phase.sin());
    }

    // DC component
    spectrum[0] = Complex64::new(normal.sample(&mut rng).abs() as f64, 0.0);

    // Create Nyquist component if n_samples is even
    if n_samples % 2 == 0 {
        spectrum[n_samples / 2] = Complex64::new(normal.sample(&mut rng).abs() as f64, 0.0);
    }

    // Inverse FFT to get the time domain signal
    let mut planner = rustfft::FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_samples);

    let mut spectrum_copy = spectrum.clone();
    ifft.process(&mut spectrum_copy);

    // Scale and extract real part
    let scale = 1.0 / n_samples as f64;
    let mut reference = Array1::zeros(n_samples);
    for i in 0..n_samples {
        reference[i] = spectrum_copy[i].re * scale;
    }

    // Create a signal with missing values (simulate downsampling)
    let mut signal = reference.clone();

    // Remove every other point
    for i in 0..n_samples {
        if i % 2 == 1 {
            signal[i] = f64::NAN;
        }
    }

    // Configure interpolation
    let config = interpolate::InterpolationConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        regularization: 1e-6,
        window_size: 10,
        extrapolate: true,
        monotonic: false,
        smoothing: false,
        smoothing_factor: 0.1,
        frequency_constraint: true,
        cutoff_frequency: 0.5, // Use full bandwidth for bandlimited signal
    };

    // Apply different interpolation methods
    let linear_result =
        interpolate::interpolate(&signal, interpolate::InterpolationMethod::Linear, &config)?;

    let spline_result = interpolate::interpolate(
        &signal,
        interpolate::InterpolationMethod::CubicSpline,
        &config,
    )?;

    let sinc_result =
        interpolate::interpolate(&signal, interpolate::InterpolationMethod::Sinc, &config)?;

    let spectral_result =
        interpolate::interpolate(&signal, interpolate::InterpolationMethod::Spectral, &config)?;

    // Calculate error metrics
    let mut linear_sse = 0.0;
    let mut spline_sse = 0.0;
    let mut sinc_sse = 0.0;
    let mut spectral_sse = 0.0;
    let mut count = 0;

    for i in 0..n_samples {
        if signal[i].is_nan() {
            let linear_err = linear_result[i] - reference[i];
            let spline_err = spline_result[i] - reference[i];
            let sinc_err = sinc_result[i] - reference[i];
            let spectral_err = spectral_result[i] - reference[i];

            linear_sse += linear_err * linear_err;
            spline_sse += spline_err * spline_err;
            sinc_sse += sinc_err * sinc_err;
            spectral_sse += spectral_err * spectral_err;
            count += 1;
        }
    }

    let linear_mse = linear_sse / count as f64;
    let spline_mse = spline_sse / count as f64;
    let sinc_mse = sinc_sse / count as f64;
    let spectral_mse = spectral_sse / count as f64;

    println!("Mean squared error (bandlimited signal):");
    println!("  Linear: {:.6}", linear_mse);
    println!("  Cubic Spline: {:.6}", spline_mse);
    println!("  Sinc: {:.6}", sinc_mse);
    println!("  Spectral: {:.6}", spectral_mse);

    // Export data for plotting
    export_to_csv(
        "interpolation_bandlimited.csv",
        &[
            ("Reference", &reference),
            ("Downsampled", &signal),
            ("Linear", &linear_result),
            ("CubicSpline", &spline_result),
            ("Sinc", &sinc_result),
            ("Spectral", &spectral_result),
        ],
    )?;

    Ok(())
}

/// Example of interpolating a 2D dataset (image)
fn interpolate_2d_data() -> SignalResult<()> {
    println!("Interpolating 2D Data (Image)");

    // Create a 2D test image
    let n_rows = 50;
    let n_cols = 50;
    let mut image = Array2::zeros((n_rows, n_cols));

    // Generate a 2D pattern (peaks function)
    for i in 0..n_rows {
        for j in 0..n_cols {
            let x = (j as f64) / (n_cols as f64) * 3.0 - 1.5;
            let y = (i as f64) / (n_rows as f64) * 3.0 - 1.5;

            let r1 = (x * x + y * y).sqrt();
            let r2 = ((x + 1.0) * (x + 1.0) + y * y).sqrt();
            let r3 = ((x - 1.0) * (x - 1.0) + y * y).sqrt();

            image[[i, j]] = 3.0 * (1.0 - r1).powi(2) * (-r1 / 3.0).exp()
                - 10.0 * (r1 / 5.0 - r1.powi(3)) * (-r1 * r1).exp()
                - 1.0 / 3.0 * (-r2 * r2).exp()
                + 1.0 / 5.0 * (-r3 * r3).exp();
        }
    }

    // Create a copy with missing values
    let mut missing_image = image.clone();

    // Remove a rectangular region
    for i in 15..30 {
        for j in 20..40 {
            missing_image[[i, j]] = f64::NAN;
        }
    }

    // Remove scattered points
    let mut rng = rng();
    for _ in 0..500 {
        let i = rng.random_range(0..n_rows);
        let j = rng.random_range(0..n_cols);
        missing_image[[i, j]] = f64::NAN;
    }

    // Configure interpolation
    let config = interpolate::InterpolationConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        regularization: 1e-6,
        window_size: 10,
        extrapolate: true,
        monotonic: false,
        smoothing: true,
        smoothing_factor: 0.1,
        frequency_constraint: true,
        cutoff_frequency: 0.3,
    };

    // Apply different interpolation methods
    let linear_result = interpolate::interpolate_2d(
        &missing_image,
        interpolate::InterpolationMethod::Linear,
        &config,
    )?;

    let spline_result = interpolate::interpolate_2d(
        &missing_image,
        interpolate::InterpolationMethod::CubicSpline,
        &config,
    )?;

    let nearest_result = interpolate::interpolate_2d(
        &missing_image,
        interpolate::InterpolationMethod::NearestNeighbor,
        &config,
    )?;

    // Calculate error metrics
    let mut linear_sse = 0.0;
    let mut spline_sse = 0.0;
    let mut nearest_sse = 0.0;
    let mut count = 0;

    for i in 0..n_rows {
        for j in 0..n_cols {
            if missing_image[[i, j]].is_nan() {
                let linear_err = linear_result[[i, j]] - image[[i, j]];
                let spline_err = spline_result[[i, j]] - image[[i, j]];
                let nearest_err = nearest_result[[i, j]] - image[[i, j]];

                linear_sse += linear_err * linear_err;
                spline_sse += spline_err * spline_err;
                nearest_sse += nearest_err * nearest_err;
                count += 1;
            }
        }
    }

    let linear_mse = linear_sse / count as f64;
    let spline_mse = spline_sse / count as f64;
    let nearest_mse = nearest_sse / count as f64;

    println!("Mean squared error (2D image):");
    println!("  Linear: {:.6}", linear_mse);
    println!("  Cubic Spline: {:.6}", spline_mse);
    println!("  Nearest Neighbor: {:.6}", nearest_mse);

    // Export data for visualization
    export_2d_to_csv("interpolation_2d_original.csv", &image)?;
    export_2d_to_csv("interpolation_2d_missing.csv", &missing_image)?;
    export_2d_to_csv("interpolation_2d_linear.csv", &linear_result)?;
    export_2d_to_csv("interpolation_2d_spline.csv", &spline_result)?;
    export_2d_to_csv("interpolation_2d_nearest.csv", &nearest_result)?;

    Ok(())
}

/// Example of automatic method selection for interpolation
fn auto_interpolation_example() -> SignalResult<()> {
    println!("Automatic Method Selection for Interpolation");

    // Create a piecewise signal with different characteristics
    let n_samples = 300;
    let mut reference = Array1::zeros(n_samples);
    let x = Array1::linspace(0.0, 10.0, n_samples);

    // Piecewise signal: smooth, linear, oscillatory
    for i in 0..n_samples {
        if x[i] < 3.0 {
            // Smooth region: polynomial
            reference[i] = 1.0 + 0.5 * x[i] - 0.1 * x[i] * x[i];
        } else if x[i] < 6.0 {
            // Linear region
            reference[i] = 1.0 + 0.3 * (x[i] - 3.0);
        } else {
            // Oscillatory region
            reference[i] = 2.0 + 0.5 * ((x[i] - 6.0) as f64).sin() * (x[i] - 6.0);
        }
    }

    // Create three test cases with different missing patterns
    let mut signals = Vec::new();

    // Case 1: Random missing values
    let mut signal1 = reference.clone();
    let _rng = rng();
    for i in 0..n_samples {
        if rand::random::<f64>() < 0.2 {
            signal1[i] = f64::NAN;
        }
    }
    signals.push(("Random", signal1));

    // Case 2: Blocks of missing values
    let mut signal2 = reference.clone();
    for i in 0..n_samples {
        if (50..70).contains(&i) || (150..180).contains(&i) || (250..270).contains(&i) {
            signal2[i] = f64::NAN;
        }
    }
    signals.push(("Blocks", signal2));

    // Case 3: Sparse sampling
    let mut signal3 = Array1::zeros(n_samples);
    signal3.fill(f64::NAN);
    for i in 0..n_samples {
        if i % 10 == 0 {
            signal3[i] = reference[i];
        }
    }
    signals.push(("Sparse", signal3));

    // Configure interpolation
    let config = interpolate::InterpolationConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        regularization: 1e-6,
        window_size: 10,
        extrapolate: true,
        monotonic: false,
        smoothing: false,
        smoothing_factor: 0.1,
        frequency_constraint: true,
        cutoff_frequency: 0.3,
    };

    // Apply auto interpolation with cross-validation
    for (name, signal) in &signals {
        let (result, best_method) = interpolate::auto_interpolate(signal, &config, true)?;

        // Calculate error metrics
        let mut sse = 0.0;
        let mut count = 0;

        for i in 0..n_samples {
            if signal[i].is_nan() {
                let err = result[i] - reference[i];
                sse += err * err;
                count += 1;
            }
        }

        let mse = sse / count as f64;

        println!("{}:", name);
        println!("  Best method: {:?}", best_method);
        println!("  MSE: {:.6}", mse);

        // Export data for plotting
        export_to_csv(
            &format!("interpolation_auto_{}.csv", name.to_lowercase()),
            &[
                ("Reference", &reference),
                ("Missing", signal),
                ("Interpolated", &result),
            ],
        )?;
    }

    Ok(())
}
