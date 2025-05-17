use ndarray::{Array, Array1, Array2};
// use num_complex::Complex64;
use rand::{rng, Rng};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::window;
use scirs2_signal::wvd::{
    cross_wigner_ville, extract_ridges, frequency_axis, smoothed_pseudo_wigner_ville, time_axis,
    wigner_ville, WvdConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Wigner-Ville Distribution Examples");
    println!("---------------------------------");

    // Generate test signals
    let signal = generate_test_signal();

    // Demonstrate different variants of the Wigner-Ville distribution
    println!("Analyzing signal with standard Wigner-Ville Distribution...");
    let (wvd, t_axis, f_axis) = analyze_with_standard_wvd(&signal);

    println!("Analyzing signal with Smoothed Pseudo Wigner-Ville Distribution...");
    let (spwvd, _) = analyze_with_smoothed_wvd(&signal, &t_axis, &f_axis)?;

    println!("Analyzing multi-component signal to show cross-terms...");
    let (wvd_multi, spwvd_multi) = compare_wvd_spwvd_multicomponent()?;

    println!("Demonstrating Cross Wigner-Ville Distribution...");
    let xwvd = demonstrate_cross_wvd();

    // Extract and analyze ridges
    println!("Extracting time-frequency ridges...");
    extract_and_analyze_ridges(&spwvd, &f_axis);

    // Save results to CSV files for external visualization
    println!("Saving results to CSV files...");
    save_results_to_csv(
        &signal,
        &wvd,
        &spwvd,
        &wvd_multi,
        &spwvd_multi,
        &xwvd,
        &t_axis,
        &f_axis,
    );

    println!("Done! CSV files have been created for visualization.");
    Ok(())
}

/// Generate a test signal with multiple chirp components and a transient
fn generate_test_signal() -> Array1<f64> {
    let n_samples = 1024;
    let fs = 1024.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a signal with:
    // 1. A linear chirp from 50Hz to 200Hz
    // 2. A short transient at t=0.5s

    // Component 1: Linear chirp
    let f0 = 50.0;
    let f1 = 200.0;
    let rate = (f1 - f0) / duration;
    let chirp = t.mapv(|ti| (2.0 * PI * (f0 * ti + 0.5 * rate * ti * ti)).sin());

    // Component 2: Transient pulse (Gaussian-windowed sinusoid)
    let center = duration / 2.0;
    let width = 0.05;
    let transient = t.mapv(|ti| {
        let gaussian = (-((ti - center) / width).powi(2)).exp();
        gaussian * (2.0 * PI * 150.0 * ti).sin() * 0.5
    });

    // Add some noise
    let noise_level = 0.05;
    let mut rng = rng();
    let noise = Array::from_iter(
        (0..n_samples).map(|_| noise_level * (2.0 * rng.random_range(0.0..1.0) - 1.0)),
    );

    // Combine components
    &chirp + &transient + &noise
}

/// Analyze a signal with the standard Wigner-Ville Distribution
fn analyze_with_standard_wvd(signal: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    // Configure the transform
    let fs = 1024.0;
    let config = WvdConfig {
        analytic: true,
        time_window: None,
        freq_window: None,
        zero_padding: true,
        fs,
    };

    // Compute the WVD
    let wvd = wigner_ville(signal, config).unwrap();

    // Create time and frequency axes
    let f_axis = frequency_axis(wvd.shape()[0], fs);
    let t_axis = time_axis(wvd.shape()[1], fs);

    println!(
        "Standard WVD: {} time points, {} frequency bins",
        wvd.shape()[1],
        wvd.shape()[0]
    );

    (wvd, t_axis, f_axis)
}

/// Analyze a signal with the Smoothed Pseudo Wigner-Ville Distribution
fn analyze_with_smoothed_wvd(
    signal: &Array1<f64>,
    _t_axis: &Array1<f64>,
    f_axis: &Array1<f64>,
) -> Result<(Array2<f64>, Vec<Vec<(usize, f64)>>), Box<dyn std::error::Error>> {
    // Create windows for time and frequency smoothing
    let time_window_size = 51;
    let freq_window_size = 101;

    let time_window = window::hamming(time_window_size, true)?;
    let freq_window = window::hamming(freq_window_size, true)?;

    // Configure the transform
    let fs = 1024.0;
    let config = WvdConfig {
        analytic: true,
        time_window: None, // Will be provided to the SPWVD function directly
        freq_window: None, // Will be provided to the SPWVD function directly
        zero_padding: true,
        fs,
    };

    // Compute the SPWVD
    let time_win = Array1::from(time_window);
    let freq_win = Array1::from(freq_window);
    let spwvd = smoothed_pseudo_wigner_ville(signal, &time_win, &freq_win, config).unwrap();

    println!(
        "Smoothed Pseudo WVD: {} time points, {} frequency bins",
        spwvd.shape()[1],
        spwvd.shape()[0]
    );

    // Extract ridges from the SPWVD
    let ridges = extract_ridges(&spwvd, f_axis, 2, 0.3);

    Ok((spwvd, ridges))
}

/// Compare standard WVD and SPWVD on a multi-component signal
fn compare_wvd_spwvd_multicomponent(
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let n_samples = 512;
    let fs = 512.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a multi-component signal (two crossing chirps)
    let chirp1 = t.mapv(|ti| (2.0 * PI * (50.0 * ti + 100.0 * ti * ti)).sin());
    let chirp2 = t.mapv(|ti| (2.0 * PI * (200.0 * ti - 100.0 * ti * ti)).sin());

    let signal = &chirp1 + &chirp2;

    // Configure the standard WVD
    let config = WvdConfig {
        analytic: true,
        time_window: None,
        freq_window: None,
        zero_padding: true,
        fs,
    };

    // Compute the standard WVD
    let wvd = wigner_ville(&signal, config.clone()).unwrap();

    // Configure windows for SPWVD
    let time_window = window::hamming(31, true)?;
    let freq_window = window::hamming(61, true)?;

    // Compute the SPWVD
    let time_win = Array1::from(time_window);
    let freq_win = Array1::from(freq_window);
    let spwvd = smoothed_pseudo_wigner_ville(&signal, &time_win, &freq_win, config).unwrap();

    println!("Multi-component analysis:");
    println!("  - Standard WVD shows interference (cross-terms) between the chirps");
    println!("  - SPWVD reduces interference at the cost of slightly reduced resolution");

    Ok((wvd, spwvd))
}

/// Demonstrate cross-Wigner-Ville Distribution between related signals
fn demonstrate_cross_wvd() -> Array2<f64> {
    let n_samples = 512;
    let fs = 512.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create two related signals with quadrature phase relationship
    let signal_sin = t.mapv(|ti| (2.0 * PI * (50.0 * ti + 50.0 * ti * ti)).sin());
    let signal_cos = t.mapv(|ti| (2.0 * PI * (50.0 * ti + 50.0 * ti * ti)).cos());

    // Configure the cross-WVD
    let config = WvdConfig {
        analytic: true,
        time_window: None,
        freq_window: None,
        zero_padding: true,
        fs,
    };

    // Compute the cross-WVD
    let xwvd_complex = cross_wigner_ville(&signal_sin, &signal_cos, config).unwrap();

    // Take magnitude for visualization
    let xwvd = xwvd_complex.mapv(|c| c.norm());

    println!("Cross-WVD between sine and cosine components:");
    println!("  - Shows the quadrature phase relationship between signals");
    println!("  - Energy concentrated along the instantaneous frequency path");

    xwvd
}

/// Extract and analyze ridges from the time-frequency representation
fn extract_and_analyze_ridges(
    tf_repr: &Array2<f64>,
    freq_axis: &Array1<f64>,
) -> Vec<Vec<(usize, f64)>> {
    // Extract the ridges (maximum 2 components)
    let ridges = extract_ridges(tf_repr, freq_axis, 2, 0.2);

    println!("Found {} significant ridges", ridges.len());

    // For each ridge, print some information
    for (i, ridge) in ridges.iter().enumerate() {
        if ridge.is_empty() {
            continue;
        }

        // Calculate statistics about the ridge
        let n_points = ridge.len();
        let start_freq = ridge.first().map(|&(_, f)| f).unwrap_or(0.0);
        let end_freq = ridge.last().map(|&(_, f)| f).unwrap_or(0.0);

        println!(
            "Ridge {}: {} points, frequency range: {:.2} Hz to {:.2} Hz",
            i + 1,
            n_points,
            start_freq,
            end_freq
        );

        if start_freq < end_freq {
            println!("  Frequency trend: Increasing chirp");
        } else if start_freq > end_freq {
            println!("  Frequency trend: Decreasing chirp");
        } else {
            println!("  Frequency trend: Approximately constant");
        }
    }

    ridges
}

/// Save all results to CSV files for external plotting
fn save_results_to_csv(
    signal: &Array1<f64>,
    wvd: &Array2<f64>,
    spwvd: &Array2<f64>,
    wvd_multi: &Array2<f64>,
    spwvd_multi: &Array2<f64>,
    xwvd: &Array2<f64>,
    t_axis: &Array1<f64>,
    f_axis: &Array1<f64>,
) {
    // Save the original signal
    let mut file = File::create("signal.csv").expect("Failed to create signal.csv");
    for (i, val) in signal.iter().enumerate() {
        writeln!(file, "{},{}", t_axis[i], val).expect("Failed to write to signal.csv");
    }

    // Save the standard WVD
    save_matrix_to_csv("wvd.csv", wvd, t_axis, f_axis);

    // Save the SPWVD
    save_matrix_to_csv("spwvd.csv", spwvd, t_axis, f_axis);

    // Save the multi-component results
    let t_multi = Array1::linspace(0.0, 1.0, wvd_multi.shape()[1]);
    let f_multi = frequency_axis(wvd_multi.shape()[0], 512.0);

    save_matrix_to_csv("wvd_multi.csv", wvd_multi, &t_multi, &f_multi);
    save_matrix_to_csv("spwvd_multi.csv", spwvd_multi, &t_multi, &f_multi);

    // Save the cross-WVD
    let t_cross = Array1::linspace(0.0, 1.0, xwvd.shape()[1]);
    let f_cross = frequency_axis(xwvd.shape()[0], 512.0);

    save_matrix_to_csv("xwvd.csv", xwvd, &t_cross, &f_cross);
}

/// Save a time-frequency matrix to CSV
fn save_matrix_to_csv(
    filename: &str,
    matrix: &Array2<f64>,
    t_axis: &Array1<f64>,
    f_axis: &Array1<f64>,
) {
    let mut file = File::create(filename).expect(&format!("Failed to create {}", filename));

    // Write header with time values
    write!(file, "frequency").expect("Failed to write header");
    for &t in t_axis.iter() {
        write!(file, ",{}", t).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write matrix data with frequency labels
    for (i, &f) in f_axis.iter().enumerate() {
        write!(file, "{}", f).expect("Failed to write data");
        for j in 0..matrix.shape()[1] {
            write!(file, ",{}", matrix[[i, j]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }
}
