use ndarray::{Array, Array1, Array2};
use rand::{rng, Rng};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::reassigned::{
    extract_ridges, reassigned_spectrogram, smoothed_reassigned_spectrogram, ReassignedConfig,
    ReassignedResult,
};
use scirs2_signal::spectral;
use scirs2_signal::window;

fn main() {
    println!("Reassigned Spectrogram Examples");
    println!("-------------------------------");

    // Generate a test signal
    let signal = generate_test_signal();

    // Demonstrate different variants of the reassigned spectrogram
    println!("Analyzing signal with standard STFT spectrogram...");
    let stft_result = compute_standard_spectrogram(&signal);

    println!("Analyzing signal with reassigned spectrogram...");
    let reassigned_result = compute_reassigned_spectrogram(&signal);

    println!("Analyzing signal with smoothed reassigned spectrogram...");
    let smoothed_result = compute_smoothed_reassigned_spectrogram(&signal);

    println!("Analyzing multi-component signal...");
    let multi_results = analyze_multicomponent_signal();

    // Extract and analyze ridges
    println!("Extracting time-frequency ridges...");
    extract_and_analyze_ridges(&reassigned_result);

    // Save results to CSV files for external visualization
    println!("Saving results to CSV files...");
    save_results_to_csv(
        &signal,
        &stft_result,
        &reassigned_result,
        &smoothed_result,
        &multi_results.0,
        &multi_results.1,
        &multi_results.2,
    );

    println!("Done! CSV files have been created for visualization.");
}

/// Generate a test signal with multiple components
fn generate_test_signal() -> Array1<f64> {
    let n_samples = 2048;
    let fs = 2048.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a signal with:
    // 1. A linear chirp from 100Hz to 400Hz
    // 2. A short burst at t=0.5s with a different frequency

    // Component 1: Linear chirp
    let f0 = 100.0;
    let f1 = 400.0;
    let rate = (f1 - f0) / duration;
    let chirp = t.mapv(|ti| (2.0 * PI * (f0 * ti + 0.5 * rate * ti * ti)).sin());

    // Component 2: Short burst
    let center = duration / 2.0;
    let width = 0.05;
    let burst = t.mapv(|ti| {
        let gaussian = (-((ti - center) / width).powi(2)).exp();
        gaussian * (2.0 * PI * 250.0 * ti).sin() * 0.8
    });

    // Add some noise
    let noise_level = 0.05;
    let mut rng = rng();
    let noise =
        Array::from_iter((0..n_samples).map(|_| noise_level * (2.0 * rng.random::<f64>() - 1.0)));

    // Combine components
    &chirp + &burst + &noise
}

/// Compute a standard STFT spectrogram
fn compute_standard_spectrogram(signal: &Array1<f64>) -> Array2<f64> {
    let fs = 2048.0;
    let window_size = 256;
    let hop_size = 64;
    let n_fft = 512;

    let _win = window::hann(window_size, true).unwrap();

    // Compute STFT using spectral module
    let stft_result = spectral::stft(
        signal.as_slice().unwrap(),
        Some(fs),
        Some("hann"),
        Some(window_size),
        Some(window_size - hop_size), // noverlap = nperseg - hop_size
        Some(n_fft),
        None,
        None,
        None,
    )
    .unwrap();

    // STFT returns (frequencies, times, STFT values)
    let (_, _, stft_complex) = stft_result;

    // Convert to power spectrogram
    let mut spectrogram = Array2::zeros((n_fft / 2 + 1, stft_complex.len()));
    for i in 0..spectrogram.shape()[0] {
        for j in 0..spectrogram.shape()[1] {
            spectrogram[[i, j]] = stft_complex[j][i].norm_sqr();
        }
    }

    println!(
        "Standard STFT spectrogram: {} time frames, {} frequency bins",
        spectrogram.shape()[1],
        spectrogram.shape()[0]
    );

    spectrogram
}

/// Compute a reassigned spectrogram
fn compute_reassigned_spectrogram(signal: &Array1<f64>) -> ReassignedResult {
    let fs = 2048.0;

    // Configure the reassigned spectrogram
    let mut config = ReassignedConfig::default();
    config.window = Array1::from(window::hann(256, true).unwrap());
    config.hop_size = 64;
    config.n_fft = Some(512);
    config.fs = fs;
    config.return_spectrogram = true;

    // Compute the reassigned spectrogram
    let result = reassigned_spectrogram(signal, config).unwrap();

    println!(
        "Reassigned spectrogram: {} time frames, {} frequency bins",
        result.reassigned.shape()[1],
        result.reassigned.shape()[0]
    );

    result
}

/// Compute a smoothed reassigned spectrogram
fn compute_smoothed_reassigned_spectrogram(signal: &Array1<f64>) -> ReassignedResult {
    let fs = 2048.0;

    // Configure the reassigned spectrogram
    let mut config = ReassignedConfig::default();
    config.window = Array1::from(window::hann(256, true).unwrap());
    config.hop_size = 64;
    config.n_fft = Some(512);
    config.fs = fs;
    config.return_spectrogram = true;

    // Compute the smoothed reassigned spectrogram with smoothing width of 3
    let result = smoothed_reassigned_spectrogram(signal, config, 3).unwrap();

    println!(
        "Smoothed reassigned spectrogram: {} time frames, {} frequency bins",
        result.reassigned.shape()[1],
        result.reassigned.shape()[0]
    );

    result
}

/// Analyze a multi-component signal to highlight the advantages of reassignment
fn analyze_multicomponent_signal() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_samples = 2048;
    let fs = 2048.0;
    let duration = n_samples as f64 / fs;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a multi-component signal with crossing chirps
    let up_chirp = t.mapv(|ti| {
        let phase = 2.0 * PI * (100.0 * ti + 150.0 * ti * ti);
        phase.sin()
    });

    let down_chirp = t.mapv(|ti| {
        let phase = 2.0 * PI * (400.0 * ti - 150.0 * ti * ti);
        phase.sin() * 0.8
    });

    let signal = &up_chirp + &down_chirp;

    // Compute standard spectrogram
    let window_size = 256;
    let hop_size = 64;
    let n_fft = 512;

    let _win = window::hann(window_size, true).unwrap();

    let stft_result = spectral::stft(
        signal.as_slice().unwrap(),
        Some(fs),
        Some("hann"),
        Some(window_size),
        Some(window_size - hop_size), // noverlap = nperseg - hop_size
        Some(n_fft),
        None,
        None,
        None,
    )
    .unwrap();

    // STFT returns (frequencies, times, STFT values)
    let (_, _, stft_complex) = stft_result;

    // Convert to power spectrogram
    let mut spectrogram = Array2::zeros((n_fft / 2 + 1, stft_complex.len()));
    for i in 0..spectrogram.shape()[0] {
        for j in 0..spectrogram.shape()[1] {
            spectrogram[[i, j]] = stft_complex[j][i].norm_sqr();
        }
    }

    // Configure the reassigned spectrogram
    let mut config = ReassignedConfig::default();
    config.window = Array1::from(window::hann(window_size, true).unwrap());
    config.hop_size = hop_size;
    config.n_fft = Some(n_fft);
    config.fs = fs;

    // Compute reassigned and smoothed reassigned spectrograms
    let reassigned_result = reassigned_spectrogram(&signal, config.clone()).unwrap();
    let smoothed_result = smoothed_reassigned_spectrogram(&signal, config, 3).unwrap();

    println!("Multi-component analysis:");
    println!("  - Standard spectrogram shows limited time-frequency resolution");
    println!("  - Reassigned spectrogram shows sharper component tracks but with some scattering");
    println!("  - Smoothed reassigned spectrogram balances resolution and clarity");

    (
        spectrogram,
        reassigned_result.reassigned,
        smoothed_result.reassigned,
    )
}

/// Extract and analyze ridges from a reassigned spectrogram
fn extract_and_analyze_ridges(result: &ReassignedResult) -> Vec<Vec<(usize, f64)>> {
    // Extract the ridges (maximum 2 components)
    let ridges = extract_ridges(&result.reassigned, &result.frequencies, 2, 0.2);

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

/// Save results to CSV files for external plotting
fn save_results_to_csv(
    signal: &Array1<f64>,
    stft_result: &Array2<f64>,
    reassigned_result: &ReassignedResult,
    smoothed_result: &ReassignedResult,
    multi_stft: &Array2<f64>,
    multi_reassigned: &Array2<f64>,
    multi_smoothed: &Array2<f64>,
) {
    // Save the original signal
    let mut file = File::create("signal.csv").expect("Failed to create signal.csv");
    for (i, val) in signal.iter().enumerate() {
        let time = i as f64 / 2048.0;
        writeln!(file, "{},{}", time, val).expect("Failed to write to signal.csv");
    }

    // Save the standard STFT spectrogram
    save_matrix_to_csv(
        "stft_spectrogram.csv",
        stft_result,
        &reassigned_result.times,
        &reassigned_result.frequencies,
    );

    // Save the reassigned spectrogram
    save_matrix_to_csv(
        "reassigned_spectrogram.csv",
        &reassigned_result.reassigned,
        &reassigned_result.times,
        &reassigned_result.frequencies,
    );

    // Save the smoothed reassigned spectrogram
    save_matrix_to_csv(
        "smoothed_reassigned_spectrogram.csv",
        &smoothed_result.reassigned,
        &smoothed_result.times,
        &smoothed_result.frequencies,
    );

    // Save the multi-component results
    let t_multi = Array1::linspace(0.0, 1.0, multi_stft.shape()[1]);
    let f_multi = Array1::linspace(0.0, 1024.0, multi_stft.shape()[0]);

    save_matrix_to_csv("multi_stft.csv", multi_stft, &t_multi, &f_multi);
    save_matrix_to_csv("multi_reassigned.csv", multi_reassigned, &t_multi, &f_multi);
    save_matrix_to_csv("multi_smoothed.csv", multi_smoothed, &t_multi, &f_multi);

    // Save any extracted ridges
    let ridges = extract_ridges(
        &reassigned_result.reassigned,
        &reassigned_result.frequencies,
        2,
        0.2,
    );
    for (i, ridge) in ridges.iter().enumerate() {
        let filename = format!("ridge_{}.csv", i + 1);
        let mut file = File::create(&filename).expect(&format!("Failed to create {}", filename));

        writeln!(file, "time,frequency").expect("Failed to write header");
        for &(t, f) in ridge {
            let time = reassigned_result.times[t];
            writeln!(file, "{},{}", time, f).expect("Failed to write data");
        }
    }
}

/// Save a matrix to CSV with time and frequency labels
fn save_matrix_to_csv(
    filename: &str,
    matrix: &Array2<f64>,
    times: &Array1<f64>,
    frequencies: &Array1<f64>,
) {
    let mut file = File::create(filename).expect(&format!("Failed to create {}", filename));

    // Write header with time values
    write!(file, "frequency").expect("Failed to write header");
    for &t in times.iter() {
        write!(file, ",{}", t).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write matrix data with frequency labels
    for (i, &f) in frequencies.iter().enumerate() {
        write!(file, "{}", f).expect("Failed to write data");
        for j in 0..matrix.shape()[1] {
            write!(file, ",{}", matrix[[i, j]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }
}
