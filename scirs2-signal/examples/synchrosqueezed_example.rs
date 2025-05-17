use ndarray::{Array, Array1, Array2};
use rand::Rng;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::sswt::{self, SynchroCwtConfig, SynchroCwtResult};
use scirs2_signal::wavelets;

fn main() {
    println!("Synchrosqueezed Wavelet Transform Example");
    println!("----------------------------------------");

    // Generate a multi-component signal
    let signal = generate_test_signal();

    // Analyze the signal
    println!("Analyzing signal with both CWT and Synchrosqueezed Transform...");
    let (cwt_result, sst_result) = analyze_signal(&signal);

    // Extract and analyze ridges
    println!("Extracting time-frequency ridges...");
    let ridges = extract_and_analyze_ridges(&sst_result);

    // Optional: Save results to CSV files for external plotting
    println!("Saving results to CSV files...");
    save_results_to_csv(&signal, &cwt_result, &sst_result, &ridges);

    println!("Done! CSV files have been created for visualization.");
}

/// Generate a test signal with multiple chirp components
fn generate_test_signal() -> Array1<f64> {
    let n_samples = 1000;
    let duration = 10.0;

    let t = Array1::linspace(0.0, duration, n_samples);

    // Create a signal with three components:
    // 1. A linear chirp from 1Hz to 6Hz
    // 2. A constant frequency component at 8Hz
    // 3. A hyperbolic chirp (decreasing frequency)

    // Component 1: Linear chirp
    let f0_1 = 1.0;
    let f1_1 = 6.0;
    let rate_1 = (f1_1 - f0_1) / duration;
    let chirp_1 = t.mapv(|ti| (2.0 * PI * (f0_1 * ti + 0.5 * rate_1 * ti * ti)).sin());

    // Component 2: Constant frequency
    let f_2 = 8.0;
    let constant = t.mapv(|ti| 0.5 * (2.0 * PI * f_2 * ti).sin());

    // Component 3: Hyperbolic chirp (decreasing frequency)
    let f0_3 = 12.0;
    let f1_3 = 4.0;
    let k = ((f1_3 / f0_3) as f64).powf(1.0 / duration);
    let chirp_3 = t.mapv(|ti| {
        if ti < 5.0 {
            return 0.0;
        }
        let ti_adj = ti - 5.0;
        let _freq = f0_3 * k.powf(ti_adj);
        let phase = 2.0 * PI * f0_3 * (k.powf(ti_adj) - 1.0) / k.ln();
        0.7 * phase.sin()
    });

    // Add some noise
    let noise_level = 0.1;
    let mut rng = rand::rng();
    let noise = Array::from_iter(
        (0..n_samples).map(|_| noise_level * (2.0 * rng.random_range(0.0..1.0) - 1.0)),
    );

    // Combine components
    &chirp_1 + &constant + &chirp_3 + &noise
}

/// Analyze the signal using both CWT and Synchrosqueezed Transform
fn analyze_signal(signal: &Array1<f64>) -> (Array2<f64>, SynchroCwtResult) {
    // Create logarithmically spaced scales for CWT
    let scales = sswt::log_scales(1.0, 24.0, 48);

    // Convert signal to Vec for wavelets::cwt
    let signal_vec: Vec<f64> = signal.iter().copied().collect();

    // Convert scales to Vec for wavelets::cwt
    let scales_vec: Vec<f64> = scales.iter().copied().collect();

    // Compute the CWT magnitude for comparison
    let w0 = 5.0; // Center frequency for the Morlet wavelet
    let cwt_vec = wavelets::cwt(
        &signal_vec,
        |points, scale| wavelets::morlet(points, w0, scale),
        &scales_vec,
    )
    .unwrap();

    // Convert the CWT result to Array2 for easier processing
    let n_scales = scales.len();
    let n_samples = signal.len();
    let mut cwt_coeffs = Array2::zeros((n_scales, n_samples));

    for (i, row) in cwt_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cwt_coeffs[[i, j]] = val;
        }
    }
    let cwt_power = cwt_coeffs.mapv(|c| c.norm().powi(2));

    // Configure the synchrosqueezed transform
    let mut config = SynchroCwtConfig::default();
    config.frequencies = sswt::frequency_bins(1.0, 16.0, 120);
    config.return_cwt = true; // Also return the CWT for comparison
    config.gamma = 1e-6; // Threshold for excluding low-amplitude coefficients

    // Compute the synchrosqueezed transform
    let w0 = 5.0; // Center frequency of the Morlet wavelet
    let sst_result = sswt::synchrosqueezed_cwt(
        signal,
        &scales,
        |points, scale| wavelets::morlet(points, w0, scale), // Using Morlet wavelet
        w0, // Center frequency of the Morlet wavelet
        config,
    )
    .unwrap();

    (cwt_power, sst_result)
}

/// Extract and analyze time-frequency ridges from the synchrosqueezed transform
fn extract_and_analyze_ridges(sst_result: &SynchroCwtResult) -> Vec<Vec<(usize, f64)>> {
    // Extract the top 3 ridges (we have 3 components in our signal)
    let ridges = sswt::extract_ridges(&sst_result.sst, &sst_result.frequencies, 3);

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

        // Analyze the frequency slope (positive = increasing, negative = decreasing)
        if n_points > 10 {
            let mid_idx = n_points / 2;
            let first_section = &ridge[0..mid_idx];
            let second_section = &ridge[mid_idx..];

            let avg_freq_first: f64 =
                first_section.iter().map(|&(_, f)| f).sum::<f64>() / first_section.len() as f64;
            let avg_freq_second: f64 =
                second_section.iter().map(|&(_, f)| f).sum::<f64>() / second_section.len() as f64;

            let slope = avg_freq_second - avg_freq_first;

            if slope.abs() < 0.5 {
                println!("  Frequency characteristic: Approximately constant");
            } else if slope > 0.0 {
                println!("  Frequency characteristic: Increasing (chirp up)");
            } else {
                println!("  Frequency characteristic: Decreasing (chirp down)");
            }
        }
    }

    ridges
}

/// Save the results to CSV files for external plotting
fn save_results_to_csv(
    signal: &Array1<f64>,
    cwt_power: &Array2<f64>,
    sst_result: &SynchroCwtResult,
    ridges: &[Vec<(usize, f64)>],
) {
    // Save the original signal
    let mut file = File::create("signal.csv").expect("Failed to create signal.csv");
    for (i, val) in signal.iter().enumerate() {
        writeln!(file, "{},{}", i, val).expect("Failed to write to signal.csv");
    }

    // Save the CWT power spectrogram
    let mut file = File::create("cwt_power.csv").expect("Failed to create cwt_power.csv");

    // Write header with scale values
    write!(file, "time").expect("Failed to write header");
    for scale in sst_result.scales.iter() {
        write!(file, ",{}", scale).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write spectrogram data
    for t in 0..cwt_power.shape()[1] {
        write!(file, "{}", t).expect("Failed to write data");
        for s in 0..cwt_power.shape()[0] {
            write!(file, ",{}", cwt_power[[s, t]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }

    // Save the synchrosqueezed transform
    let mut file = File::create("sst_power.csv").expect("Failed to create sst_power.csv");

    // Write header with frequency values
    write!(file, "time").expect("Failed to write header");
    for freq in sst_result.frequencies.iter() {
        write!(file, ",{}", freq).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write spectrogram data
    for t in 0..sst_result.sst.shape()[1] {
        write!(file, "{}", t).expect("Failed to write data");
        for f in 0..sst_result.sst.shape()[0] {
            write!(file, ",{}", sst_result.sst[[f, t]].norm()).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }

    // Save the ridge data
    for (i, ridge) in ridges.iter().enumerate() {
        let filename = format!("ridge_{}.csv", i + 1);
        let mut file = File::create(&filename).expect(&format!("Failed to create {}", filename));

        writeln!(file, "time,frequency").expect("Failed to write header");
        for &(t, f) in ridge {
            writeln!(file, "{},{}", t, f).expect("Failed to write data");
        }
    }
}
