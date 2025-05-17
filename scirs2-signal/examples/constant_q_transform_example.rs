use ndarray::Array1;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::cqt::{
    chromagram, constant_q_transform, cqt_magnitude, inverse_constant_q_transform, CqtConfig,
};

fn main() {
    println!("Constant-Q Transform Examples");
    println!("----------------------------");

    // Generate test signals
    println!("Generating test signals...");
    let signals = generate_test_signals();

    // Analyze single-tone signal
    println!("\n1. Basic CQT of a single tone");
    analyze_single_tone(&signals.0);

    // Analyze a chord
    println!("\n2. Analyzing a musical chord");
    analyze_chord(&signals.1);

    // Time-frequency analysis of a chirp
    println!("\n3. Time-frequency analysis of a chirp signal");
    analyze_chirp(&signals.2);

    // Frequency resolution comparison
    println!("\n4. Comparing CQT with linear frequency resolution");
    compare_frequency_resolution(&signals.3);

    // Signal reconstruction
    println!("\n5. Signal reconstruction from CQT");
    test_reconstruction(&signals.0);

    // Create chromagram from a chord progression
    println!("\n6. Chromagram analysis of a chord progression");
    analyze_chord_progression(&signals.4);

    println!("\nDone! All results have been saved to CSV files for visualization.");
}

/// Generate various test signals for the examples
fn generate_test_signals() -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    // Common parameters
    let fs = 22050.0;
    let duration = 2.0;
    let n_samples = (fs * duration) as usize;
    let t = Array1::linspace(0.0, duration, n_samples);

    // 1. Simple sine wave (A4 = 440 Hz)
    let single_tone = t.mapv(|ti| (2.0 * PI * 440.0 * ti).sin());

    // 2. C major chord (C4 = 261.63 Hz, E4 = 329.63 Hz, G4 = 392.0 Hz)
    let chord = t.mapv(|ti| {
        0.5 * (2.0 * PI * 261.63 * ti).sin()
            + 0.5 * (2.0 * PI * 329.63 * ti).sin()
            + 0.5 * (2.0 * PI * 392.0 * ti).sin()
    });

    // 3. Exponential chirp (110 Hz to 880 Hz)
    let chirp = t.mapv(|ti| {
        let rate = (880.0_f64 / 110.0).ln() / duration;
        let phase = 2.0 * PI * 110.0 * (rate * ti).exp() / rate;
        phase.sin()
    });

    // 4. Two closely-spaced frequencies (400 Hz and 424 Hz, only 1 semitone apart)
    let close_tones =
        t.mapv(|ti| 0.5 * (2.0 * PI * 400.0 * ti).sin() + 0.5 * (2.0 * PI * 424.0 * ti).sin());

    // 5. Simple chord progression (C major -> G major -> A minor -> F major)
    // Each chord lasts for 0.5 seconds
    let chord_progression = t.mapv(|ti| {
        let segment = (ti / 0.5).floor() as usize % 4;

        match segment {
            0 => {
                // C major (C4, E4, G4)
                0.33 * (2.0 * PI * 261.63 * ti).sin()
                    + 0.33 * (2.0 * PI * 329.63 * ti).sin()
                    + 0.33 * (2.0 * PI * 392.0 * ti).sin()
            }
            1 => {
                // G major (G3, B3, D4)
                0.33 * (2.0 * PI * 196.0 * ti).sin()
                    + 0.33 * (2.0 * PI * 246.94 * ti).sin()
                    + 0.33 * (2.0 * PI * 293.66 * ti).sin()
            }
            2 => {
                // A minor (A3, C4, E4)
                0.33 * (2.0 * PI * 220.0 * ti).sin()
                    + 0.33 * (2.0 * PI * 261.63 * ti).sin()
                    + 0.33 * (2.0 * PI * 329.63 * ti).sin()
            }
            3 => {
                // F major (F3, A3, C4)
                0.33 * (2.0 * PI * 174.61 * ti).sin()
                    + 0.33 * (2.0 * PI * 220.0 * ti).sin()
                    + 0.33 * (2.0 * PI * 261.63 * ti).sin()
            }
            _ => 0.0,
        }
    });

    (single_tone, chord, chirp, close_tones, chord_progression)
}

/// Analyze a single tone with the Constant-Q Transform
fn analyze_single_tone(signal: &Array1<f64>) {
    // Configure CQT for single tone analysis
    let config = CqtConfig {
        f_min: 55.0,         // A1
        f_max: 2000.0,       // Up to ~B6
        bins_per_octave: 12, // Standard semitone resolution
        fs: 22050.0,
        ..CqtConfig::default()
    };

    // Compute CQT
    let cqt_result = constant_q_transform(signal, &config).unwrap();

    // Compute magnitude in dB
    let magnitude = cqt_magnitude(&cqt_result, true, None);

    // Find the peak frequency bin
    let mut max_bin = 0;
    let mut max_value = -f64::INFINITY;

    for (i, &value) in magnitude.slice(ndarray::s![.., 0]).iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_bin = i;
        }
    }

    println!("Peak frequency: {:.2} Hz", cqt_result.frequencies[max_bin]);
    println!("Magnitude at peak: {:.2} dB", magnitude[[max_bin, 0]]);

    // Save the frequency response to a CSV file
    let mut file = File::create("single_tone_cqt.csv").expect("Could not create file");
    writeln!(file, "frequency,magnitude").expect("Failed to write header");

    for (i, &freq) in cqt_result.frequencies.iter().enumerate() {
        writeln!(file, "{},{}", freq, magnitude[[i, 0]]).expect("Failed to write data");
    }
}

/// Analyze a musical chord with the Constant-Q Transform
fn analyze_chord(signal: &Array1<f64>) {
    // Configure CQT
    let config = CqtConfig {
        f_min: 65.41,        // C2
        f_max: 1046.5,       // C6
        bins_per_octave: 24, // Quarter-tone resolution for better precision
        fs: 22050.0,
        ..CqtConfig::default()
    };

    // Compute CQT
    let cqt_result = constant_q_transform(signal, &config).unwrap();

    // Compute magnitude in dB
    let magnitude = cqt_magnitude(&cqt_result, true, None);

    // Find peaks (local maxima) in the spectrum
    let mut peaks = Vec::new();
    for i in 1..(magnitude.shape()[0] - 1) {
        if magnitude[[i, 0]] > magnitude[[i - 1, 0]] && magnitude[[i, 0]] > magnitude[[i + 1, 0]] {
            // Only consider significant peaks (at least -40 dB)
            if magnitude[[i, 0]] > -40.0 {
                peaks.push((i, cqt_result.frequencies[i], magnitude[[i, 0]]));
            }
        }
    }

    // Sort peaks by magnitude
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Print the top peaks (corresponding to chord notes)
    println!("Detected peaks in the chord:");
    for (i, (_, freq, mag)) in peaks.iter().take(3).enumerate() {
        println!("  Note {}: {:.2} Hz ({:.2} dB)", i + 1, freq, mag);
    }

    // Save the frequency response to a CSV file
    let mut file = File::create("chord_cqt.csv").expect("Could not create file");
    writeln!(file, "frequency,magnitude").expect("Failed to write header");

    for (i, &freq) in cqt_result.frequencies.iter().enumerate() {
        writeln!(file, "{},{}", freq, magnitude[[i, 0]]).expect("Failed to write data");
    }

    // Compute and save chromagram
    let chroma = chromagram(&cqt_result, None, None).unwrap();

    let mut file = File::create("chord_chroma.csv").expect("Could not create file");
    writeln!(file, "pitch_class,magnitude").expect("Failed to write header");

    let pitch_classes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for (i, &pc) in pitch_classes.iter().enumerate() {
        writeln!(file, "{},{}", pc, chroma[[i, 0]]).expect("Failed to write data");
    }
}

/// Analyze a chirp signal to demonstrate time-frequency analysis
fn analyze_chirp(signal: &Array1<f64>) {
    // Configure CQT for spectrogram
    let config = CqtConfig {
        f_min: 55.0,         // A1
        f_max: 2000.0,       // Up to ~B6
        bins_per_octave: 12, // Standard semitone resolution
        fs: 22050.0,
        hop_size: Some(512), // Hop size for spectrogram
        ..CqtConfig::default()
    };

    // Compute CQT spectrogram
    let cqt_result = constant_q_transform(signal, &config).unwrap();

    // Compute magnitude in dB
    let magnitude = cqt_magnitude(&cqt_result, true, None);

    // Save the spectrogram to a CSV file
    let mut file = File::create("chirp_spectrogram.csv").expect("Could not create file");

    // Write header with time values
    write!(file, "frequency").expect("Failed to write header");
    for &t in cqt_result.times.as_ref().unwrap().iter() {
        write!(file, ",{}", t).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write data
    for (i, &freq) in cqt_result.frequencies.iter().enumerate() {
        write!(file, "{}", freq).expect("Failed to write data");
        for j in 0..magnitude.shape()[1] {
            write!(file, ",{}", magnitude[[i, j]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }

    println!(
        "CQT spectrogram dimensions: {} frequency bins x {} time frames",
        magnitude.shape()[0],
        magnitude.shape()[1]
    );
    println!(
        "Frequency range: {:.2} Hz to {:.2} Hz",
        cqt_result.frequencies[0],
        cqt_result.frequencies[cqt_result.frequencies.len() - 1]
    );
    println!(
        "Time range: {:.2} s to {:.2} s",
        cqt_result.times.as_ref().unwrap()[0],
        cqt_result.times.as_ref().unwrap()[cqt_result.times.as_ref().unwrap().len() - 1]
    );
}

/// Compare CQT's logarithmic frequency resolution with linear resolution
fn compare_frequency_resolution(signal: &Array1<f64>) {
    // CQT with high resolution
    let cqt_config = CqtConfig {
        f_min: 300.0,        // Start near the first tone (400 Hz)
        f_max: 500.0,        // End just after the second tone (424 Hz)
        bins_per_octave: 36, // High resolution (1/3 semitone)
        fs: 22050.0,
        ..CqtConfig::default()
    };

    // Compute CQT
    let cqt_result = constant_q_transform(signal, &cqt_config).unwrap();

    // Compute magnitude in dB
    let cqt_magnitude_db = cqt_magnitude(&cqt_result, true, None);

    // Save the CQT result to a CSV file
    let mut file = File::create("resolution_comparison_cqt.csv").expect("Could not create file");
    writeln!(file, "frequency,magnitude").expect("Failed to write header");

    for (i, &freq) in cqt_result.frequencies.iter().enumerate() {
        writeln!(file, "{},{}", freq, cqt_magnitude_db[[i, 0]]).expect("Failed to write data");
    }

    // For comparison, compute FFT-based power spectral density (with linear frequency bins)
    let n_fft = 8192; // Large FFT size for better resolution
    let (frequencies, psd) = scirs2_signal::spectral::periodogram(
        signal.as_slice().unwrap(),
        Some(22050.0),
        None,
        Some(n_fft),
        None,
        Some("density"),
    )
    .unwrap();

    // Convert PSD to dB
    let max_psd = psd.iter().cloned().fold(0.0, f64::max);
    let psd_db: Vec<f64> = psd.iter().map(|&p| 10.0 * (p / max_psd).log10()).collect();

    // Save the FFT result to a CSV file
    let mut file = File::create("resolution_comparison_fft.csv").expect("Could not create file");
    writeln!(file, "frequency,magnitude").expect("Failed to write header");

    // Only save the frequencies between 300 and 500 Hz (for comparison with CQT)
    for (i, &freq) in frequencies.iter().enumerate() {
        if freq >= 300.0 && freq <= 500.0 {
            writeln!(file, "{},{}", freq, psd_db[i]).expect("Failed to write data");
        }
    }

    // Analyze the results
    println!("Signal has two tones at 400 Hz and 424 Hz (1 semitone apart)");
    println!("CQT frequency resolution in this range:");

    let avg_freq_step = (cqt_result.frequencies[cqt_result.frequencies.len() - 1]
        - cqt_result.frequencies[0])
        / (cqt_result.frequencies.len() - 1) as f64;

    println!("  Average bin width: {:.2} Hz", avg_freq_step);
    println!(
        "  Frequency ratio between consecutive bins: {:.4}",
        (cqt_result.frequencies[1] / cqt_result.frequencies[0])
    );

    // Find peaks in CQT
    let mut cqt_peaks = Vec::new();
    for i in 1..(cqt_magnitude_db.shape()[0] - 1) {
        if cqt_magnitude_db[[i, 0]] > cqt_magnitude_db[[i - 1, 0]]
            && cqt_magnitude_db[[i, 0]] > cqt_magnitude_db[[i + 1, 0]]
            && cqt_magnitude_db[[i, 0]] > -40.0
        {
            cqt_peaks.push((cqt_result.frequencies[i], cqt_magnitude_db[[i, 0]]));
        }
    }

    if cqt_peaks.len() >= 2 {
        println!("  CQT detected {} major peaks:", cqt_peaks.len());
        for (i, (freq, mag)) in cqt_peaks.iter().enumerate() {
            println!("    Peak {}: {:.2} Hz ({:.2} dB)", i + 1, freq, mag);
        }
    } else {
        println!("  CQT could not resolve the two tones clearly");
    }
}

/// Test signal reconstruction from CQT coefficients
fn test_reconstruction(signal: &Array1<f64>) {
    // Configure CQT
    let config = CqtConfig {
        f_min: 55.0,         // A1
        f_max: 2000.0,       // Up to ~B6
        bins_per_octave: 24, // Higher resolution for better reconstruction
        fs: 22050.0,
        ..CqtConfig::default()
    };

    // Compute CQT
    let cqt_result = constant_q_transform(signal, &config).unwrap();

    // Reconstruct signal
    let reconstructed = inverse_constant_q_transform(&cqt_result, Some(signal.len())).unwrap();

    // Compute error metrics
    let total_error: f64 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(&orig, &rec)| (orig - rec).powi(2))
        .sum();

    let signal_power: f64 = signal.iter().map(|&x| x.powi(2)).sum();
    let reconstruction_snr = 10.0 * (signal_power / total_error).log10();

    println!("Reconstruction SNR: {:.2} dB", reconstruction_snr);

    // Save the original and reconstructed signals to CSV
    let mut file = File::create("reconstruction_comparison.csv").expect("Could not create file");
    writeln!(file, "sample,original,reconstructed").expect("Failed to write header");

    // Only save a portion of the signal for better visualization
    let start = 0;
    let end = (signal.len() as f64 * 0.1) as usize; // First 10% of the signal

    for i in start..end {
        writeln!(file, "{},{},{}", i, signal[i], reconstructed[i]).expect("Failed to write data");
    }
}

/// Analyze a chord progression using a chromagram
fn analyze_chord_progression(signal: &Array1<f64>) {
    // Configure CQT for chromagram
    let config = CqtConfig {
        f_min: 65.41,        // C2
        f_max: 1046.5,       // C6
        bins_per_octave: 12, // Standard semitone resolution
        fs: 22050.0,
        hop_size: Some(4096), // Larger hop size for smoother chromagram
        ..CqtConfig::default()
    };

    // Compute CQT
    let cqt_result = constant_q_transform(signal, &config).unwrap();

    // Compute chromagram
    let chroma = chromagram(&cqt_result, None, None).unwrap();

    // Save the chromagram to a CSV file
    let mut file = File::create("chord_progression_chroma.csv").expect("Could not create file");

    // Write header with time values
    write!(file, "pitch_class").expect("Failed to write header");
    for &t in cqt_result.times.as_ref().unwrap().iter() {
        write!(file, ",{}", t).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write data
    let pitch_classes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for (i, &pc) in pitch_classes.iter().enumerate() {
        write!(file, "{}", pc).expect("Failed to write data");
        for j in 0..chroma.shape()[1] {
            write!(file, ",{}", chroma[[i, j]]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }

    // Print chord progression analysis
    println!(
        "Chromagram has {} time frames over {:.2} seconds",
        chroma.shape()[1],
        cqt_result.times.as_ref().unwrap()[cqt_result.times.as_ref().unwrap().len() - 1]
    );

    // For each frame, find the dominant pitch classes (top 3)
    println!("\nDominant pitch classes in each segment:");
    let segments = [
        "0.0-0.5s (C major)",
        "0.5-1.0s (G major)",
        "1.0-1.5s (A minor)",
        "1.5-2.0s (F major)",
    ];

    for (segment_idx, segment_name) in segments.iter().enumerate() {
        // Find frames within this segment
        let segment_frames: Vec<usize> = cqt_result
            .times
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, &t)| t >= segment_idx as f64 * 0.5 && t < (segment_idx + 1) as f64 * 0.5)
            .map(|(i, _)| i)
            .collect();

        if segment_frames.is_empty() {
            continue;
        }

        // Average chroma values over the segment
        let mut segment_chroma = vec![0.0; 12];
        for &frame in &segment_frames {
            for pc in 0..12 {
                segment_chroma[pc] += chroma[[pc, frame]] / segment_frames.len() as f64;
            }
        }

        // Find top 3 pitch classes
        let mut pc_values: Vec<(usize, f64)> = segment_chroma
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        pc_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!(
            "  {}: {} ({:.2}), {} ({:.2}), {} ({:.2})",
            segment_name,
            pitch_classes[pc_values[0].0],
            pc_values[0].1,
            pitch_classes[pc_values[1].0],
            pc_values[1].1,
            pitch_classes[pc_values[2].0],
            pc_values[2].1
        );
    }
}
