use rand::Rng;
use scirs2_signal::multitaper::{adaptive_psd, coherence, dpss, pmtm};
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Multitaper Spectral Estimation Example");
    println!("--------------------------------------");

    // Parameters for our analysis
    let fs = 1000.0; // Sampling frequency (Hz)
    let duration = 1.0; // Signal duration (seconds)
    let n_samples = (fs * duration) as usize; // Number of samples

    // Generate time vector
    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();

    println!("\nGenerating test signals...");
    println!("  Sampling frequency: {} Hz", fs);
    println!("  Duration: {} seconds", duration);
    println!("  Number of samples: {}", n_samples);

    // Create a multi-component test signal
    // - 50 Hz sinusoid
    // - 150 Hz sinusoid (lower amplitude)
    // - 220 Hz narrow-band noise
    // - White noise

    let mut rng = rand::thread_rng();
    let mut signal = vec![0.0; n_samples];

    // Add sinusoidal components
    for i in 0..n_samples {
        let t_i = t[i];

        // 50 Hz sinusoid
        signal[i] += (2.0 * PI * 50.0 * t_i).sin();

        // 150 Hz sinusoid (lower amplitude)
        signal[i] += 0.5 * (2.0 * PI * 150.0 * t_i).sin();

        // 220 Hz narrow-band noise
        signal[i] += 0.3 * (2.0 * PI * 220.0 * t_i).sin() * 0.2 * rng.gen::<f64>();

        // White noise
        signal[i] += 0.2 * (2.0 * rng.gen::<f64>() - 1.0);
    }

    println!("  Signal components:");
    println!("    - 50 Hz sinusoid (amplitude: 1.0)");
    println!("    - 150 Hz sinusoid (amplitude: 0.5)");
    println!("    - 220 Hz narrow-band noise (amplitude: ~0.3)");
    println!("    - White noise (amplitude: 0.2)");

    // Generate a second signal for coherence analysis
    // - Has the 50 Hz component from the first signal
    // - Different 150 Hz component (phase shifted)
    // - Independent narrow-band noise at 250 Hz
    // - Independent white noise

    let mut signal2 = vec![0.0; n_samples];

    for i in 0..n_samples {
        let t_i = t[i];

        // 50 Hz sinusoid (same as signal1, should show high coherence)
        signal2[i] += (2.0 * PI * 50.0 * t_i).sin();

        // 150 Hz sinusoid (phase shifted by 90 degrees, should show medium coherence)
        signal2[i] += 0.5 * (2.0 * PI * 150.0 * t_i + PI / 2.0).sin();

        // 250 Hz narrow-band noise (different from signal1, should show low coherence)
        signal2[i] += 0.3 * (2.0 * PI * 250.0 * t_i).sin() * 0.2 * rng.gen::<f64>();

        // White noise
        signal2[i] += 0.2 * (2.0 * rng.gen::<f64>() - 1.0);
    }

    // Compute Discrete Prolate Spheroidal Sequences (DPSS) tapers
    println!("\nComputing DPSS tapers...");
    let nw = 4.0; // Time-bandwidth product
    let k = 7; // Number of tapers

    let (tapers, eigenvalues) = dpss(n_samples, nw, k, true)?;

    println!("  Time-bandwidth product (NW): {}", nw);
    println!("  Number of tapers: {}", k);
    println!(
        "  Taper shape: {} x {}",
        tapers.shape()[0],
        tapers.shape()[1]
    );

    // Print eigenvalues
    println!("  Eigenvalues:");
    for (i, &lambda) in eigenvalues.as_ref().unwrap().iter().enumerate() {
        println!("    Taper {}: {:.6}", i, lambda);
    }

    // Perform multitaper spectral analysis
    println!("\nComputing multitaper power spectral density...");

    let (freqs, psd, _, _) = pmtm(
        &signal,
        Some(fs),
        Some(nw),
        Some(k),
        None,
        Some(true),
        Some(false),
    )?;

    // Find the peak frequencies
    let mut peaks = Vec::new();
    for i in 1..(freqs.len() - 1) {
        if psd[i] > psd[i - 1]
            && psd[i] > psd[i + 1]
            && psd[i] > 0.01 * psd.iter().copied().fold(0.0, f64::max)
        {
            peaks.push((freqs[i], psd[i]));
        }
    }

    // Sort peaks by amplitude
    peaks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    println!("  Top spectral peaks:");
    for (i, &(freq, power)) in peaks.iter().take(5).enumerate() {
        println!("    Peak {}: {:.1} Hz (power: {:.6})", i + 1, freq, power);
    }

    // Compute adaptive multitaper PSD
    println!("\nComputing adaptive multitaper PSD...");

    let (_, adaptive_psd) = adaptive_psd(
        &signal,
        Some(fs),
        Some(nw),
        Some(k),
        None,
        Some(true),
        Some(true),
    )?;

    // Calculate peak-to-noise ratio for both methods
    // First, find peaks at our known frequencies
    let idx_50hz = freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 50.0).abs().partial_cmp(&(b - 50.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let idx_150hz = freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 150.0).abs().partial_cmp(&(b - 150.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // Calculate average noise level in frequency ranges away from our signals
    let noise_indices: Vec<usize> = (0..freqs.len())
        .filter(|&i| i < freqs.len() / 2 && // Only look at first half (positive frequencies)
                (freqs[i] < 30.0 || // Noise band 1: 0-30 Hz
                 (freqs[i] > 70.0 && freqs[i] < 130.0) || // Noise band 2: 70-130 Hz
                 (freqs[i] > 170.0 && freqs[i] < 200.0))) // Noise band 3: 170-200 Hz
        .collect();

    let avg_noise_standard =
        noise_indices.iter().map(|&i| psd[i]).sum::<f64>() / noise_indices.len() as f64;
    let avg_noise_adaptive =
        noise_indices.iter().map(|&i| adaptive_psd[i]).sum::<f64>() / noise_indices.len() as f64;

    println!(
        "  50 Hz peak (standard): {:.2} dB SNR",
        10.0 * (psd[idx_50hz] / avg_noise_standard).log10()
    );
    println!(
        "  50 Hz peak (adaptive): {:.2} dB SNR",
        10.0 * (adaptive_psd[idx_50hz] / avg_noise_adaptive).log10()
    );
    println!(
        "  150 Hz peak (standard): {:.2} dB SNR",
        10.0 * (psd[idx_150hz] / avg_noise_standard).log10()
    );
    println!(
        "  150 Hz peak (adaptive): {:.2} dB SNR",
        10.0 * (adaptive_psd[idx_150hz] / avg_noise_adaptive).log10()
    );

    // Coherence analysis between two signals
    println!("\nComputing multitaper coherence between two signals...");

    let (coh_freqs, coh) = coherence(
        &signal,
        &signal2,
        Some(fs),
        Some(nw),
        Some(k),
        None,
        Some(true),
    )?;

    // Get coherence at 50 Hz and 150 Hz
    let coh_idx_50hz = coh_freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 50.0).abs().partial_cmp(&(b - 50.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let coh_idx_150hz = coh_freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 150.0).abs().partial_cmp(&(b - 150.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let coh_idx_220hz = coh_freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 220.0).abs().partial_cmp(&(b - 220.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let coh_idx_250hz = coh_freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - 250.0).abs().partial_cmp(&(b - 250.0).abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!(
        "  Coherence at 50 Hz: {:.3} (identical components)",
        coh[coh_idx_50hz]
    );
    println!(
        "  Coherence at 150 Hz: {:.3} (phase-shifted components)",
        coh[coh_idx_150hz]
    );
    println!(
        "  Coherence at 220 Hz: {:.3} (noise band in signal 1 only)",
        coh[coh_idx_220hz]
    );
    println!(
        "  Coherence at 250 Hz: {:.3} (noise band in signal 2 only)",
        coh[coh_idx_250hz]
    );

    println!("\nMultitaper analysis complete!");

    Ok(())
}
