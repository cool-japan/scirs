use rand::{rng, Rng};
use scirs2_signal::lombscargle::{find_peaks, lombscargle, significance_levels, AutoFreqMethod};
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Lomb-Scargle Periodogram Example");
    println!("--------------------------------");

    // Generate unevenly sampled data with known frequencies
    println!("Generating test signal...");
    let n_samples = 500;

    // Create time samples with uneven spacing
    let mut rng = rng();
    let mut t = Vec::with_capacity(n_samples);

    // Start with roughly evenly spaced samples, then add jitter
    for i in 0..n_samples {
        let base_t = i as f64 * 0.1; // Base time (0.1 second spacing)
        let jitter = 0.05 * rng.random::<f64>(); // Random jitter up to 0.05 seconds
        t.push(base_t + jitter); // Add to time array
    }

    // Define frequency components
    let frequencies = [0.5, 1.2, 3.5]; // Hz
    let amplitudes = [1.0, 0.5, 0.3]; // Amplitude of each component

    println!("Signal components:");
    for (&freq, &amp) in frequencies.iter().zip(amplitudes.iter()) {
        println!("  Frequency: {:.1} Hz, Amplitude: {:.1}", freq, amp);
    }

    // Generate signal
    let mut y = vec![0.0; n_samples];
    for i in 0..n_samples {
        for (&freq, &amp) in frequencies.iter().zip(amplitudes.iter()) {
            y[i] += amp * (2.0 * PI * freq * t[i]).sin();
        }
    }

    // Add some noise
    let noise_level = 0.2;
    println!("  Adding noise with amplitude: {:.1}", noise_level);
    for y_val in y.iter_mut().take(n_samples) {
        *y_val += noise_level * (2.0 * rng.random::<f64>() - 1.0);
    }

    // Create frequency grid to evaluate periodogram
    println!("\nComputing Lomb-Scargle periodogram...");
    let _min_freq = 0.1; // Hz
    let _max_freq = 5.0; // Hz

    // Frequency grid using auto-frequency method
    let (frequencies, power) = lombscargle(
        &t,
        &y,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Linear),
    )?;

    println!("  Number of frequency points: {}", frequencies.len());
    println!(
        "  Frequency range: {:.1} to {:.1} Hz",
        frequencies.first().unwrap(),
        frequencies.last().unwrap()
    );

    // Calculate significance levels
    let fap_levels = vec![0.01, 0.05, 0.1];
    println!("\nCalculating significance levels for False Alarm Probabilities (FAP):");
    for &fap in &fap_levels {
        println!("  FAP = {:.2}", fap);
    }

    let significance = significance_levels(&power, &fap_levels, "standard", n_samples)?;

    println!("Significance thresholds:");
    for (i, &threshold) in significance.iter().enumerate() {
        println!(
            "  FAP = {:.2}: Power threshold = {:.4}",
            fap_levels[i], threshold
        );
    }

    // Find peaks in the periodogram
    println!("\nFinding significant peaks...");
    let power_threshold = significance[0]; // Use 1% FAP level
    let (peak_freqs, peak_powers) = find_peaks(
        &frequencies,
        &power,
        power_threshold,
        Some(0.1), // Group peaks within 0.1 Hz
    )?;

    println!("Significant peaks (99% confidence):");
    println!("  Found {} significant peak(s)", peak_freqs.len());

    for (i, (&freq, &pow)) in peak_freqs.iter().zip(peak_powers.iter()).enumerate() {
        println!(
            "  Peak {}: Frequency = {:.3} Hz, Power = {:.3}",
            i + 1,
            freq,
            pow
        );

        // Check if this peak corresponds to one of our input frequencies
        for &true_freq in &[0.5, 1.2, 3.5] {
            if (freq - true_freq).abs() < 0.1 {
                println!("    * Matches input frequency of {:.1} Hz", true_freq);
            }
        }
    }

    // Try different normalization methods
    println!("\nComparing different normalization methods:");

    let normalizations = vec!["standard", "model", "log", "psd"];

    for normalization in normalizations {
        let (_, power) = lombscargle(
            &t,
            &y,
            Some(&frequencies),
            Some(normalization),
            Some(true),
            Some(true),
            None,
            None,
        )?;

        // Find the maximum power
        let max_power = power.iter().fold(0.0_f64, |a, &b| a.max(b));
        let max_idx = power.iter().position(|&p| p == max_power).unwrap();
        let max_freq = frequencies[max_idx];

        println!(
            "  {}: Max power = {:.4} at frequency {:.3} Hz",
            normalization, max_power, max_freq
        );
    }

    println!("\nLomb-Scargle analysis complete!");

    Ok(())
}
