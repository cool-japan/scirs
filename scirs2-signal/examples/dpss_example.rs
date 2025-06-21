//! Example of DPSS (Discrete Prolate Spheroidal Sequence) windows for multitaper spectral estimation

use scirs2_signal::window::{dpss, dpss_windows, get_window};
use std::f64::consts::PI;

fn main() {
    println!("DPSS (Slepian) Windows Example");
    println!("==============================\n");

    // Example 1: Single DPSS window
    println!("1. Single DPSS Window");
    println!("--------------------");

    let n = 64;
    let nw = 2.5; // Time-bandwidth product
    let window = dpss(n, nw, None, true).unwrap();

    println!("Window length: {}", window.len());
    println!("Time-bandwidth product NW: {}", nw);
    println!("First 10 coefficients:");
    for (i, &coeff) in window.iter().take(10).enumerate() {
        println!("  w[{}] = {:.6}", i, coeff);
    }

    // Calculate energy concentration
    let total_energy = window.iter().map(|&x| x * x).sum::<f64>();
    let center_energy = window[16..48].iter().map(|&x| x * x).sum::<f64>(); // Central half
    let concentration = center_energy / total_energy;
    println!(
        "Energy concentration in central half: {:.3}%",
        concentration * 100.0
    );

    // Example 2: Multiple DPSS windows for multitaper analysis
    println!("\n\n2. Multiple DPSS Windows for Multitaper Analysis");
    println!("-----------------------------------------------");

    let n = 128;
    let nw = 4.0;
    let num_tapers = 7; // 2*NW - 1 = 7
    let windows = dpss_windows(n, nw, Some(num_tapers), true).unwrap();

    println!("Generated {} DPSS windows", windows.len());
    println!("Window length: {}", n);
    println!("Time-bandwidth product NW: {}", nw);

    // Show properties of each window
    for (i, window) in windows.iter().enumerate() {
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let energy = window.iter().map(|&x| x * x).sum::<f64>();
        println!("Window {}: max={:.4}, energy={:.4}", i, max_val, energy);
    }

    // Test orthogonality
    println!("\nOrthogonality matrix (should be near-diagonal):");
    for i in 0..num_tapers.min(5) {
        print!("Row {}: ", i);
        for j in 0..num_tapers.min(5) {
            let dot_product = windows[i]
                .iter()
                .zip(windows[j].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
            print!("{:6.3} ", dot_product);
        }
        println!();
    }

    // Example 3: Different time-bandwidth products
    println!("\n\n3. Effect of Different Time-Bandwidth Products");
    println!("---------------------------------------------");

    let n = 64;
    let nw_values = vec![1.5, 2.0, 3.0, 4.0];

    for &nw in &nw_values {
        let window = dpss(n, nw, None, true).unwrap();

        // Calculate effective bandwidth
        let mut weighted_freq_sum = 0.0;
        let mut weight_sum = 0.0;

        for (k, &w) in window.iter().enumerate() {
            let freq = k as f64 / n as f64;
            weighted_freq_sum += w * w * freq;
            weight_sum += w * w;
        }

        let effective_bandwidth = weighted_freq_sum / weight_sum;
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        println!(
            "NW={:.1}: max={:.4}, eff_bw={:.4}",
            nw, max_val, effective_bandwidth
        );
    }

    // Example 4: Comparison with other windows
    println!("\n\n4. Comparison with Other Windows");
    println!("-------------------------------");

    let n = 64;
    let windows_to_compare = vec![
        ("Hann", get_window("hann", n, false).unwrap()),
        ("Hamming", get_window("hamming", n, false).unwrap()),
        ("DPSS (NW=2.5)", dpss(n, 2.5, None, true).unwrap()),
        ("DPSS (NW=4.0)", dpss(n, 4.0, None, true).unwrap()),
    ];

    println!("Window comparison (64 points):");
    println!("Type          Max Value  Main Lobe  Side Lobe");
    println!("----------    ---------  ---------  ---------");

    for (name, window) in &windows_to_compare {
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        // Estimate main lobe width (simple approach)
        let center = n / 2;
        let mut main_lobe_width = 0;
        let threshold = max_val * 0.5; // -3dB point

        for i in 1..center {
            if window[center - i] < threshold {
                main_lobe_width = 2 * i;
                break;
            }
        }

        // Estimate maximum side lobe
        let mut max_side_lobe = 0.0f64;
        let exclude_range = main_lobe_width / 2;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if (i < center - exclude_range) || (i > center + exclude_range) {
                max_side_lobe = max_side_lobe.max(window[i].abs());
            }
        }

        println!(
            "{:<12}  {:8.4}   {:8}   {:8.4}",
            name, max_val, main_lobe_width, max_side_lobe
        );
    }

    // Example 5: Frequency domain properties
    println!("\n\n5. Frequency Domain Analysis");
    println!("---------------------------");

    let n = 64;
    let window = dpss(n, 3.0, None, true).unwrap();

    // Simple DFT for frequency response (would normally use FFT)
    let nfft = 256;
    let mut freq_response = vec![0.0; nfft];

    #[allow(clippy::needless_range_loop)]
    for k in 0..nfft {
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (j, &w) in window.iter().enumerate() {
            let phase = -2.0 * PI * (k as f64) * (j as f64) / (nfft as f64);
            real_sum += w * phase.cos();
            imag_sum += w * phase.sin();
        }

        freq_response[k] = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
    }

    // Find peak and -3dB bandwidth
    let max_response = freq_response.iter().fold(0.0f64, |a, &b| a.max(b));
    let threshold_3db = max_response / 2.0_f64.sqrt(); // -3dB

    let mut bandwidth_3db = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 1..nfft / 2 {
        if freq_response[i] < threshold_3db {
            bandwidth_3db = i;
            break;
        }
    }

    println!("DPSS window (NW=3.0) frequency response:");
    println!("Peak response: {:.4}", max_response);
    println!(
        "-3dB bandwidth: {} bins ({:.3} normalized frequency)",
        bandwidth_3db,
        bandwidth_3db as f64 / nfft as f64
    );

    // Show first few frequency bins
    println!("First 10 frequency response values:");
    for (i, &resp) in freq_response.iter().take(10).enumerate() {
        let freq_norm = i as f64 / nfft as f64;
        println!("  f={:.3}: |H(f)|={:.4}", freq_norm, resp);
    }

    println!("\n\nDPSS windows are optimal for multitaper spectral estimation because:");
    println!("- They maximize energy concentration within a given bandwidth");
    println!("- Multiple DPSS windows are approximately orthogonal");
    println!("- They provide excellent control over spectral leakage");
    println!("- The time-bandwidth product NW controls the trade-off between");
    println!("  frequency resolution and variance reduction");
}
