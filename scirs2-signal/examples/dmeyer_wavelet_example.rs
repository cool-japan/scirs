use plotly::common::Mode;
use plotly::layout::Layout;
use plotly::{Plot, Scatter};
use rand::Rng;
use scirs2_signal::dwt::{wavedec, waverec, Wavelet};
use scirs2_signal::waveforms::chirp;

fn main() {
    // Generate a chirp signal with increasing frequency
    let fs = 1000.0; // Sample rate in Hz
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>(); // 1024 samples
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();

    // Add some noise to the signal
    let mut rng = rand::rng();
    let noisy_signal = signal
        .iter()
        .map(|&x| x + 0.1 * rng.random_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    // Compare Meyer and DMeyer wavelets for signal denoising
    let wavelets = vec![
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "Discrete Meyer (DMeyer)"),
    ];

    // Create a plot for the comparison
    let mut plot = Plot::new();

    // Add original signal
    let original_trace = Scatter::new(t.to_vec(), signal.to_vec())
        .name("Original Signal")
        .mode(Mode::Lines);
    plot.add_trace(original_trace);

    // Add noisy signal
    let noisy_trace = Scatter::new(t.to_vec(), noisy_signal.to_vec())
        .name("Noisy Signal")
        .mode(Mode::Lines);
    plot.add_trace(noisy_trace);

    // Create a coefficients plot for Meyer vs DMeyer comparison
    let mut coeffs_plot = Plot::new();

    // Storage for performance metrics
    let mut computation_times = Vec::new();

    // Test each wavelet
    for (wavelet, name) in wavelets {
        // Measure decomposition time
        let start_time = std::time::Instant::now();

        // Perform DWT with 3 levels of decomposition
        let coeffs = wavedec(&noisy_signal, wavelet, Some(3), None).unwrap();

        let decomp_time = start_time.elapsed();
        println!("{} decomposition time: {:?}", name, decomp_time);

        // Extract the coefficients
        let approx = &coeffs[0]; // Final approximation (cA3)
        let detail3 = &coeffs[1]; // Detail level 3 (cD3)
        let detail2 = &coeffs[2]; // Detail level 2 (cD2)
        let detail1 = &coeffs[3]; // Detail level 1 (cD1)

        // Print coefficient info
        println!(
            "{} coefficient lengths: cA3={}, cD3={}, cD2={}, cD1={}",
            name,
            approx.len(),
            detail3.len(),
            detail2.len(),
            detail1.len()
        );

        // Add coefficient traces to comparison plot (we'll just show cD1 for comparison)
        let x_detail1 = (0..detail1.len()).map(|x| x as f64).collect::<Vec<f64>>();
        let detail1_trace = Scatter::new(x_detail1, detail1.clone())
            .name(format!("Detail L1 ({})", name))
            .mode(Mode::Lines);

        coeffs_plot.add_trace(detail1_trace);

        // Modify coefficients for denoising (simple hard thresholding)
        let mut denoised_coeffs = coeffs.clone();

        // Apply different thresholds to different levels
        apply_threshold(&mut denoised_coeffs[1], 0.2); // Less thresholding for coarser scales
        apply_threshold(&mut denoised_coeffs[2], 0.3);
        apply_threshold(&mut denoised_coeffs[3], 0.4); // More thresholding for finer scales

        // Measure reconstruction time
        let start_time = std::time::Instant::now();

        // Reconstruct the denoised signal
        let denoised_signal = waverec(&denoised_coeffs, wavelet).unwrap();

        let recon_time = start_time.elapsed();
        println!("{} reconstruction time: {:?}", name, recon_time);

        // Store metrics
        computation_times.push((
            name,
            decomp_time.as_micros() as f64 / 1000.0,
            recon_time.as_micros() as f64 / 1000.0,
        ));

        // Calculate MSE
        let mse: f64 = signal
            .iter()
            .zip(denoised_signal.iter())
            .map(|(&s, &d)| (s - d).powi(2))
            .sum::<f64>()
            / signal.len() as f64;

        println!("{} denoising MSE: {:.6e}", name, mse);

        // Add denoised signal to plot
        let denoised_trace = Scatter::new(t.to_vec(), denoised_signal)
            .name(format!("Denoised ({})", name))
            .mode(Mode::Lines);
        plot.add_trace(denoised_trace);
    }

    // Print performance comparison
    println!("\nPerformance Comparison (milliseconds):");
    println!(
        "{:<15} {:>12} {:>12}",
        "Wavelet", "Decomp Time", "Recon Time"
    );
    println!("{:-<15} {:->12} {:->12}", "", "", "");
    for (name, decomp_time, recon_time) in computation_times {
        println!("{:<15} {:>12.3} {:>12.3}", name, decomp_time, recon_time);
    }

    // Set layout for denoising comparison plot
    let layout = Layout::new().title("Meyer vs. DMeyer Wavelet Denoising Comparison");

    plot.set_layout(layout);

    // Save denoising plot
    plot.write_html("meyer_vs_dmeyer_denoising.html");
    println!("Denoising plot saved to meyer_vs_dmeyer_denoising.html");

    // Set layout for coefficients comparison plot
    let coeffs_layout = Layout::new().title("Meyer vs. DMeyer Level 1 Detail Coefficients");

    coeffs_plot.set_layout(coeffs_layout);

    // Save coefficients plot
    coeffs_plot.write_html("meyer_vs_dmeyer_coefficients.html");
    println!("Coefficients plot saved to meyer_vs_dmeyer_coefficients.html");
}

// Helper function to apply thresholding to coefficients
fn apply_threshold(coeffs: &mut [f64], threshold: f64) {
    for val in coeffs.iter_mut() {
        if val.abs() < threshold {
            *val = 0.0;
        }
    }
}
