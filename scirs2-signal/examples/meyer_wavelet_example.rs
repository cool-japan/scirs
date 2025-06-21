use plotly::common::Mode;
use plotly::{Plot, Scatter};
use rand::Rng;
use scirs2_signal::dwt::{wavedec, waverec, Wavelet};
use scirs2_signal::waveforms::chirp;

fn main() {
    // Generate a chirp signal
    let fs = 1000.0; // Sample rate in Hz
    let t = (0..1024).map(|i| i as f64 / fs).collect::<Vec<f64>>(); // 1024 samples
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();

    // Add some noise to the signal
    let mut rng = rand::rng();
    let noisy_signal = signal
        .iter()
        .map(|&x| x + 0.1 * rng.random_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    // Perform DWT with Meyer wavelet (3 levels)
    let coeffs = wavedec(&noisy_signal, Wavelet::Meyer, Some(3), None).unwrap();

    // Extract the coefficients
    let approx = &coeffs[0]; // Final approximation (cA3)
    let detail3 = &coeffs[1]; // Detail level 3 (cD3)
    let detail2 = &coeffs[2]; // Detail level 2 (cD2)
    let detail1 = &coeffs[3]; // Detail level 1 (cD1)

    println!(
        "Coefficient lengths: cA3={}, cD3={}, cD2={}, cD1={}",
        approx.len(),
        detail3.len(),
        detail2.len(),
        detail1.len()
    );

    // Modify coefficients for denoising (simple hard thresholding)
    let mut denoised_coeffs = coeffs.clone();

    // Apply different thresholds to different levels
    apply_threshold(&mut denoised_coeffs[1], 0.2); // Less thresholding for coarser scales
    apply_threshold(&mut denoised_coeffs[2], 0.3);
    apply_threshold(&mut denoised_coeffs[3], 0.4); // More thresholding for finer scales

    // Reconstruct the denoised signal
    let denoised_signal = waverec(&denoised_coeffs, Wavelet::Meyer).unwrap();

    // Plot the results
    let mut plot = Plot::new();

    // Original signal
    let original_trace = Scatter::new(t.clone(), signal.clone().to_vec())
        .name("Original Signal")
        .mode(Mode::Lines);

    // Noisy signal
    let noisy_trace = Scatter::new(t.clone(), noisy_signal.clone().to_vec())
        .name("Noisy Signal")
        .mode(Mode::Lines);

    // Denoised signal
    let denoised_trace = Scatter::new(t.clone(), denoised_signal.to_vec())
        .name("Denoised Signal (Meyer)")
        .mode(Mode::Lines);

    // Add traces to plot
    plot.add_trace(original_trace);
    plot.add_trace(noisy_trace);
    plot.add_trace(denoised_trace);

    // Set layout
    let mut layout = plotly::Layout::new();
    // Use proper layout API for plotly
    layout = layout.title("Meyer Wavelet Denoising Example");

    plot.set_layout(layout);

    // Save to HTML file
    plot.write_html("meyer_wavelet_denoising.html");
    println!("Plot saved to meyer_wavelet_denoising.html");

    // Also create a plot of the wavelet coefficients
    let mut coeffs_plot = Plot::new();

    // Approximation coefficients
    let x_approx = (0..approx.len()).map(|x| x as f64).collect::<Vec<f64>>();
    let approx_trace = Scatter::new(x_approx, approx.clone())
        .name("Approximation (Level 3)")
        .mode(Mode::Lines);

    // Detail coefficients
    let x_detail3 = (0..detail3.len()).map(|x| x as f64).collect::<Vec<f64>>();
    let detail3_trace = Scatter::new(x_detail3, detail3.clone())
        .name("Detail (Level 3)")
        .mode(Mode::Lines);

    let x_detail2 = (0..detail2.len()).map(|x| x as f64).collect::<Vec<f64>>();
    let detail2_trace = Scatter::new(x_detail2, detail2.clone())
        .name("Detail (Level 2)")
        .mode(Mode::Lines);

    let x_detail1 = (0..detail1.len()).map(|x| x as f64).collect::<Vec<f64>>();
    let detail1_trace = Scatter::new(x_detail1, detail1.clone())
        .name("Detail (Level 1)")
        .mode(Mode::Lines);

    // Add coefficient traces to plot
    coeffs_plot.add_trace(approx_trace);
    coeffs_plot.add_trace(detail3_trace);
    coeffs_plot.add_trace(detail2_trace);
    coeffs_plot.add_trace(detail1_trace);

    // Set layout for coefficients plot
    let coeffs_layout = plotly::Layout::new().title("Meyer Wavelet Coefficients");

    coeffs_plot.set_layout(coeffs_layout);

    // Save coefficients plot
    coeffs_plot.write_html("meyer_wavelet_coefficients.html");
    println!("Coefficients plot saved to meyer_wavelet_coefficients.html");

    // Compare with other wavelets
    compare_wavelets(&signal, &noisy_signal, &t);
}

// Helper function to apply thresholding to coefficients
fn apply_threshold(coeffs: &mut [f64], threshold: f64) {
    for val in coeffs.iter_mut() {
        if val.abs() < threshold {
            *val = 0.0;
        }
    }
}

// Compare the Meyer wavelet with other wavelet families
fn compare_wavelets(signal: &[f64], noisy_signal: &[f64], t: &[f64]) {
    // Wavelet families to compare
    let wavelets = vec![
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "Daubechies-4"),
        (Wavelet::Sym(4), "Symlet-4"),
        (Wavelet::Coif(3), "Coiflet-3"),
        (Wavelet::Meyer, "Meyer"),
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

    // Process with each wavelet
    for (wavelet, name) in wavelets {
        // Decompose
        let coeffs = wavedec(noisy_signal, wavelet, Some(3), None).unwrap();

        // Apply thresholding
        let mut denoised_coeffs = coeffs.clone();
        apply_threshold(&mut denoised_coeffs[1], 0.2);
        apply_threshold(&mut denoised_coeffs[2], 0.3);
        apply_threshold(&mut denoised_coeffs[3], 0.4);

        // Reconstruct
        let denoised = waverec(&denoised_coeffs, wavelet).unwrap();

        // Calculate MSE
        let mse: f64 = signal
            .iter()
            .zip(denoised.iter())
            .map(|(&s, &d)| (s - d).powi(2))
            .sum::<f64>()
            / signal.len() as f64;

        println!("{} wavelet MSE: {:.6e}", name, mse);

        // Add trace to comparison plot
        let denoised_trace = Scatter::new(t.to_vec(), denoised)
            .name(format!("Denoised ({})", name))
            .mode(Mode::Lines);
        plot.add_trace(denoised_trace);
    }

    // Set layout for comparison plot
    let comp_layout = plotly::Layout::new().title("Wavelet Family Comparison for Denoising");

    plot.set_layout(comp_layout);

    // Save comparison plot
    plot.write_html("wavelet_comparison.html");
    println!("Comparison plot saved to wavelet_comparison.html");
}
