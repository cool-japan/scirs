use plotly::common::Mode;
use plotly::{Plot, Scatter};
use rand::Rng;
use scirs2_signal::dwt::Wavelet;
use scirs2_signal::swt::{iswt, swt};
use scirs2_signal::waveforms::chirp;

fn main() {
    // Generate a chirp signal
    let fs = 1000.0; // Sample rate in Hz
    let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    let signal = chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap();

    // Add some noise to the signal
    let mut rng = rand::rng();
    let noisy_signal = signal
        .iter()
        .map(|&x| x + 0.1 * rng.random_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    // Perform SWT decomposition with 3 levels
    let (details, approx) = swt(&noisy_signal, Wavelet::DB(4), 3, None).unwrap();

    // Modify detail coefficients to denoise (simple hard threshold)
    let mut modified_details = details.clone();
    for (level, detail_level) in modified_details.iter_mut().enumerate() {
        let threshold = 0.2 / (level + 1) as f64; // Decreasing threshold with level
        for val in detail_level.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
            }
        }
    }

    // Reconstruct the signal using modified coefficients
    let denoised_signal = iswt(&modified_details, &approx, Wavelet::DB(4)).unwrap();

    // Reconstruct the signal using original coefficients as a reference
    let reconstructed_signal = iswt(&details, &approx, Wavelet::DB(4)).unwrap();

    // Plot the results
    let mut plot = Plot::new();

    // Plot the original signal
    let original_trace = Scatter::new(t.clone(), signal.clone())
        .name("Original Signal")
        .mode(Mode::Lines);

    // Plot the noisy signal
    let noisy_trace = Scatter::new(t.clone(), noisy_signal.clone())
        .name("Noisy Signal")
        .mode(Mode::Lines);

    // Plot the denoised signal
    let denoised_trace = Scatter::new(t.clone(), denoised_signal)
        .name("Denoised Signal")
        .mode(Mode::Lines);

    // Add traces to the plot
    plot.add_trace(original_trace);
    plot.add_trace(noisy_trace);
    plot.add_trace(denoised_trace);

    // Create simple layout with title and axis labels
    // Note: We're using a much simpler approach that avoids relying on internal implementation details
    let layout = plotly::Layout::new().title("SWT Denoising Example");

    plot.set_layout(layout);

    // Save the plot to an HTML file
    plot.write_html("swt_denoising_example.html");
    println!("Plot saved to swt_denoising_example.html");

    // Also create a plot showing the wavelet coefficients
    let mut coeffs_plot = Plot::new();

    // Final approximation
    let approx_trace = Scatter::new(
        (0..approx.len()).map(|x| x as f64).collect::<Vec<f64>>(),
        approx.clone(),
    )
    .name("Approximation (Level 3)")
    .mode(Mode::Lines);

    // Detail coefficients at each level
    for (i, detail) in details.iter().enumerate() {
        let detail_trace = Scatter::new(
            (0..detail.len()).map(|x| x as f64).collect::<Vec<f64>>(),
            detail.clone(),
        )
        .name(format!("Detail (Level {})", i + 1))
        .mode(Mode::Lines);

        coeffs_plot.add_trace(detail_trace);
    }

    coeffs_plot.add_trace(approx_trace);

    // Add layout information for coefficients plot
    let coeffs_layout = plotly::Layout::new().title("SWT Coefficients");

    coeffs_plot.set_layout(coeffs_layout);

    // Save the coefficients plot to an HTML file
    coeffs_plot.write_html("swt_coefficients_example.html");
    println!("Coefficients plot saved to swt_coefficients_example.html");

    // Print information about the transform
    println!("Stationary Wavelet Transform with DB4 wavelet, 3 levels");
    println!("Original signal length: {}", signal.len());
    println!("Number of detail coefficient arrays: {}", details.len());
    for (i, detail) in details.iter().enumerate() {
        println!("  Detail level {}: {} coefficients", i + 1, detail.len());
    }
    println!("Final approximation: {} coefficients", approx.len());

    // Calculate and print reconstruction error
    let mut mse = 0.0;
    for (x, y) in signal.iter().zip(reconstructed_signal.iter()) {
        mse += (x - y).powi(2);
    }
    mse /= signal.len() as f64;
    println!("Reconstruction Mean Squared Error: {:.10e}", mse);
}
