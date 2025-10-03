//! Example demonstrating the Chirp Z-Transform (CZT)
//!
//! This example shows:
//! - Basic CZT usage
//! - Zoom FFT for frequency analysis
//! - CZT along arbitrary contours

use plotly::{Plot, Scatter};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex;
use scirs2_fft::{czt, czt_points, zoom_fft, CZT};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test signal with multiple frequency components
    let n = 256;
    let t: Array1<f64> = Array1::linspace(0.0, 1.0, n);

    // Signal with frequencies at 10 Hz, 25 Hz, and 40 Hz
    let signal: Array1<Complex<f64>> = t.mapv(|ti| {
        let s = (2.0 * PI * 10.0 * ti).sin()
            + 0.5 * (2.0 * PI * 25.0 * ti).sin()
            + 0.3 * (2.0 * PI * 40.0 * ti).sin();
        Complex::new(s, 0.0)
    });

    println!("1. Comparing CZT with default parameters to FFT");

    // Basic CZT with default parameters (equivalent to FFT)
    let czt_result = czt(&signal, None, None, None, None)?;
    let fft_result = scirs2_fft::fft(&signal.to_vec(), None)?;

    // Convert results to magnitude spectrum
    let czt_mag: Array1<f64> = czt_result
        .view()
        .into_shape_with_order(n)?
        .mapv(|c| c.norm());
    let fft_mag: Vec<f64> = fft_result.iter().map(|c| c.norm()).collect();

    // Verify that CZT with default parameters matches FFT
    let mut max_diff = 0.0;
    for i in 0..n {
        let diff = (czt_mag[i] - fft_mag[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("Maximum difference between CZT and FFT: {max_diff:.2e}");

    println!("\n2. Using Zoom FFT to analyze a specific frequency range");

    // Use zoom FFT to focus on 20-30 Hz range
    let m = 64; // Number of output points
    let f0 = 20.0 / 100.0; // Normalized start frequency (20 Hz / Nyquist)
    let f1 = 30.0 / 100.0; // Normalized end frequency (30 Hz / Nyquist)

    let zoom_result = zoom_fft(&signal, m, f0, f1, Some(4.0))?;
    let zoom_mag: Array1<f64> = zoom_result
        .view()
        .into_shape_with_order(32)?
        .mapv(|c| c.norm());

    // Create frequency axis for zoom FFT
    let zoom_freqs: Array1<f64> = Array1::linspace(20.0, 30.0, m);

    // Plot zoom FFT results
    let trace = Scatter::new(zoom_freqs.to_vec(), zoom_mag.to_vec()).name("Zoom FFT (20-30 Hz)");

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.write_html("czt_zoom_fft.html");
    println!("Zoom FFT plot saved to czt_zoom_fft.html");

    println!("\n3. CZT along a spiral contour");

    // Define a logarithmic spiral in the z-plane
    let a = Complex::from_polar(0.9, PI / 6.0); // Starting point
    let w = Complex::from_polar(0.98, -0.1); // Spiral ratio

    // Compute CZT along the spiral
    let czt_spiral = czt(&signal, Some(128), Some(w), Some(a), None)?;
    let spiral_mag: Array1<f64> = czt_spiral
        .view()
        .into_shape_with_order(m)?
        .mapv(|c| c.norm());

    // Get the actual points on the spiral
    let spiral_points = czt_points(128, Some(a), Some(w));

    // Plot the spiral in the complex plane
    let real_parts: Vec<f64> = spiral_points.iter().map(|p| p.re).collect();
    let imag_parts: Vec<f64> = spiral_points.iter().map(|p| p.im).collect();

    let spiral_trace = Scatter::new(real_parts, imag_parts)
        .mode(plotly::common::Mode::LinesMarkers)
        .name("CZT Spiral Path");

    let mut spiral_plot = Plot::new();
    spiral_plot.add_trace(spiral_trace);

    // Add unit circle for reference
    let circle_points: Vec<f64> = (0..=100).map(|i| 2.0 * PI * i as f64 / 100.0).collect();
    let circle_x: Vec<f64> = circle_points.iter().map(|&theta| theta.cos()).collect();
    let circle_y: Vec<f64> = circle_points.iter().map(|&theta| theta.sin()).collect();

    let circle_trace = Scatter::new(circle_x, circle_y)
        .mode(plotly::common::Mode::Lines)
        .name("Unit Circle");

    spiral_plot.add_trace(circle_trace);
    spiral_plot.write_html("czt_spiral_path.html");
    println!("Spiral path plot saved to czt_spiral_path.html");

    // Plot magnitude along the spiral
    let spiral_mag_trace = Scatter::new((0..128).collect::<Vec<_>>(), spiral_mag.to_vec())
        .name("Magnitude along spiral");

    let mut mag_plot = Plot::new();
    mag_plot.add_trace(spiral_mag_trace);
    mag_plot.write_html("czt_spiral_magnitude.html");
    println!("Spiral magnitude plot saved to czt_spiral_magnitude.html");

    println!("\n4. Using CZT object for repeated transforms");

    // Create a CZT object for efficient repeated use
    let czt_obj = CZT::new(n, Some(n), None, None)?;

    // Apply to multiple signals
    let signal2: Array1<Complex<f64>> = t.mapv(|ti| {
        let s = (2.0 * PI * 15.0 * ti).sin();
        Complex::new(s, 0.0)
    });

    let _result1 = czt_obj.transform(&signal, None)?;
    let _result2 = czt_obj.transform(&signal2, None)?;

    println!("Applied CZT to {} different signals", 2);

    println!("\n5. Prime-length FFT using CZT");

    // CZT can efficiently compute FFTs of prime lengths
    let prime_n = 97; // Prime number
    let prime_signal: Array1<Complex<f64>> =
        Array1::linspace(0.0, 1.0, prime_n).mapv(|x| Complex::new(x, 0.0));

    let prime_czt = CZT::new(prime_n, None, None, None)?;
    let prime_result = prime_czt.transform(&prime_signal, None)?;

    println!("Computed {prime_n}-point FFT (prime length) using CZT");
    println!("Result length: {}", prime_result.len());

    Ok(())
}
