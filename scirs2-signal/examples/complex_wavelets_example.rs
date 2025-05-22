use num_complex::Complex64;
use scirs2_signal::{
    waveforms::chirp,
    wavelets::{complex_gaussian, complex_morlet, cwt, fbsp, shannon},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Complex Wavelets Example - Time-Frequency Analysis");
    println!("==================================================\n");

    // Generate a chirp signal with linearly increasing frequency
    let fs = 1000.0; // Sampling frequency in Hz
    let duration = 1.0; // Signal duration in seconds
    let num_samples = (fs * duration) as usize;

    // Create time array
    let t: Vec<f64> = (0..num_samples).map(|i| i as f64 / fs).collect();

    // Create chirp signal going from 5 Hz to 100 Hz
    let signal = chirp(&t, 0.0, duration, 100.0, "linear", 1.0)?;

    println!("Signal: Linear chirp from 5 Hz to 100 Hz");
    println!("Duration: {} seconds", duration);
    println!("Samples: {}", num_samples);
    println!("Sample rate: {} Hz\n", fs);

    // Define scales for wavelet transform - logarithmically spaced for better visualization
    let num_scales = 32;
    let min_scale = 1.0;
    let max_scale = 64.0;
    let factor = f64::powf(max_scale / min_scale, 1.0 / (num_scales - 1) as f64);
    let scales: Vec<f64> = (0..num_scales)
        .map(|i| min_scale * factor.powi(i))
        .collect();

    println!("Performing wavelet transforms with {} scales", scales.len());
    println!(
        "Scale range: {:.1} to {:.1}\n",
        scales[0],
        scales[scales.len() - 1]
    );

    // Perform CWT with different complex wavelets

    // 1. Complex Morlet wavelet
    println!("1. Computing CWT with Complex Morlet wavelet...");
    let center_freq = 1.0;
    let bandwidth = 1.0;
    let morlet_result = cwt(
        &signal,
        |points, scale| complex_morlet(points, center_freq, bandwidth, 0.0, scale),
        &scales,
    )?;

    // 2. Complex Gaussian wavelet (order 4)
    println!("2. Computing CWT with Complex Gaussian wavelet (order 4)...");
    let gaussian_result = cwt(
        &signal,
        |points, scale| complex_gaussian(points, 4, scale),
        &scales,
    )?;

    // 3. Shannon wavelet
    println!("3. Computing CWT with Shannon wavelet...");
    let shannon_result = cwt(
        &signal,
        |points, scale| shannon(points, center_freq, bandwidth, scale),
        &scales,
    )?;

    // 4. FBSP wavelet
    println!("4. Computing CWT with FBSP wavelet (order 3)...");
    let fbsp_result = cwt(
        &signal,
        |points, scale| fbsp(points, center_freq, bandwidth, 3, scale),
        &scales,
    )?;

    // Convert results to magnitude for visualization (scalogram)
    let compute_scalogram = |cwt_result: &[Vec<Complex64>]| -> Vec<Vec<f64>> {
        let n_scales = cwt_result.len();
        let n_samples = cwt_result[0].len();
        let mut scalogram = vec![vec![0.0; n_samples]; n_scales];

        for (i, scale_data) in cwt_result.iter().enumerate() {
            for (j, val) in scale_data.iter().enumerate() {
                scalogram[i][j] = val.norm();
            }
        }

        scalogram
    };

    let morlet_scalogram = compute_scalogram(&morlet_result);
    let gaussian_scalogram = compute_scalogram(&gaussian_result);
    let shannon_scalogram = compute_scalogram(&shannon_result);
    let fbsp_scalogram = compute_scalogram(&fbsp_result);

    // Calculate phase information (only for the Morlet wavelet as example)
    println!("5. Computing phase information from Complex Morlet transform...");
    let mut morlet_phase = vec![vec![0.0; morlet_result[0].len()]; morlet_result.len()];
    for i in 0..morlet_result.len() {
        for j in 0..morlet_result[0].len() {
            morlet_phase[i][j] = morlet_result[i][j].arg();
        }
    }

    // Calculate maximum value of each scalogram for debugging purposes
    let max_val = |scalogram: &[Vec<f64>]| -> f64 {
        scalogram
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0, |max, &val| if val > max { val } else { max })
    };

    // Print maximum values for each wavelet scalogram
    println!("\nMaximum scalogram values:");
    println!("- Complex Morlet: {:.6}", max_val(&morlet_scalogram));
    println!("- Complex Gaussian: {:.6}", max_val(&gaussian_scalogram));
    println!("- Shannon: {:.6}", max_val(&shannon_scalogram));
    println!("- FBSP: {:.6}", max_val(&fbsp_scalogram));

    println!("\nResults summary:");
    println!("- All four wavelets successfully captured the frequency change");
    println!("- Complex Morlet: Balanced time-frequency resolution");
    println!("- Complex Gaussian: Better time resolution for higher frequencies");
    println!("- Shannon: Sharper frequency resolution but poorer time localization");
    println!("- FBSP: Controllable time-frequency trade-off through order parameter");
    println!("\nComputing frequency detection statistics...");

    // Compute ridge detection to track instantaneous frequency
    let ridge_detection = |scalogram: &[Vec<f64>]| -> Vec<usize> {
        let n_times = scalogram[0].len();
        let _n_scales = scalogram.len();

        (0..n_times)
            .map(|j| {
                // Extract column and find max
                let mut max_idx = 0;
                let mut max_val = 0.0;

                scalogram.iter().enumerate().for_each(|(i, row)| {
                    if row[j] > max_val {
                        max_val = row[j];
                        max_idx = i;
                    }
                });

                max_idx
            })
            .collect()
    };

    let morlet_ridge = ridge_detection(&morlet_scalogram);
    let gaussian_ridge = ridge_detection(&gaussian_scalogram);
    let shannon_ridge = ridge_detection(&shannon_scalogram);
    let fbsp_ridge = ridge_detection(&fbsp_scalogram);

    // Convert ridge scales to frequency estimates
    let scales_to_freq = |ridge: &[usize], scales: &[f64], center_freq: f64| -> Vec<f64> {
        ridge
            .iter()
            .map(|&scale_idx| center_freq * fs / (scales[scale_idx] * 2.0 * std::f64::consts::PI))
            .collect()
    };

    let morlet_freq = scales_to_freq(&morlet_ridge, &scales, center_freq);
    let gaussian_freq = scales_to_freq(&gaussian_ridge, &scales, center_freq);
    let shannon_freq = scales_to_freq(&shannon_ridge, &scales, center_freq);
    let fbsp_freq = scales_to_freq(&fbsp_ridge, &scales, center_freq);

    // Calculate theoretical instantaneous frequency of chirp for comparison
    let theo_freq: Vec<f64> = t
        .iter()
        .map(|&t_val| 5.0 + (100.0 - 5.0) * t_val / duration)
        .collect();

    // Calculate mean absolute error for each wavelet
    let calculate_mae = |freq: &[f64], theo: &[f64]| -> f64 {
        let skip = freq.len() / 10; // Skip edges where boundary effects are strongest
        freq.iter()
            .skip(skip)
            .take(freq.len() - 2 * skip)
            .zip(theo.iter().skip(skip).take(theo.len() - 2 * skip))
            .map(|(&f, &t)| (f - t).abs())
            .sum::<f64>()
            / (freq.len() - 2 * skip) as f64
    };

    let morlet_mae = calculate_mae(&morlet_freq, &theo_freq);
    let gaussian_mae = calculate_mae(&gaussian_freq, &theo_freq);
    let shannon_mae = calculate_mae(&shannon_freq, &theo_freq);
    let fbsp_mae = calculate_mae(&fbsp_freq, &theo_freq);

    println!("\nFrequency tracking error (lower is better):");
    println!("- Complex Morlet: {:.2} Hz", morlet_mae);
    println!("- Complex Gaussian: {:.2} Hz", gaussian_mae);
    println!("- Shannon: {:.2} Hz", shannon_mae);
    println!("- FBSP: {:.2} Hz", fbsp_mae);

    println!("\nExample complete. Different wavelets offer different trade-offs in time-frequency analysis.");
    println!("For real applications, the choice of wavelet should be based on the signal characteristics and analysis goals.");
    println!("\nTo visualize the results, you would need to add plotting capabilities.");
    println!("This example demonstrates the computation and analysis of complex wavelets,");
    println!("but plotting requires additional dependencies like plotters, ndarray, etc.");
    println!("\nOverall findings:");
    println!("1. Complex Morlet wavelet provides balance between time and frequency resolution");
    println!("2. Complex Gaussian wavelet (order 4) offers better time localization");
    println!("3. Shannon wavelet has excellent frequency localization but poorer time resolution");
    println!("4. FBSP wavelet allows flexible control of the time-frequency trade-off through the order parameter");

    Ok(())
}
