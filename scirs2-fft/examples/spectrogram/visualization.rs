//! Spectrogram visualization examples
//!
//! This example demonstrates various ways to visualize time-frequency data
//! including spectrograms and waterfall plots.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use scirs2_fft::{
    apply_colormap, fft, spectrogram, spectrogram_normalized, waterfall_3d, waterfall_lines,
    window::Window,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spectrogram Visualization Example");
    println!("---------------------------------\n");

    // Generate a test signal with a chirp (frequency sweep)
    let fs = 1000.0; // Sampling frequency: 1 kHz
    let t_max = 10.0; // 10 second signal
    let n_samples = (t_max * fs) as usize;

    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
    
    // Create a linear chirp signal (frequency sweep from 0 to 200 Hz)
    let mut signal = Vec::with_capacity(n_samples);
    
    for &ti in &t {
        // Linear chirp from 10 Hz to 200 Hz over the duration
        let freq = 10.0 + (200.0 - 10.0) * (ti / t_max);
        let phase = 2.0 * PI * (10.0 * ti + 0.5 * (200.0 - 10.0) / t_max * ti.powi(2));
        signal.push(phase.sin());
    }
    
    // Add some intermittent tones
    for (i, s) in signal.iter_mut().enumerate() {
        let ti = i as f64 / fs;
        
        // Add a 250 Hz tone that pulses on and off
        if (ti * 0.5).sin() > 0.0 {
            *s += 0.5 * (2.0 * PI * 250.0 * ti).sin();
        }
        
        // Add a 300 Hz tone that appears in the middle
        if ti > 3.0 && ti < 7.0 {
            let window = 0.5 * (1.0 - ((ti - 5.0) * PI / 2.0).cos());
            *s += 0.7 * window * (2.0 * PI * 300.0 * ti).sin();
        }
    }
    
    println!("Generated chirp signal with intermittent tones");
    println!("  - Duration: {} seconds", t_max);
    println!("  - Sample rate: {} Hz", fs);
    println!("  - Samples: {}", n_samples);
    
    // === Basic Spectrogram ===
    println!("\n1. Basic Spectrogram");
    
    // Use a moderate-size FFT window
    let nperseg = 256;
    let noverlap = 192;
    
    let (freqs, times, spect) = spectrogram(
        &signal,
        Some(fs),
        Some(Window::Hann),
        Some(nperseg),
        Some(noverlap),
        None,  // default nfft
        None,  // default scaling
        Some(true),  // return one-sided
        Some("power"), // power spectrogram
    )?;
    
    println!("  Computed spectrogram with:");
    println!("  - Window: Hann");
    println!("  - Window size: {} samples", nperseg);
    println!("  - Overlap: {} samples", noverlap);
    println!("  - Frequency bins: {}", freqs.len());
    println!("  - Time frames: {}", times.len());
    
    // Convert to dB scale (10*log10) with reasonable floor
    let db_floor = -80.0;
    let mut spect_db = Array2::zeros(spect.dim());
    for ((i, j), &val) in spect.indexed_iter() {
        let db_val = 10.0 * val.log10();
        spect_db[[i, j]] = if db_val.is_finite() { db_val.max(db_floor) } else { db_floor };
    }
    
    // Normalize to [0, 1] range for visualization
    let spect_norm = spectrogram_normalized(&signal, 
        Some(fs),
        Some(Window::Hann),
        Some(nperseg),
        Some(noverlap),
        None,
        None,
        Some(true))?;
    
    println!("  Normalized spectrogram data range: [0.0, 1.0]");
    
    // === Colormap Visualization ===
    println!("\n2. Colormap Visualization");
    
    // Create flattened amplitude data for colormap demo
    let n_test = 100;
    let test_amplitudes: Vec<f64> = (0..n_test).map(|i| i as f64 / (n_test as f64 - 1.0)).collect();
    
    let colormaps = vec!["jet", "viridis", "plasma", "grayscale", "hot"];
    
    println!("  Available colormaps:");
    for &cmap in &colormaps {
        let colors = apply_colormap(&test_amplitudes, cmap)?;
        println!("  - {}: {} RGB triplets", cmap, colors.shape()[0]);
    }
    
    // === Waterfall Plot Data ===
    println!("\n3. Waterfall Plot Data");
    
    // Create waterfall plot data
    let (wf_freqs, wf_times, wf_vals) = waterfall_lines(
        &signal,
        Some(fs),
        Some(nperseg),
        Some(noverlap),
        None,
        Some("dB"),
        Some("jet"),
        None, // Default dB range
    )?;
    
    println!("  Generated waterfall plot data:");
    println!("  - Frequency bins: {}", wf_freqs.len());
    println!("  - Time frames: {}", wf_times.len());
    println!("  - 3D points: {} × {} × 3", wf_vals.shape()[0], wf_vals.shape()[1]);
    
    // === 3D Surface Data ===
    println!("\n4. 3D Surface Data");
    
    // Generate data for a 3D surface plot
    let (surf_x, surf_y, surf_z, surf_c) = waterfall_3d(
        &signal,
        Some(fs),
        Some(nperseg),
        Some(noverlap),
        None,
        Some("dB"),
        Some("viridis"),
        Some(-60.0), // 60dB dynamic range
    )?;
    
    println!("  Generated 3D surface data:");
    println!("  - X points (time): {}", surf_x.shape()[0]);
    println!("  - Y points (frequency): {}", surf_y.shape()[0]);
    println!("  - Z mesh: {} × {}", surf_z.shape()[0], surf_z.shape()[1]);
    println!("  - Colors: {} × {} × 3", surf_c.shape()[0], surf_c.shape()[1]);
    
    // === Time-Varying Spectrum Analysis ===
    println!("\n5. Time-Varying Spectrum Analysis");
    
    // Extract time slices for analysis
    let quarter_idx = times.len() / 4;
    let mid_idx = times.len() / 2;
    let three_quarter_idx = 3 * times.len() / 4;
    
    let times_of_interest = vec![
        (0, "Start"),
        (quarter_idx, "25%"),
        (mid_idx, "Middle"),
        (three_quarter_idx, "75%"),
        (times.len() - 1, "End")
    ];
    
    println!("  Analyzing spectrum at specific time points:");
    
    for &(idx, label) in &times_of_interest {
        let time = times[idx];
        
        // Find the top 3 frequency components at this time
        let mut freq_powers: Vec<(usize, f64)> = freqs.iter()
            .zip(spect.column(idx).iter())
            .enumerate()
            .map(|(i, (_, &power))| (i, power))
            .collect();
        
        // Sort by power (descending)
        freq_powers.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  * Time {} ({:.2} s):", label, time);
        for i in 0..3.min(freq_powers.len()) {
            let (idx, power) = freq_powers[i];
            let freq = freqs[idx];
            println!("    - {:.1} Hz: {:.1} dB", freq, 10.0 * power.log10());
        }
    }
    
    println!("\nSpectrogram visualization complete!");
    Ok(())
}
