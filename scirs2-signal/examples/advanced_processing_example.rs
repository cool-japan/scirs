use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::{
    interpolate::polynomial::*,
    stft::{MemoryEfficientStft, MemoryEfficientStftConfig, StftConfig},
    window::{self, analysis::*, hann, kaiser},
};

fn main() {
    println!("Advanced Signal Processing Examples");
    println!("==================================");

    // 1. Memory-efficient STFT for large signals
    println!("\n1. Memory-efficient STFT processing...");
    demonstrate_memory_efficient_stft();

    // 2. Window analysis and design
    println!("\n2. Window analysis and design...");
    demonstrate_window_analysis();

    // 3. Polynomial interpolation methods
    println!("\n3. Polynomial interpolation methods...");
    demonstrate_polynomial_interpolation();

    println!("\nAll examples completed successfully!");
}

fn demonstrate_memory_efficient_stft() {
    // Create a long signal that would consume significant memory
    let fs = 8000.0;
    let duration = 10.0; // 10 seconds
    let n = (fs * duration) as usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Create a complex signal with multiple frequency components
    let signal: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(_i, &ti)| {
            let f1 = 100.0 + 50.0 * (ti / 2.0).sin(); // Frequency modulated component
            let f2 = 400.0; // Constant frequency component
            let f3 = 800.0 * if ti > 5.0 { 1.0 } else { 0.0 }; // Component that appears after 5s

            0.5 * (2.0 * PI * f1 * ti).sin()
                + 0.3 * (2.0 * PI * f2 * ti).sin()
                + 0.2 * (2.0 * PI * f3 * ti).sin()
                + 0.1 * (rand::random::<f64>() - 0.5) // Add some noise
        })
        .collect();

    println!("  Signal length: {} samples ({:.1} seconds)", n, duration);

    // Configure memory-efficient STFT
    let window_length = 512;
    let hop_size = 256;
    let window = hann(window_length, true).unwrap();

    let memory_config = MemoryEfficientStftConfig {
        max_memory_mb: 50,      // Limit memory usage to 50 MB
        chunk_size: Some(8000), // Process in 1-second chunks
        parallel: false,
        magnitude_only: true, // Save memory by storing only magnitudes
    };

    let stft_config = StftConfig::default();
    let mem_stft =
        MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config).unwrap();

    // Get memory estimate
    let memory_estimate = mem_stft.memory_estimate(signal.len());
    println!("  Estimated memory usage: {:.2} MB", memory_estimate);

    // Process the signal efficiently
    let start_time = std::time::Instant::now();
    let spectrogram = mem_stft.spectrogram_chunked(&signal).unwrap();
    let processing_time = start_time.elapsed();

    println!(
        "  Processing time: {:.2} seconds",
        processing_time.as_secs_f64()
    );
    println!(
        "  Spectrogram dimensions: {} x {}",
        spectrogram.shape()[0],
        spectrogram.shape()[1]
    );

    // Save a portion of the spectrogram
    save_spectrogram_sample(&spectrogram, "memory_efficient_spectrogram.csv");
    println!("  Saved spectrogram sample to memory_efficient_spectrogram.csv");
}

fn demonstrate_window_analysis() {
    // Create different windows for comparison
    let length = 64;
    let hann_win = hann(length, true).unwrap();
    let hamming_win = window::hamming(length, true).unwrap();
    let blackman_win = window::blackman(length, true).unwrap();
    let kaiser_win = kaiser(length, 8.0, true).unwrap(); // High beta for good sidelobe suppression

    // Analyze each window
    println!("  Analyzing window functions:");

    let windows = [
        ("Hann", hann_win.as_slice()),
        ("Hamming", hamming_win.as_slice()),
        ("Blackman", blackman_win.as_slice()),
        ("Kaiser(β=8)", kaiser_win.as_slice()),
    ];

    let comparison = compare_windows(&windows).unwrap();

    // Print analysis results
    for (name, analysis) in &comparison {
        println!("    {}:", name);
        println!("      Coherent gain: {:.3}", analysis.coherent_gain);
        println!("      NENBW: {:.3}", analysis.nenbw);
        println!(
            "      Scalloping loss: {:.2} dB",
            analysis.scalloping_loss_db
        );
        println!("      Max sidelobe: {:.1} dB", analysis.max_sidelobe_db);
        println!("      3dB bandwidth: {:.2} bins", analysis.bandwidth_3db);
        println!(
            "      Processing gain: {:.1} dB",
            analysis.processing_gain_db
        );
        println!();
    }

    // Design windows with specific requirements
    println!("  Designing windows for specific requirements:");

    let requirements = [
        ("Moderate sidelobes", -20.0),
        ("Low sidelobes", -40.0),
        ("Very low sidelobes", -80.0),
    ];

    for &(desc, sidelobe_req) in &requirements {
        let designed_window = design_window_with_constraints(64, sidelobe_req, None).unwrap();
        let analysis = analyze_window(&designed_window, None).unwrap();
        println!(
            "    {} ({:.0} dB): Achieved {:.1} dB sidelobes",
            desc, sidelobe_req, analysis.max_sidelobe_db
        );
    }

    // Save window comparison data
    save_window_comparison(&comparison, "window_analysis.csv");
    println!("  Saved window analysis to window_analysis.csv");
}

fn demonstrate_polynomial_interpolation() {
    // Create test data with a known function: f(x) = sin(2πx) + 0.5*sin(6πx)
    let x_data = vec![0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0];
    let y_data: Vec<f64> = x_data
        .iter()
        .map(|&x| (2.0 * PI * x).sin() + 0.5 * (6.0 * PI * x).sin())
        .collect();

    // Create fine grid for interpolation
    let x_fine: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let y_true: Vec<f64> = x_fine
        .iter()
        .map(|&x| (2.0 * PI * x).sin() + 0.5 * (6.0 * PI * x).sin())
        .collect();

    println!("  Comparing interpolation methods:");
    println!("    Data points: {}", x_data.len());
    println!("    Interpolation points: {}", x_fine.len());

    // Test different interpolation methods
    let methods = [
        ("Lagrange", lagrange_interpolate(&x_data, &y_data, &x_fine)),
        ("Newton", newton_interpolate(&x_data, &y_data, &x_fine)),
    ];

    for (name, result) in methods {
        match result {
            Ok(y_interp) => {
                let rmse = calculate_rmse(&y_true, &y_interp);
                let max_error = calculate_max_error(&y_true, &y_interp);
                println!("    {} interpolation:", name);
                println!("      RMSE: {:.6}", rmse);
                println!("      Max error: {:.6}", max_error);
            }
            Err(e) => {
                println!("    {} interpolation failed: {:?}", name, e);
            }
        }
    }

    // Test Newton polynomial interpolation
    println!("\n  Testing Newton polynomial interpolation:");
    match newton_interpolate(&x_data, &y_data, &x_fine) {
        Ok(y_newton) => {
            let rmse = calculate_rmse(&y_true, &y_newton);
            let max_error = calculate_max_error(&y_true, &y_newton);
            println!("    Newton polynomial interpolation:");
            println!("      RMSE: {:.6}", rmse);
            println!("      Max error: {:.6}", max_error);
            println!("      Using {} data points", x_data.len());
        }
        Err(e) => {
            println!("    Newton interpolation failed: {:?}", e);
        }
    }

    // Test Lagrange polynomial interpolation
    match lagrange_interpolate(&x_data, &y_data, &x_fine) {
        Ok(y_lagrange) => {
            let rmse = calculate_rmse(&y_true, &y_lagrange);
            let max_error = calculate_max_error(&y_true, &y_lagrange);
            println!("    Lagrange polynomial interpolation:");
            println!("      RMSE: {:.6}", rmse);
            println!("      Max error: {:.6}", max_error);
        }
        Err(e) => {
            println!("    Lagrange interpolation failed: {:?}", e);
        }
    }

    // Save interpolation comparison
    if let Ok(y_lagrange) = lagrange_interpolate(&x_data, &y_data, &x_fine) {
        save_interpolation_comparison(
            &x_fine,
            &y_true,
            &y_lagrange,
            "interpolation_comparison.csv",
        );
        println!("  Saved interpolation comparison to interpolation_comparison.csv");
    }
}

// Helper functions

fn save_spectrogram_sample(spectrogram: &ndarray::Array2<f64>, filename: &str) {
    let mut file = File::create(filename).expect("Could not create file");

    // Save first 50 frequency bins and every 10th time frame for visualization
    writeln!(file, "freq_bin,time_frame,magnitude").expect("Failed to write header");

    let freq_step = spectrogram.shape()[0] / 50;
    let time_step = 10;

    for f in (0..spectrogram.shape()[0]).step_by(freq_step.max(1)) {
        for t in (0..spectrogram.shape()[1]).step_by(time_step) {
            writeln!(file, "{},{},{:.6}", f, t, spectrogram[[f, t]]).expect("Failed to write data");
        }
    }
}

fn save_window_comparison(comparison: &[(String, WindowAnalysis)], filename: &str) {
    let mut file = File::create(filename).expect("Could not create file");

    writeln!(file, "window,coherent_gain,nenbw,scalloping_loss_db,max_sidelobe_db,bandwidth_3db,processing_gain_db")
        .expect("Failed to write header");

    for (name, analysis) in comparison {
        writeln!(
            file,
            "{},{:.3},{:.3},{:.2},{:.1},{:.2},{:.1}",
            name,
            analysis.coherent_gain,
            analysis.nenbw,
            analysis.scalloping_loss_db,
            analysis.max_sidelobe_db,
            analysis.bandwidth_3db,
            analysis.processing_gain_db
        )
        .expect("Failed to write data");
    }
}

fn save_interpolation_comparison(x: &[f64], y_true: &[f64], y_interp: &[f64], filename: &str) {
    let mut file = File::create(filename).expect("Could not create file");

    writeln!(file, "x,y_true,y_interpolated,error").expect("Failed to write header");

    for i in 0..x.len() {
        let error = y_true[i] - y_interp[i];
        writeln!(
            file,
            "{:.3},{:.6},{:.6},{:.6}",
            x[i], y_true[i], y_interp[i], error
        )
        .expect("Failed to write data");
    }
}

fn calculate_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mse: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum::<f64>()
        / y_true.len() as f64;
    mse.sqrt()
}

fn calculate_max_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).abs())
        .fold(0.0, f64::max)
}
