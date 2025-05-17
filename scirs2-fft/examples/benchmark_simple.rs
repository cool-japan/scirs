//! Simple benchmark example to test FFT performance

use num_complex::Complex64;
use scirs2_fft::{fft, frft, rfft};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("Simple FFT Benchmark");
    println!("===================");

    // Test different sizes
    let sizes = vec![64, 256, 1024, 4096];

    for &size in &sizes {
        println!("\nSize: {}", size);

        // Generate test signal
        let signal: Vec<f64> = (0..size)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())
            .collect();

        // Complex signal
        let complex_signal: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Benchmark FFT
        let start = Instant::now();
        let _ = fft(&complex_signal, None).unwrap();
        let fft_time = start.elapsed();
        println!("  FFT:  {:?}", fft_time);

        // Benchmark RFFT
        let start = Instant::now();
        let _ = rfft(&signal, None).unwrap();
        let rfft_time = start.elapsed();
        println!("  RFFT: {:?}", rfft_time);

        // Benchmark FrFT
        let start = Instant::now();
        let _ = frft(&signal, 0.5, None).unwrap();
        let frft_time = start.elapsed();
        println!("  FrFT: {:?}", frft_time);
    }

    // Compare with memory-efficient versions
    use scirs2_fft::memory_efficient::{fft_inplace, FftMode};

    println!("\n\nMemory-Efficient Comparison (size=4096)");
    println!("======================================");

    let size = 4096;
    let mut complex_signal: Vec<Complex64> = (0..size)
        .map(|i| {
            let x = (2.0 * PI * 10.0 * i as f64 / size as f64).sin();
            Complex64::new(x, 0.0)
        })
        .collect();

    // Regular FFT
    let signal_copy = complex_signal.clone();
    let start = Instant::now();
    let _ = fft(&signal_copy, None).unwrap();
    let regular_time = start.elapsed();
    println!("Regular FFT: {:?}", regular_time);

    // In-place FFT
    let mut output_buffer = vec![Complex64::new(0.0, 0.0); size];
    let start = Instant::now();
    fft_inplace(
        &mut complex_signal,
        &mut output_buffer,
        FftMode::Forward,
        false,
    )
    .unwrap();
    let inplace_time = start.elapsed();
    println!("In-place FFT: {:?}", inplace_time);

    let speedup = regular_time.as_secs_f64() / inplace_time.as_secs_f64();
    if speedup > 1.0 {
        println!("\nIn-place is {:.2}x faster", speedup);
    } else {
        println!("\nRegular is {:.2}x faster", 1.0 / speedup);
    }
}
