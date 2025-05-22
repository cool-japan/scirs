//! Example of using SIMD-accelerated FFT functions
//!
//! This example demonstrates the use of SIMD-accelerated FFT functions and
//! compares their performance with standard FFT implementations.
//! It also shows how to use the adaptive dispatchers that automatically select
//! the most efficient implementation for the current hardware.

use scirs2_fft::{fft, fft_adaptive, fft_simd, simd_support_available};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("SIMD-accelerated FFT Example");
    println!("----------------------------");

    // Check for SIMD support
    let simd_available = simd_support_available();
    println!("SIMD support available: {}", simd_available);

    // Generate a test signal - a sum of two sine waves
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.7 * (2.0 * PI * 5.0 * t).sin() + 0.3 * (2.0 * PI * 20.0 * t).sin()
        })
        .collect();

    println!("\nComputing FFT with various implementations...");

    // Standard FFT
    let start = Instant::now();
    let standard_fft = fft(&signal, None).unwrap();
    let standard_time = start.elapsed();
    println!("Standard FFT time: {:?}", standard_time);

    // SIMD-accelerated FFT (if available)
    if simd_available {
        let start = Instant::now();
        let simd_fft = fft_simd(&signal, None, None).unwrap();
        let simd_time = start.elapsed();
        println!("SIMD-accelerated FFT time: {:?}", simd_time);

        // Verify results match
        let max_diff = standard_fft
            .iter()
            .zip(simd_fft.iter())
            .map(|(s, a)| (s.re - a.re).abs() + (s.im - a.im).abs())
            .fold(0.0f64, |max_val, diff| max_val.max(diff));
        println!(
            "Maximum difference between standard and SIMD results: {:.2e}",
            max_diff
        );

        // Calculate speedup
        let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();
        println!("SIMD speedup: {:.2}x", speedup);
    }

    // Adaptive FFT (automatically selects the best implementation)
    let start = Instant::now();
    let _adaptive_fft = fft_adaptive(&signal, None, None).unwrap();
    let adaptive_time = start.elapsed();
    println!("Adaptive FFT time: {:?}", adaptive_time);

    // Performance benchmark for larger arrays
    println!("\nPerformance benchmark for larger arrays:");
    benchmark_performance();
}

fn benchmark_performance() {
    let sizes = [1024, 4096, 16384, 65536];

    for &size in &sizes {
        // Generate a signal
        let signal: Vec<f64> = (0..size)
            .map(|i| {
                let t = i as f64 / size as f64;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        println!("\nSignal size: {}", size);

        // Standard FFT
        let start = Instant::now();
        let _ = fft(&signal, None).unwrap();
        let standard_time = start.elapsed();
        println!("  Standard FFT time: {:?}", standard_time);

        // Adaptive FFT
        let start = Instant::now();
        let _ = fft_adaptive(&signal, None, None).unwrap();
        let adaptive_time = start.elapsed();
        println!("  Adaptive FFT time: {:?}", adaptive_time);

        if simd_support_available() {
            // SIMD FFT
            let start = Instant::now();
            let _ = fft_simd(&signal, None, None).unwrap();
            let simd_time = start.elapsed();
            println!("  SIMD FFT time: {:?}", simd_time);

            // Calculate speedup
            let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();
            println!("  SIMD speedup: {:.2}x", speedup);
        }

        // FFT with plan caching (first run)
        let start = Instant::now();
        let _ = fft(&signal, None).unwrap();
        let plan_time_first = start.elapsed();
        println!("  FFT with plan (first run): {:?}", plan_time_first);

        // FFT with plan caching (second run - should be faster due to cache)
        let start = Instant::now();
        let _ = fft(&signal, None).unwrap();
        let plan_time_second = start.elapsed();
        println!("  FFT with plan (cached): {:?}", plan_time_second);
    }

    println!("\nPlan cache statistics:");
    println!("  Note: Plan cache statistics are handled internally by the FFT implementation");
}
