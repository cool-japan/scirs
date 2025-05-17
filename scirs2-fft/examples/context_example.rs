//! Example demonstrating FFT context managers
//!
//! This example shows how to use context managers to temporarily change FFT settings.

use scirs2_fft::{
    fft, fft_context, get_backend_name, get_global_cache, get_workers, with_backend,
    with_fft_settings, with_workers, without_cache,
};
use std::time::Instant;

fn main() {
    println!("FFT Context Managers Example");
    println!("===========================");
    println!();

    // Create test signal
    let signal: Vec<f64> = (0..1024).map(|i| (i as f64).sin()).collect();

    // Show default settings
    println!("Default settings:");
    println!("  Backend: {}", get_backend_name());
    println!("  Workers: {}", get_workers());
    println!("  Cache enabled: {}", get_global_cache().is_enabled());
    println!();

    // Test 1: Temporarily disable cache
    println!("Test 1: Without cache");
    let _result1 = without_cache(|| {
        println!(
            "  Cache enabled inside context: {}",
            get_global_cache().is_enabled()
        );

        let start = Instant::now();
        let spectrum = fft(&signal, None).unwrap();
        let duration1 = start.elapsed();

        let start = Instant::now();
        let _ = fft(&signal, None).unwrap();
        let duration2 = start.elapsed();

        println!("  First FFT: {:?}", duration1);
        println!("  Second FFT: {:?} (no caching)", duration2);

        spectrum
    })
    .unwrap();

    println!(
        "  Cache enabled after context: {}",
        get_global_cache().is_enabled()
    );
    println!();

    // Test 2: With specific backend (same as default in this case)
    println!("Test 2: With specific backend");
    let _result2 = with_backend("rustfft", || {
        println!("  Backend inside context: {}", get_backend_name());
        fft(&signal, None).unwrap()
    })
    .unwrap();
    println!("  Backend after context: {}", get_backend_name());
    println!();

    // Test 3: With specific number of workers
    println!("Test 3: With specific workers");
    let _result3 = with_workers(2, || {
        // Note: Due to current implementation limitations, this doesn't actually
        // change the worker count, but demonstrates the API
        println!("  Would use 2 workers if fully implemented");
        fft(&signal, None)
    })
    .unwrap();
    println!();

    // Test 4: Combined settings using builder
    println!("Test 4: Combined settings");
    let _result4 = with_fft_settings(
        fft_context()
            .backend("rustfft")
            .workers(4)
            .cache_enabled(false),
        || {
            println!("  Backend: {}", get_backend_name());
            println!("  Cache enabled: {}", get_global_cache().is_enabled());

            let start = Instant::now();
            let spectrum = fft(&signal, None).unwrap();
            let duration = start.elapsed();

            println!("  FFT duration: {:?}", duration);

            spectrum
        },
    )
    .unwrap();

    println!("  Settings restored after context");
    println!("  Backend: {}", get_backend_name());
    println!("  Cache enabled: {}", get_global_cache().is_enabled());
    println!();

    // Test 5: Nested contexts
    println!("Test 5: Nested contexts");
    with_backend("rustfft", || {
        println!("  Outer context - Backend: {}", get_backend_name());

        without_cache(|| {
            println!("  Inner context - Backend: {}", get_backend_name());
            println!(
                "  Inner context - Cache: {}",
                get_global_cache().is_enabled()
            );
        })
        .unwrap();

        println!(
            "  Back to outer context - Cache: {}",
            get_global_cache().is_enabled()
        );
    })
    .unwrap();

    println!();

    // Test 6: Verify FFT functionality within contexts
    println!("Test 6: FFT functionality check");
    let original_spectrum = fft(&signal, None).unwrap();

    let context_spectrum = without_cache(|| fft(&signal, None).unwrap()).unwrap();

    // Check if results are the same
    let results_match = original_spectrum
        .iter()
        .zip(context_spectrum.iter())
        .all(|(a, b)| (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);

    println!("  Results match: {}", results_match);

    // Show cache statistics
    let stats = get_global_cache().get_stats();
    println!("\nCache statistics:");
    println!("{}", stats);
}
