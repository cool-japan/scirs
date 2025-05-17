//! Example demonstrating the FFT backend system
//!
//! This example shows how to use and switch between different FFT backends.

// use num_complex::Complex64;  // Unused import
use scirs2_fft::{fft, get_backend_info, get_backend_name, ifft, list_backends, BackendContext};

fn main() {
    println!("FFT Backend System Example");
    println!("=========================");
    println!();

    // List available backends
    let backends = list_backends();
    println!("Available backends:");
    for backend_name in &backends {
        if let Some(info) = get_backend_info(backend_name) {
            println!("  {}", info);
        }
    }
    println!();

    // Show current backend
    println!("Current backend: {}", get_backend_name());
    println!();

    // Create test signal
    let signal: Vec<f64> = (0..16).map(|i| i as f64).collect();
    println!("Test signal: {:?}", signal);

    // Use default backend
    println!("\nUsing default backend ({}):", get_backend_name());
    let spectrum = fft(&signal, None).unwrap();
    println!("  FFT result (first 4 values): {:?}", &spectrum[0..4]);

    // Verify inverse FFT
    let recovered = ifft(&spectrum, None).unwrap();
    println!(
        "  IFFT recovery successful: {}",
        signal
            .iter()
            .zip(recovered.iter())
            .all(|(a, b)| (a - b.re).abs() < 1e-10 && b.im.abs() < 1e-10)
    );

    // Demonstrate backend context (for future use when we have multiple backends)
    println!("\nUsing backend context:");
    {
        let _ctx = BackendContext::new("rustfft").unwrap();
        println!("  Inside context: backend = {}", get_backend_name());
        let _ = fft(&signal, None).unwrap();
    }
    println!("  Outside context: backend = {}", get_backend_name());

    // Test backend info
    println!("\nBackend capabilities:");
    if let Some(backend_manager) = Some(scirs2_fft::get_backend_manager()) {
        let backend = backend_manager.get_backend();
        let features = vec![
            "1d_fft",
            "2d_fft",
            "nd_fft",
            "cached_plans",
            "gpu_acceleration",
        ];

        for feature in features {
            println!(
                "  Supports {}: {}",
                feature,
                backend.supports_feature(feature)
            );
        }
    }

    // Demonstrate array-like object handling (conceptual)
    println!("\nArray interoperability:");

    // Convert from various array types
    let vec_input = vec![1.0, 2.0, 3.0, 4.0];
    let ndarray_input = ndarray::Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

    // All should work with the same interface
    let result1 = fft(&vec_input, None).unwrap();
    let result2 = fft(&ndarray_input.as_slice().unwrap(), None).unwrap();

    println!("  Vec input FFT: success");
    println!("  ndarray input FFT: success");
    println!(
        "  Results match: {}",
        result1
            .iter()
            .zip(result2.iter())
            .all(|(a, b)| (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10)
    );

    // Future features demonstration (commented out as they're not implemented yet)
    /*
    // Switch to a hypothetical FFTW backend
    set_backend("fftw").ok();

    // Use GPU backend if available
    if backends.contains(&"cuda_fft".to_string()) {
        set_backend("cuda_fft").ok();
        println!("\nUsing GPU backend:");
        let gpu_result = fft(&signal, None).unwrap();
        println!("  GPU FFT completed");
    }
    */
}
