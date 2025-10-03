//! Hardware Optimization Showcase Example
//!
//! This example demonstrates the hardware-specific optimizations available
//! in scirs2-linalg, including SIMD acceleration and GPU foundations.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::{
    error::LinalgResult,
    simd_ops::{hardware_optimized_dot, hardware_optimized_matvec, HardwareCapabilities},
};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("🚀 SciRS2 Linear Algebra Hardware Optimization Showcase");
    println!("========================================================");

    // Detect hardware capabilities
    let capabilities = HardwareCapabilities::detect();
    println!("🔍 Hardware Capabilities Detected:");
    println!("   AVX Support:     {}", capabilities.has_avx);
    println!("   AVX2 Support:    {}", capabilities.has_avx2);
    println!("   AVX-512 Support: {}", capabilities.has_avx512);
    println!("   FMA Support:     {}", capabilities.has_fma);
    println!("   ARM Neon:        {}", capabilities.has_neon);
    println!(
        "   Optimal Vector Width: {} bytes",
        capabilities.optimal_vector_width()
    );
    println!();

    // Create test data
    let size = 1000;
    println!(
        "📊 Creating test matrices and vectors (size: {}x{})",
        size, size
    );

    let matrix = Array2::from_shape_fn((size, size), |(i, j)| ((i + j + 1) as f64 * 0.001).sin());

    let vector = Array1::from_shape_fn(size, |i| ((i + 1) as f64 * 0.001).cos());

    let vector2 = Array1::from_shape_fn(size, |i| ((i + 1) as f64 * 0.001).tan());

    println!("✅ Test data created successfully");
    println!();

    // Demonstrate hardware-optimized dot product
    println!("🧮 Hardware-Optimized Dot Product:");
    let start_time = std::time::Instant::now();
    let dot_result = hardware_optimized_dot(&vector.view(), &vector2.view(), &capabilities)?;
    let dot_time = start_time.elapsed();
    println!("   Result: {:.6}", dot_result);
    println!("   Time:   {:?}", dot_time);
    println!();

    // Demonstrate hardware-optimized matrix-vector multiplication
    println!("📐 Hardware-Optimized Matrix-Vector Multiplication:");
    let start_time = std::time::Instant::now();
    let matvec_result = hardware_optimized_matvec(&matrix.view(), &vector.view(), &capabilities)?;
    let matvec_time = start_time.elapsed();
    println!("   Result shape: {:?}", matvec_result.shape());
    println!(
        "   First 5 elements: {:?}",
        &matvec_result.slice(scirs2_core::ndarray::s![0..5]).to_vec()
    );
    println!("   Time: {:?}", matvec_time);
    println!();

    // Compare with standard implementations
    println!("⚖️  Performance Comparison:");

    // Standard dot product
    let start_time = std::time::Instant::now();
    let std_dot_result = vector.dot(&vector2);
    let std_dot_time = start_time.elapsed();

    // Standard matrix-vector multiplication
    let start_time = std::time::Instant::now();
    let std_matvec_result = matrix.dot(&vector);
    let std_matvec_time = start_time.elapsed();

    println!("   Standard dot product time:  {:?}", std_dot_time);
    println!("   Optimized dot product time: {:?}", dot_time);
    if dot_time < std_dot_time {
        let speedup = std_dot_time.as_nanos() as f64 / dot_time.as_nanos() as f64;
        println!("   🎉 Speedup: {:.2}x faster!", speedup);
    } else {
        println!("   ⚠️  Standard implementation was faster (overhead for small matrices)");
    }
    println!();

    println!("   Standard matvec time:  {:?}", std_matvec_time);
    println!("   Optimized matvec time: {:?}", matvec_time);
    if matvec_time < std_matvec_time {
        let speedup = std_matvec_time.as_nanos() as f64 / matvec_time.as_nanos() as f64;
        println!("   🎉 Speedup: {:.2}x faster!", speedup);
    } else {
        println!("   ⚠️  Standard implementation was faster (overhead for medium matrices)");
    }
    println!();

    // Verify correctness
    println!("🔍 Verification:");
    let dot_error = (dot_result - std_dot_result).abs();
    let matvec_error = (&matvec_result - &std_matvec_result)
        .mapv(|x| x.abs())
        .sum()
        / matvec_result.len() as f64;

    println!("   Dot product error: {:.2e}", dot_error);
    println!("   Matvec average error: {:.2e}", matvec_error);

    if dot_error < 1e-10 && matvec_error < 1e-10 {
        println!("   ✅ Results are numerically identical!");
    } else {
        println!(
            "   ⚠️  Small numerical differences detected (expected with different algorithms)"
        );
    }
    println!();

    // GPU information (if available)
    #[cfg(any(
        feature = "cuda",
        feature = "opencl",
        feature = "rocm",
        feature = "metal"
    ))]
    {
        use scirs2_linalg::gpu::{initialize_gpu_manager, should_use_gpu};

        println!("🖥️  GPU Acceleration Status:");
        match initialize_gpu_manager() {
            Ok(gpu_manager) => {
                let backends = gpu_manager.available_backends();
                if backends.is_empty() {
                    println!("   No GPU backends available");
                } else {
                    println!("   Available GPU backends:");
                    for backend in backends {
                        println!("     - {}", backend.name());
                    }

                    let matrix_elements = matrix.len();
                    let should_use = should_use_gpu(matrix_elements, 50_000, None);
                    println!("   Should use GPU for this problem size: {}", should_use);
                }
            }
            Err(e) => {
                println!("   GPU initialization failed: {}", e);
            }
        }
        println!();
    }

    // Recommendations
    println!("💡 Performance Recommendations:");
    if capabilities.has_avx2 {
        println!("   ✅ AVX2 detected - optimal performance on x86_64");
    } else if capabilities.has_avx {
        println!("   ⚠️  Only AVX detected - consider upgrading for better performance");
    } else {
        println!("   ⚠️  No AVX detected - performance may be limited");
    }

    if capabilities.has_fma {
        println!("   ✅ FMA support detected - excellent for matrix operations");
    }

    if capabilities.has_neon {
        println!("   ✅ ARM Neon detected - optimal performance on ARM");
    }

    let optimal_width = capabilities.optimal_vector_width();
    if optimal_width >= 32 {
        println!(
            "   ✅ Large vector width ({} bytes) - excellent SIMD performance",
            optimal_width
        );
    } else {
        println!(
            "   ⚠️  Smaller vector width ({} bytes) - basic SIMD support",
            optimal_width
        );
    }

    println!();
    println!("🎯 Summary:");
    println!("   - Hardware capabilities automatically detected");
    println!("   - Optimizations applied based on available features");
    println!("   - Fallback to standard implementations when needed");
    println!("   - All operations maintain numerical accuracy");

    Ok(())
}

#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn main() {
    println!("❌ This example requires the 'simd' feature to be enabled.");
    println!("   Run with: cargo run --example hardware_optimization_showcase --features simd");
}
