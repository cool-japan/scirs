use scirs2_core::ndarray_ext::Array1;
use std::time::Instant;

#[cfg(feature = "simd")]
use scirs2_core::simd::{
    simd_mul_f32, simd_mul_f32_branchfree, simd_mul_f32_cacheline, simd_mul_f32_hyperoptimized,
    simd_mul_f32_pipelined, simd_mul_f32_tlb_optimized,
};

#[cfg(not(feature = "simd"))]
fn main() {
    println!("This example requires the 'simd' feature to be enabled.");
    println!("Run with: cargo run --example hyper_performance_benchmark --features simd");
}

#[cfg(feature = "simd")]
fn benchmark_operation<F>(name: &str, size: usize, iterations: usize, mut op: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        op();
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        op();
    }
    let elapsed = start.elapsed();

    let throughput = (size as f64 * iterations as f64 * 4.0) / elapsed.as_secs_f64() / 1e9;
    println!(
        "{:<30} | Time: {:>8.2} ms | Throughput: {:>6.2} GB/s",
        name,
        elapsed.as_millis(),
        throughput
    );

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

#[cfg(feature = "simd")]
fn run_benchmark_suite(size: usize, iterations: usize) {
    println!("\n{}", "=".repeat(75));
    println!(
        "Array size: {} elements ({:.2} MB)",
        size,
        size as f64 * 4.0 / 1_048_576.0
    );
    println!("Iterations: {}", iterations);
    println!("{}", "-".repeat(75));

    // Create test arrays
    let a = Array1::from_vec(vec![1.5f32; size]);
    let b = Array1::from_vec(vec![2.5f32; size]);

    // Scalar baseline
    let scalar_time = benchmark_operation("Scalar baseline", size, iterations, || {
        let mut result = vec![0.0f32; size];
        for i in 0..size {
            result[i] = a[i] * b[i];
        }
        std::hint::black_box(&result);
    });

    // Standard SIMD
    let standard_time = benchmark_operation("Standard SIMD", size, iterations, || {
        let result = simd_mul_f32(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // Branch-free SIMD (current champion)
    let branchfree_time = benchmark_operation("Branch-free SIMD", size, iterations, || {
        let result = simd_mul_f32_branchfree(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // Cache-line aware
    let cacheline_time = benchmark_operation("Cache-line aware", size, iterations, || {
        let result = simd_mul_f32_cacheline(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // Software pipelined
    let pipelined_time = benchmark_operation("Software pipelined", size, iterations, || {
        let result = simd_mul_f32_pipelined(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // TLB-optimized
    let tlb_time = benchmark_operation("TLB-optimized", size, iterations, || {
        let result = simd_mul_f32_tlb_optimized(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // Hyper-optimized
    let hyper_time = benchmark_operation("HYPER-OPTIMIZED", size, iterations, || {
        let result = simd_mul_f32_hyperoptimized(&a.view(), &b.view());
        std::hint::black_box(&result);
    });

    // Performance comparisons
    println!("{}", "-".repeat(75));
    println!("Performance vs Scalar:");
    println!("  Standard SIMD:      {:.2}x", scalar_time / standard_time);
    println!(
        "  Branch-free SIMD:   {:.2}x",
        scalar_time / branchfree_time
    );
    println!("  Cache-line aware:   {:.2}x", scalar_time / cacheline_time);
    println!("  Software pipelined: {:.2}x", scalar_time / pipelined_time);
    println!("  TLB-optimized:      {:.2}x", scalar_time / tlb_time);
    println!("  HYPER-OPTIMIZED:    {:.2}x", scalar_time / hyper_time);

    // Find winner
    let mut times = [
        ("Standard SIMD", standard_time),
        ("Branch-free SIMD", branchfree_time),
        ("Cache-line aware", cacheline_time),
        ("Software pipelined", pipelined_time),
        ("TLB-optimized", tlb_time),
        ("HYPER-OPTIMIZED", hyper_time),
    ];
    times.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("{}", "-".repeat(75));
    println!(
        "🏆 WINNER: {} ({:.2}x faster than scalar)",
        times[0].0,
        scalar_time / times[0].1
    );
}

#[cfg(feature = "simd")]
fn main() {
    println!("\n╔{}╗", "═".repeat(73));
    println!("║{:^73}║", "HYPER-PERFORMANCE SIMD BENCHMARK");
    println!("║{:^73}║", "Testing Ultra-Optimized Implementations");
    println!("╚{}╝", "═".repeat(73));

    // Test different array sizes
    let test_cases = vec![
        (256, 10000),  // L1 cache (1 KB)
        (4096, 5000),  // L2 cache (16 KB)
        (65536, 1000), // L3 cache (256 KB)
        (524288, 100), // Large (2 MB)
        (4194304, 10), // Huge (16 MB)
        (16777216, 5), // Massive (64 MB)
    ];

    for (size, iterations) in test_cases {
        run_benchmark_suite(size, iterations);
    }

    println!("\n╔{}╗", "═".repeat(73));
    println!("║{:^73}║", "BENCHMARK COMPLETE");
    println!("╚{}╝", "═".repeat(73));

    // Test correctness
    println!("\n📊 Correctness Verification:");
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

    let standard = simd_mul_f32(&a.view(), &b.view());
    let cacheline = simd_mul_f32_cacheline(&a.view(), &b.view());
    let pipelined = simd_mul_f32_pipelined(&a.view(), &b.view());
    let tlb = simd_mul_f32_tlb_optimized(&a.view(), &b.view());
    let hyper = simd_mul_f32_hyperoptimized(&a.view(), &b.view());

    let all_correct =
        standard == cacheline && standard == pipelined && standard == tlb && standard == hyper;

    if all_correct {
        println!("✅ All implementations produce identical results!");
    } else {
        println!("❌ Implementation mismatch detected!");
    }
}
