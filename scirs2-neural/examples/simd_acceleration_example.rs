//! SIMD acceleration example for neural network operations
//!
//! This example demonstrates the performance benefits of SIMD-accelerated
//! neural network operations including activations, matrix operations,
//! normalization, and loss functions.

use ndarray::Array2;
use scirs2_neural::error::Result;
#[cfg(not(feature = "simd"))]
#[allow(unused_imports)]
use scirs2_neural::performance::PerformanceOptimizer;
#[cfg(feature = "simd")]
use scirs2_neural::performance::{PerformanceOptimizer, SIMDOperations, SIMDStats};
use std::time::Instant;

fn main() -> Result<()> {
    println!("SIMD Acceleration Demo for Neural Networks");
    println!("==========================================");

    // Check SIMD capabilities
    check_simd_capabilities();

    // Demo 1: Activation Functions
    demo_simd_activations()?;

    // Demo 2: Matrix Operations
    demo_simd_matrix_operations()?;

    // Demo 3: Normalization Operations
    demo_simd_normalization()?;

    // Demo 4: Loss Functions
    demo_simd_loss_functions()?;

    // Demo 5: Performance Comparison
    demo_performance_comparison()?;

    // Demo 6: Integrated Neural Network Forward Pass
    demo_neural_network_forward_pass()?;

    Ok(())
}

fn check_simd_capabilities() {
    println!("\n🔍 SIMD Capabilities Check");
    println!("-------------------------");

    #[cfg(feature = "simd")]
    {
        let simd_available = SIMDOperations::is_simd_available();
        println!("SIMD Available: {}", if simd_available { "✓" } else { "✗" });

        if simd_available {
            let stats = SIMDStats {
                simd_available: true,
                vector_width_f32: 8,
                vector_width_f64: 4,
                supported_operations: vec![
                    "relu".to_string(),
                    "sigmoid".to_string(),
                    "tanh".to_string(),
                    "gelu".to_string(),
                    "swish".to_string(),
                    "softmax".to_string(),
                    "matmul".to_string(),
                    "add".to_string(),
                    "mul".to_string(),
                    "batch_norm".to_string(),
                    "layer_norm".to_string(),
                    "cross_entropy".to_string(),
                    "mse".to_string(),
                ],
            };
            println!("{}", stats);
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD features not enabled. Compile with --features simd for acceleration.");
    }
}

#[cfg(feature = "simd")]
fn demo_simd_activations() -> Result<()> {
    println!("\n⚡ SIMD Activation Functions Demo");
    println!("--------------------------------");

    // Create test data
    let input = Array2::from_shape_fn((256, 512), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.001).sin() * 2.0 - 1.0
    })
    .into_dyn();

    println!("Input shape: {:?}", input.shape());

    // Test ReLU
    println!("\nTesting ReLU activation...");
    let start = Instant::now();
    let relu_result = SIMDOperations::simd_relu_f32(&input.view());
    let relu_time = start.elapsed();
    println!("SIMD ReLU time: {:?}", relu_time);
    println!(
        "ReLU output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        relu_result.iter().cloned().fold(f32::INFINITY, f32::min),
        relu_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        relu_result.mean().unwrap_or(0.0)
    );

    // Test Sigmoid
    println!("\nTesting Sigmoid activation...");
    let start = Instant::now();
    let sigmoid_result = SIMDOperations::simd_sigmoid_f32(&input.view());
    let sigmoid_time = start.elapsed();
    println!("SIMD Sigmoid time: {:?}", sigmoid_time);
    println!(
        "Sigmoid output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        sigmoid_result.iter().cloned().fold(f32::INFINITY, f32::min),
        sigmoid_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        sigmoid_result.mean().unwrap_or(0.0)
    );

    // Test Tanh
    println!("\nTesting Tanh activation...");
    let start = Instant::now();
    let tanh_result = SIMDOperations::simd_tanh_f32(&input.view());
    let tanh_time = start.elapsed();
    println!("SIMD Tanh time: {:?}", tanh_time);
    println!(
        "Tanh output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        tanh_result.iter().cloned().fold(f32::INFINITY, f32::min),
        tanh_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        tanh_result.mean().unwrap_or(0.0)
    );

    // Test GELU
    println!("\nTesting GELU activation...");
    let start = Instant::now();
    let gelu_result = SIMDOperations::simd_gelu_f32(&input.view());
    let gelu_time = start.elapsed();
    println!("SIMD GELU time: {:?}", gelu_time);
    println!(
        "GELU output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        gelu_result.iter().cloned().fold(f32::INFINITY, f32::min),
        gelu_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        gelu_result.mean().unwrap_or(0.0)
    );

    // Test Swish
    println!("\nTesting Swish activation...");
    let start = Instant::now();
    let swish_result = SIMDOperations::simd_swish_f32(&input.view());
    let swish_time = start.elapsed();
    println!("SIMD Swish time: {:?}", swish_time);
    println!(
        "Swish output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        swish_result.iter().cloned().fold(f32::INFINITY, f32::min),
        swish_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        swish_result.mean().unwrap_or(0.0)
    );

    Ok(())
}

#[cfg(not(feature = "simd"))]
fn demo_simd_activations() -> Result<()> {
    println!("\n⚡ SIMD Activation Functions Demo");
    println!("--------------------------------");
    println!("SIMD features not enabled. Skipping SIMD activation demo.");
    Ok(())
}

#[cfg(feature = "simd")]
fn demo_simd_matrix_operations() -> Result<()> {
    println!("\n🔢 SIMD Matrix Operations Demo");
    println!("-----------------------------");

    // Create test matrices
    let a = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 * 0.01).into_dyn();
    let b =
        Array2::from_shape_fn((256, 64), |(i, j)| (i as i32 - j as i32) as f32 * 0.01).into_dyn();

    println!("Matrix A shape: {:?}", a.shape());
    println!("Matrix B shape: {:?}", b.shape());

    // Test matrix multiplication
    println!("\nTesting SIMD matrix multiplication...");
    let start = Instant::now();
    let matmul_result = SIMDOperations::simd_matmul_f32(&a.view(), &b.view())?;
    let matmul_time = start.elapsed();
    println!("SIMD MatMul time: {:?}", matmul_time);
    println!("Result shape: {:?}", matmul_result.shape());
    println!(
        "MatMul result stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        matmul_result.iter().cloned().fold(f32::INFINITY, f32::min),
        matmul_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        matmul_result.mean().unwrap_or(0.0)
    );

    // Test element-wise addition
    let c = Array2::from_elem((128, 64), 1.0).into_dyn();
    println!("\nTesting SIMD element-wise addition...");
    let start = Instant::now();
    let add_result = SIMDOperations::simd_add_f32(&matmul_result.view(), &c.view())?;
    let add_time = start.elapsed();
    println!("SIMD Add time: {:?}", add_time);
    println!(
        "Add result stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        add_result.iter().cloned().fold(f32::INFINITY, f32::min),
        add_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        add_result.mean().unwrap_or(0.0)
    );

    // Test element-wise multiplication
    println!("\nTesting SIMD element-wise multiplication...");
    let start = Instant::now();
    let mul_result = SIMDOperations::simd_mul_f32(&matmul_result.view(), &c.view())?;
    let mul_time = start.elapsed();
    println!("SIMD Mul time: {:?}", mul_time);
    println!(
        "Mul result stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        mul_result.iter().cloned().fold(f32::INFINITY, f32::min),
        mul_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        mul_result.mean().unwrap_or(0.0)
    );

    Ok(())
}

#[cfg(not(feature = "simd"))]
fn demo_simd_matrix_operations() -> Result<()> {
    println!("\n🔢 SIMD Matrix Operations Demo");
    println!("-----------------------------");
    println!("SIMD features not enabled. Skipping SIMD matrix operations demo.");
    Ok(())
}

#[cfg(feature = "simd")]
fn demo_simd_normalization() -> Result<()> {
    println!("\n📊 SIMD Normalization Demo");
    println!("-------------------------");

    // Create test data
    let input = Array2::from_shape_fn((32, 128), |(i, j)| {
        (i as f32 + j as f32) * 0.1 + (i as f32 * j as f32).sin()
    })
    .into_dyn();

    // Test batch normalization
    let mean = Array2::from_shape_fn((1, 128), |(_, j)| j as f32 * 0.01).into_dyn();
    let variance = Array2::from_elem((1, 128), 1.0).into_dyn();
    let gamma = Array2::from_elem((1, 128), 1.0).into_dyn();
    let beta = Array2::from_elem((1, 128), 0.0).into_dyn();

    println!("Input shape: {:?}", input.shape());
    println!("\nTesting SIMD batch normalization...");
    let start = Instant::now();
    let batch_norm_result = SIMDOperations::simd_batch_norm_f32(
        &input.view(),
        &mean.view(),
        &variance.view(),
        Some(&gamma.view()),
        Some(&beta.view()),
        1e-5,
    )?;
    let batch_norm_time = start.elapsed();
    println!("SIMD Batch Norm time: {:?}", batch_norm_time);
    println!(
        "Batch Norm result stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        batch_norm_result
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        batch_norm_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        batch_norm_result.mean().unwrap_or(0.0)
    );

    // Test layer normalization
    println!("\nTesting SIMD layer normalization...");
    let start = Instant::now();
    let layer_norm_result = SIMDOperations::simd_layer_norm_f32(
        &input.view(),
        Some(&gamma.view()),
        Some(&beta.view()),
        1e-5,
    )?;
    let layer_norm_time = start.elapsed();
    println!("SIMD Layer Norm time: {:?}", layer_norm_time);
    println!(
        "Layer Norm result stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        layer_norm_result
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        layer_norm_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        layer_norm_result.mean().unwrap_or(0.0)
    );

    Ok(())
}

#[cfg(not(feature = "simd"))]
fn demo_simd_normalization() -> Result<()> {
    println!("\n📊 SIMD Normalization Demo");
    println!("-------------------------");
    println!("SIMD features not enabled. Skipping SIMD normalization demo.");
    Ok(())
}

#[cfg(feature = "simd")]
fn demo_simd_loss_functions() -> Result<()> {
    println!("\n📉 SIMD Loss Functions Demo");
    println!("---------------------------");

    // Create test data for classification
    let predictions = Array2::from_shape_fn((64, 10), |(i, j)| {
        let logit = (i + j) as f32 * 0.1;
        logit.exp() / (0..10).map(|k| ((i + k) as f32 * 0.1).exp()).sum::<f32>()
    })
    .into_dyn();

    let targets = Array2::from_shape_fn((64, 10), |(i, _)| {
        let class = i % 10;
        if i % 10 == class {
            1.0
        } else {
            0.0
        }
    })
    .into_dyn();

    println!("Predictions shape: {:?}", predictions.shape());
    println!("Targets shape: {:?}", targets.shape());

    // Test cross-entropy loss
    println!("\nTesting SIMD cross-entropy loss...");
    let start = Instant::now();
    let ce_loss =
        SIMDOperations::simd_cross_entropy_loss_f32(&predictions.view(), &targets.view(), 1e-7)?;
    let ce_time = start.elapsed();
    println!("SIMD Cross-Entropy time: {:?}", ce_time);
    println!("Cross-Entropy Loss: {:.6}", ce_loss);

    // Test MSE loss
    println!("\nTesting SIMD MSE loss...");
    let start = Instant::now();
    let mse_loss = SIMDOperations::simd_mse_loss_f32(&predictions.view(), &targets.view())?;
    let mse_time = start.elapsed();
    println!("SIMD MSE time: {:?}", mse_time);
    println!("MSE Loss: {:.6}", mse_loss);

    // Test softmax
    println!("\nTesting SIMD softmax...");
    let logits = Array2::from_shape_fn((32, 10), |(i, j)| (i + j) as f32 * 0.1).into_dyn();
    let start = Instant::now();
    let softmax_result = SIMDOperations::simd_softmax_f32(&logits, Some(1))?;
    let softmax_time = start.elapsed();
    println!("SIMD Softmax time: {:?}", softmax_time);
    println!(
        "Softmax result stats - Min: {:.6}, Max: {:.6}, Sum per row: {:.6}",
        softmax_result.iter().cloned().fold(f32::INFINITY, f32::min),
        softmax_result
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        softmax_result
            .sum_axis(ndarray::Axis(1))
            .mean()
            .unwrap_or(0.0)
    );

    Ok(())
}

#[cfg(not(feature = "simd"))]
fn demo_simd_loss_functions() -> Result<()> {
    println!("\n📉 SIMD Loss Functions Demo");
    println!("---------------------------");
    println!("SIMD features not enabled. Skipping SIMD loss functions demo.");
    Ok(())
}

fn demo_performance_comparison() -> Result<()> {
    println!("\n🏃 Performance Comparison");
    println!("------------------------");

    // Create large test data
    let size = 1000;
    let input = Array2::from_shape_fn((size, size), |(i, j)| {
        (i as f32 * 0.001 + j as f32 * 0.001).sin()
    })
    .into_dyn();

    println!(
        "Testing with {}x{} matrix ({} elements)",
        size,
        size,
        size * size
    );

    // Test scalar vs SIMD ReLU
    println!("\nReLU Performance Comparison:");

    // Scalar version
    let start = Instant::now();
    let _scalar_result = input.mapv(|x| x.max(0.0));
    let scalar_time = start.elapsed();
    println!("  Scalar ReLU time: {:?}", scalar_time);

    // SIMD version
    #[cfg(feature = "simd")]
    {
        let start = Instant::now();
        let simd_result = SIMDOperations::simd_relu_f32(&input.view());
        let simd_time = start.elapsed();
        println!("  SIMD ReLU time: {:?}", simd_time);

        if scalar_time.as_nanos() > 0 {
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("  Speedup: {:.2}x", speedup);
        }

        // Verify results are the same
        let diff = (&_scalar_result - &simd_result).mapv(|x| x.abs()).sum();
        println!("  Result difference: {:.2e}", diff);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  SIMD version not available (compile with --features simd)");
    }

    Ok(())
}

#[cfg(feature = "simd")]
fn demo_neural_network_forward_pass() -> Result<()> {
    println!("\n🧠 Integrated Neural Network Forward Pass");
    println!("------------------------------------------");

    let mut optimizer = PerformanceOptimizer::new(Some(64), Some(512), Some(4), true)?;

    // Simulate a mini neural network forward pass
    let batch_size = 32;
    let input_size = 784; // 28x28 MNIST-like
    let hidden_size = 256;
    let output_size = 10;

    println!(
        "Network: {} -> {} -> {}",
        input_size, hidden_size, output_size
    );
    println!("Batch size: {}", batch_size);

    // Create input data
    let input = Array2::from_shape_fn((batch_size, input_size), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.001).sin()
    })
    .into_dyn();

    // Create weight matrices
    let w1 = Array2::from_shape_fn((input_size, hidden_size), |(i, j)| {
        ((i + j) as f32 * 0.001).sin() * 0.1
    })
    .into_dyn();
    let w2 = Array2::from_shape_fn((hidden_size, output_size), |(i, j)| {
        ((i as i32 - j as i32) as f32 * 0.001).cos() * 0.1
    })
    .into_dyn();

    println!("\nExecuting SIMD-accelerated forward pass...");
    let total_start = Instant::now();

    // Layer 1: Linear + ReLU
    let z1 = optimizer.optimized_matmul(&input, &w1)?;
    let a1 = optimizer.simd_relu_f32(&z1);

    // Layer 2: Linear + Softmax
    let z2 = optimizer.optimized_matmul(&a1, &w2)?;
    let a2 = optimizer.simd_softmax_f32(&z2, Some(1))?;

    let total_time = total_start.elapsed();

    println!("Total forward pass time: {:?}", total_time);
    println!("Output shape: {:?}", a2.shape());
    println!(
        "Output stats - Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        a2.iter().cloned().fold(f32::INFINITY, f32::min),
        a2.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        a2.mean().unwrap_or(0.0)
    );

    // Verify softmax properties (sum should be ~1 for each sample)
    let row_sums = a2.sum_axis(ndarray::Axis(1));
    let avg_sum = row_sums.mean().unwrap_or(0.0);
    println!("Average softmax row sum: {:.6} (should be ~1.0)", avg_sum);

    // Show profiling results
    println!("\nPerformance Profile:");
    optimizer.profiler().print_summary();

    // Show SIMD capabilities
    let simd_stats = optimizer.get_simd_stats();
    println!("\n{}", simd_stats);

    Ok(())
}

#[cfg(not(feature = "simd"))]
fn demo_neural_network_forward_pass() -> Result<()> {
    println!("\n🧠 Integrated Neural Network Forward Pass");
    println!("------------------------------------------");
    println!("SIMD features not enabled. Skipping integrated demo.");
    println!("Compile with --features simd for full functionality.");
    Ok(())
}
