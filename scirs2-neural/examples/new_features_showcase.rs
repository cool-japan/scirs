//! Showcase of new neural network features
//!
//! This example demonstrates the new features added to the neural network module:
//! - Adaptive pooling layers (AdaptiveAvgPool2D, AdaptiveMaxPool2D)
//! - Activity regularization (L1, L2, and combined)
//! - Performance optimizations (thread pools, profiling)

use ndarray::{Array, Array4};
use scirs2_neural::{
    layers::{
        ActivityRegularization, AdaptiveAvgPool2D, AdaptiveMaxPool2D, L1ActivityRegularization,
        L2ActivityRegularization, Layer,
    },
    performance::{PerformanceOptimizer, ThreadPoolManager},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Neural Network New Features Showcase ===\n");

    // 1. Demonstrate Adaptive Pooling Layers
    demonstrate_adaptive_pooling()?;

    // 2. Demonstrate Activity Regularization
    demonstrate_activity_regularization()?;

    // 3. Demonstrate Performance Optimizations
    demonstrate_performance_optimizations()?;

    println!("=== All demonstrations completed successfully! ===");
    Ok(())
}

fn demonstrate_adaptive_pooling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Adaptive Pooling Layers Demonstration");
    println!("========================================\n");

    // Create input tensor: batch_size=2, channels=3, height=32, width=32
    let input = Array4::<f64>::from_elem((2, 3, 32, 32), 1.5);
    println!("Input shape: {:?}", input.shape());

    // Adaptive Average Pooling to 7x7
    println!("\n1. Adaptive Average Pooling (32x32 → 7x7):");
    let adaptive_avg_pool = AdaptiveAvgPool2D::new((7, 7), Some("adaptive_avg_7x7"))?;
    let avg_output = adaptive_avg_pool.forward(&input.clone().into_dyn())?;
    println!("   Output shape: {:?}", avg_output.shape());
    println!(
        "   Layer description: {}",
        adaptive_avg_pool.layer_description()
    );

    // Adaptive Max Pooling to 4x4
    println!("\n2. Adaptive Max Pooling (32x32 → 4x4):");
    let adaptive_max_pool = AdaptiveMaxPool2D::new((4, 4), Some("adaptive_max_4x4"))?;
    let max_output = adaptive_max_pool.forward(&input.into_dyn())?;
    println!("   Output shape: {:?}", max_output.shape());
    println!(
        "   Layer description: {}",
        adaptive_max_pool.layer_description()
    );

    // Non-square adaptive pooling
    println!("\n3. Non-square Adaptive Pooling (32x32 → 3x5):");
    let non_square_pool = AdaptiveAvgPool2D::new((3, 5), Some("non_square"))?;
    let non_square_output =
        non_square_pool.forward(&Array4::<f64>::from_elem((1, 2, 16, 20), 2.0).into_dyn())?;
    println!("   Input shape: [1, 2, 16, 20]");
    println!("   Output shape: {:?}", non_square_output.shape());

    println!("✅ Adaptive pooling demonstration completed!\n");
    Ok(())
}

fn demonstrate_activity_regularization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Activity Regularization Demonstration");
    println!("=======================================\n");

    // Create some activations to regularize
    let activations =
        Array::from_shape_vec((2, 4), vec![1.5, -2.0, 0.5, 3.0, -1.0, 0.0, 2.5, -0.5])?.into_dyn();

    println!("Input activations:");
    println!("{:?}\n", activations);

    // 1. L1 Activity Regularization
    println!("1. L1 Activity Regularization (factor=0.1):");
    let l1_reg = L1ActivityRegularization::new(0.1, Some("l1_regularizer"))?;
    let l1_output = l1_reg.forward(&activations)?;
    let l1_loss = l1_reg.get_activity_loss()?;
    println!("   Output (unchanged): {:?}", l1_output.shape());
    println!("   L1 activity loss: {:.4}", l1_loss);
    println!("   Layer description: {}", l1_reg.layer_description());

    // 2. L2 Activity Regularization
    println!("\n2. L2 Activity Regularization (factor=0.05):");
    let l2_reg = L2ActivityRegularization::new(0.05, Some("l2_regularizer"))?;
    let l2_output = l2_reg.forward(&activations)?;
    let l2_loss = l2_reg.get_activity_loss()?;
    println!("   Output (unchanged): {:?}", l2_output.shape());
    println!("   L2 activity loss: {:.4}", l2_loss);
    println!("   Layer description: {}", l2_reg.layer_description());

    // 3. Combined L1 + L2 Activity Regularization
    println!("\n3. Combined L1+L2 Activity Regularization:");
    let combined_reg = ActivityRegularization::new(Some(0.1), Some(0.05), Some("combined_reg"))?;
    let combined_output = combined_reg.forward(&activations)?;
    let combined_loss = combined_reg.get_activity_loss()?;
    println!("   Output (unchanged): {:?}", combined_output.shape());
    println!("   Combined activity loss: {:.4}", combined_loss);
    println!("   Layer description: {}", combined_reg.layer_description());

    // Demonstrate backward pass
    println!("\n4. Backward Pass with Gradient Modification:");
    let grad_output = Array::ones(activations.raw_dim());
    let grad_input = combined_reg.backward(&activations, &grad_output)?;
    println!("   Gradient input shape: {:?}", grad_input.shape());
    println!(
        "   Sample gradient values: [{:.3}, {:.3}, {:.3}, {:.3}]",
        grad_input[[0, 0]],
        grad_input[[0, 1]],
        grad_input[[0, 2]],
        grad_input[[0, 3]]
    );

    println!("✅ Activity regularization demonstration completed!\n");
    Ok(())
}

fn demonstrate_performance_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Performance Optimizations Demonstration");
    println!("========================================\n");

    // 1. Thread Pool Manager
    println!("1. Thread Pool Manager:");
    let thread_pool = ThreadPoolManager::new(Some(4))?;
    println!(
        "   Created thread pool with {} threads",
        thread_pool.num_threads()
    );

    // Demonstrate parallel matrix multiplication
    let matrix_a = Array::from_elem((100, 50), 2.0f32).into_dyn();
    let matrix_b = Array::from_elem((50, 75), 3.0f32).into_dyn();

    let start_time = std::time::Instant::now();
    let result = thread_pool.parallel_matmul(&matrix_a, &matrix_b)?;
    let elapsed = start_time.elapsed();

    println!(
        "   Parallel matrix multiplication: {}x{} * {}x{} = {}x{}",
        matrix_a.shape()[0],
        matrix_a.shape()[1],
        matrix_b.shape()[0],
        matrix_b.shape()[1],
        result.shape()[0],
        result.shape()[1]
    );
    println!("   Time elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Result sample: {:.1}", result[[0, 0]]);

    // 2. Performance Profiler
    println!("\n2. Performance Profiler:");
    let mut optimizer = PerformanceOptimizer::new(Some(1024), Some(512), Some(4), true)?;

    // Simulate some operations with profiling
    {
        let timer = optimizer.profiler_mut().start_timer("matrix_setup");
        let test_matrix_a = Array::from_elem((200, 100), 1.5f32).into_dyn();
        let test_matrix_b = Array::from_elem((100, 150), 2.5f32).into_dyn();
        optimizer
            .profiler_mut()
            .end_timer("matrix_setup".to_string(), timer);

        let _result = optimizer.optimized_matmul(&test_matrix_a, &test_matrix_b)?;
    }

    println!("   Performance profile summary:");
    optimizer.profiler().print_summary();

    // 3. Optimization Capabilities
    println!("\n3. Optimization Capabilities:");
    let capabilities = optimizer.get_capabilities();
    println!("{}", capabilities);

    println!("✅ Performance optimizations demonstration completed!\n");
    Ok(())
}
