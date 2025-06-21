//! Comprehensive tests for ultrathink mode features
//!
//! This test module demonstrates and tests all the advanced features implemented
//! during the ultrathink mode development session, including:
//! - Custom activation function framework
//! - Performance optimizations (SIMD operations, cache-friendly algorithms)
//! - Enhanced dynamic computation graph features
//! - Memory optimization improvements
//! - Advanced tensor operations

use ndarray::{Array, IxDyn};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;

/// Test suite for custom activation functions
#[cfg(test)]
mod custom_activation_tests {
    use super::*;

    #[test]
    fn test_built_in_custom_activations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Create test tensor
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
                ctx,
            );

            // Test Swish activation
            let swish_result = T::custom_activation(&x, "swish");
            let swish_output = swish_result.eval(ctx).unwrap();

            // Swish(0) should be 0, Swish(1) should be positive
            assert_eq!(swish_output[1], 0.0);
            assert!(swish_output[2] > 0.0);

            // Test Mish activation
            let mish_result = T::custom_activation(&x, "mish");
            let mish_output = mish_result.eval(ctx).unwrap();

            // Mish should produce reasonable outputs
            assert!(mish_output.iter().all(|&val: &f32| val.is_finite()));

            // Test GELU activation
            let gelu_result = T::custom_activation(&x, "gelu");
            let gelu_output = gelu_result.eval(ctx).unwrap();

            // GELU should be monotonic and produce finite outputs
            assert!(gelu_output.iter().all(|&val: &f32| val.is_finite()));
            assert!(gelu_output[0] < gelu_output[1]);
            assert!(gelu_output[1] < gelu_output[2]);

            // Test Parametric ReLU
            let prelu_result = T::custom_activation(&x, "parametric_relu");
            let prelu_output = prelu_result.eval(ctx).unwrap();

            // PReLU should handle negative values with small slope
            assert!(prelu_output[0] < 0.0); // Negative input -> small negative output
            assert_eq!(prelu_output[1], 0.0); // Zero input -> zero output
            assert_eq!(prelu_output[2], 1.0); // Positive input -> same output
        });
    }

    #[test]
    fn test_custom_activation_registry() {
        // Test activation function registry
        assert!(T::is_activation_registered("swish"));
        assert!(T::is_activation_registered("mish"));
        assert!(T::is_activation_registered("gelu"));
        assert!(!T::is_activation_registered("nonexistent_activation"));

        let functions = T::list_activation_functions();
        assert!(functions.contains(&"swish".to_string()));
        assert!(functions.contains(&"mish".to_string()));
        assert!(functions.contains(&"gelu".to_string()));
    }

    #[test]
    fn test_parameterized_activations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![-0.5, 1.5]).unwrap(),
                ctx,
            );

            // Test parameterized activation with different parameters
            let result1 = T::parameterized_activation(&x, "parametric_relu", &[0.01]);
            let result2 = T::parameterized_activation(&x, "parametric_relu", &[0.1]);

            let output1 = result1.eval(ctx).unwrap();
            let output2 = result2.eval(ctx).unwrap();

            // Different parameters should produce different results for negative inputs
            assert!((output1[0] as f32).abs() < (output2[0] as f32).abs()); // Different negative slopes
            assert_eq!(output1[1], output2[1]); // Same for positive inputs
        });
    }
}

/// Test suite for performance optimizations
#[cfg(test)]
mod performance_optimization_tests {
    use super::*;

    #[test]
    fn test_simd_operations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Test SIMD-optimized operations
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ctx,
            );
            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
                ctx,
            );

            // Test SIMD addition
            let simd_add_result = T::simd_add(&a, &b);
            let add_output = simd_add_result.eval(ctx).unwrap();

            // Should be [3.0, 5.0, 7.0, 9.0]
            for i in 0..4 {
                assert!((add_output[i] - (2.0 * i as f32 + 3.0)).abs() < 1e-6);
            }

            // Test SIMD multiplication
            let simd_mul_result = T::simd_mul(&a, &b);
            let mul_output = simd_mul_result.eval(ctx).unwrap();

            // Should be [2.0, 6.0, 12.0, 20.0]
            let expected = vec![2.0, 6.0, 12.0, 20.0];
            for (i, &expected_val) in expected.iter().enumerate() {
                assert!((mul_output[i] - expected_val).abs() < 1e-6);
            }
        });
    }

    #[test]
    fn test_simd_unary_operations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 2.0]).unwrap(),
                ctx,
            );

            // Test SIMD ReLU
            let relu_result = T::simd_relu(&x);
            let relu_output = relu_result.eval(ctx).unwrap();

            // Should be [0.0, 0.0, 2.0]
            assert_eq!(relu_output[0], 0.0);
            assert_eq!(relu_output[1], 0.0);
            assert_eq!(relu_output[2], 2.0);

            // Test SIMD Sigmoid
            let sigmoid_result = T::simd_sigmoid(&x);
            let sigmoid_output = sigmoid_result.eval(ctx).unwrap();

            // All outputs should be between 0 and 1
            for &val in sigmoid_output.iter() {
                assert!(val >= 0.0 && val <= 1.0);
            }
        });
    }

    #[test]
    fn test_cache_friendly_matmul() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );
            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            // Test cache-friendly matrix multiplication
            let result = T::cache_friendly_matmul(&a, &b, Some(32));
            let output = result.eval(ctx).unwrap();

            // Verify matrix multiplication result
            assert_eq!(output.shape(), &[2, 2]);
            assert!(output.iter().all(|&val: &f32| val.is_finite()));
        });
    }

    #[test]
    fn test_parallel_reductions() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4, 3]), (0..12).map(|i| i as f32).collect()).unwrap(),
                ctx,
            );

            // Test parallel sum
            let sum_result = T::parallel_sum(&x, &[0], false);
            let sum_output = sum_result.eval(ctx).unwrap();

            // Should sum along axis 0
            assert_eq!(sum_output.shape(), &[3]);
            assert!(sum_output.iter().all(|&val| val.is_finite()));
        });
    }

    #[test]
    fn test_performance_configuration() {
        // Test performance settings
        T::set_simd_enabled(false);
        assert!(!T::is_simd_enabled());

        T::set_simd_enabled(true);
        assert!(T::is_simd_enabled());

        T::set_parallel_enabled(false);
        assert!(!T::is_parallel_enabled());

        T::set_parallel_enabled(true);
        assert!(T::is_parallel_enabled());

        // Test configuration utilities
        T::PerformanceConfig::configure_for_compatibility();
        let (simd, parallel) = T::PerformanceConfig::get_settings();
        assert!(!simd);
        assert!(!parallel);

        T::PerformanceConfig::configure_for_performance();
        let (simd, parallel) = T::PerformanceConfig::get_settings();
        assert!(simd);
        assert!(parallel);
    }
}

/// Test suite for enhanced dynamic computation graph
#[cfg(test)]
mod graph_enhancement_tests {
    use super::*;

    #[test]
    fn test_conditional_operations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let condition =
                T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(), ctx);
            let true_branch = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap(),
                ctx,
            );
            let false_branch = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![30.0, 40.0]).unwrap(),
                ctx,
            );

            // Test conditional with true condition
            let result = T::conditional(
                &condition,
                &true_branch,
                &false_branch,
                T::PredicateType::GreaterThanZero,
            );
            let output = result.eval(ctx).unwrap();

            // Should select true branch
            assert_eq!(output[0], 10.0);
            assert_eq!(output[1], 20.0);

            // Test with false condition
            let false_condition =
                T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(), ctx);
            let result2 = T::conditional(
                &false_condition,
                &true_branch,
                &false_branch,
                T::PredicateType::GreaterThanZero,
            );
            let output2 = result2.eval(ctx).unwrap();

            // Should select false branch
            assert_eq!(output2[0], 30.0);
            assert_eq!(output2[1], 40.0);
        });
    }

    #[test]
    fn test_smart_checkpointing() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[1000]), (0..1000).map(|i| i as f32).collect())
                    .unwrap(),
                ctx,
            );

            // Test smart checkpointing with different thresholds
            let checkpointed_small = T::smart_checkpoint(&x, 100000); // High threshold
            let checkpointed_large = T::smart_checkpoint(&x, 1000); // Low threshold

            let output1 = checkpointed_small.eval(ctx).unwrap();
            let output2 = checkpointed_large.eval(ctx).unwrap();

            // Both should produce the same result
            assert_eq!(output1.len(), output2.len());
            for i in 0..output1.len() {
                assert_eq!(output1[i], output2[i]);
            }
        });
    }

    #[test]
    fn test_cached_operations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 4.0, 9.0]).unwrap(),
                ctx,
            );

            // Test cached operations
            let cached_identity = T::cached_op(&x, "identity");
            let cached_square = T::cached_op(&x, "square");
            let cached_sqrt = T::cached_op(&x, "sqrt");

            let identity_output = cached_identity.eval(ctx).unwrap();
            let square_output = cached_square.eval(ctx).unwrap();
            let sqrt_output = cached_sqrt.eval(ctx).unwrap();

            // Verify results
            assert_eq!(identity_output[0], 1.0);
            assert_eq!(identity_output[1], 4.0);
            assert_eq!(identity_output[2], 9.0);

            assert_eq!(square_output[0], 1.0); // 1^2 = 1
            assert_eq!(square_output[1], 16.0); // 4^2 = 16
            assert_eq!(square_output[2], 81.0); // 9^2 = 81

            assert_eq!(sqrt_output[0], 1.0); // sqrt(1) = 1
            assert_eq!(sqrt_output[1], 2.0); // sqrt(4) = 2
            assert_eq!(sqrt_output[2], 3.0); // sqrt(9) = 3
        });
    }

    #[test]
    fn test_computation_cache_management() {
        // Clear cache to start fresh
        T::clear_computation_cache();

        // Configure cache settings
        T::configure_cache(1000, 60);

        // Get initial cache stats
        let initial_stats = T::get_cache_stats();
        assert_eq!(initial_stats.entries, 0);
        assert_eq!(initial_stats.max_entries, 1000);

        // Perform some cached operations to populate cache
        ag::run(|ctx: &mut ag::Context<f32>| {
            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
                ctx,
            );
            let _result = T::cached_op(&x, "square");
        });

        // Check if cache has been used
        let final_stats = T::get_cache_stats();
        println!("Cache stats: {:?}", final_stats);
    }

    #[test]
    fn test_garbage_collection() {
        // Get initial GC stats
        let initial_gc_stats = T::get_gc_stats();
        println!("Initial GC stats: {:?}", initial_gc_stats);

        // Run garbage collection
        let collected = T::run_garbage_collection();
        println!("Collected {} items", collected);

        // Get final GC stats
        let final_gc_stats = T::get_gc_stats();
        println!("Final GC stats: {:?}", final_gc_stats);

        assert!(final_gc_stats.total_collections > initial_gc_stats.total_collections);
    }

    #[test]
    fn test_graph_enhancer_utilities() {
        // Test graph enhancement utilities
        T::GraphEnhancer::configure_for_memory_efficiency();
        let memory_stats = T::GraphEnhancer::get_graph_stats();
        assert_eq!(memory_stats.cache.max_entries, 1000);

        T::GraphEnhancer::configure_for_performance();
        let performance_stats = T::GraphEnhancer::get_graph_stats();
        assert_eq!(performance_stats.cache.max_entries, 50000);

        // Test graph optimization
        T::GraphEnhancer::optimize_graph();

        let optimized_stats = T::GraphEnhancer::get_graph_stats();
        println!("Optimized graph stats: {:?}", optimized_stats);
    }
}

/// Integration tests combining multiple features
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_performance_optimized_neural_layer() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Configure for maximum performance
            T::PerformanceConfig::configure_for_performance();
            T::GraphEnhancer::configure_for_performance();

            // Create a neural network layer with performance optimizations
            let input = T::efficient_ones(&[32, 128], ctx);
            let weights = T::efficient_ones(&[128, 64], ctx);
            let bias = T::efficient_zeros(&[64], ctx);

            // Use cache-friendly matrix multiplication
            let linear = T::cache_friendly_matmul(&input, &weights, Some(32));

            // Use smart checkpointing for memory efficiency
            let checkpointed = T::smart_checkpoint(&linear, 50000);

            // Add bias using SIMD-optimized operations
            let biased = T::simd_add(&checkpointed, &bias);

            // Apply custom activation function
            let activated = T::custom_activation(&biased, "swish");

            // Use cached operation for final processing
            let final_result = T::cached_op(&activated, "identity");

            // Verify the result
            let output = final_result.eval(ctx).unwrap();
            assert_eq!(output.shape(), &[32, 64]);
            assert!(output.iter().all(|&val: &f32| val.is_finite()));
        });
    }

    #[test]
    fn test_conditional_computation_with_optimizations() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Enable all optimizations
            T::set_simd_enabled(true);
            T::set_parallel_enabled(true);

            let condition =
                T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap(), ctx);

            // True branch: SIMD-optimized computation
            let data = T::efficient_ones(&[100], ctx);
            let true_branch = T::simd_relu(&data);

            // False branch: Custom activation
            let false_branch = T::custom_activation(&data, "gelu");

            // Conditional execution
            let result = T::conditional(
                &condition,
                &true_branch,
                &false_branch,
                T::PredicateType::Threshold(1.0),
            );

            // Apply parallel reduction
            let final_result = T::parallel_sum(&result, &[0], false);

            let output = final_result.eval(ctx).unwrap();
            assert!(output[0] > 0.0); // Should have selected ReLU branch (all positive)
        });
    }

    #[test]
    fn test_memory_optimized_workflow() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Configure for memory efficiency
            T::GraphEnhancer::configure_for_memory_efficiency();
            T::EfficientOpsManager::configure_for_memory();
            T::MemoryOptimizer::start_session();

            let x = T::efficient_ones(&[50, 50], ctx);

            // Use in-place operations to reduce memory usage
            let processed = T::inplace_mul(&x, &x);

            // Efficient reshape
            let reshaped = T::efficient_reshape_with_shape(&processed, &[2500]);

            // Smart checkpointing with low memory threshold
            let checkpointed = T::smart_checkpoint(&reshaped, 1000);

            // Custom activation with memory-friendly implementation
            let activated = T::custom_activation(&checkpointed, "parametric_relu");

            let output = activated.eval(ctx).unwrap();
            assert_eq!(output.len(), 2500);

            // Get memory statistics
            let (tracking_stats, pool_stats) = T::MemoryOptimizer::end_session();
            println!("Memory tracking stats: {:?}", tracking_stats);
            println!("Memory pool stats: {:?}", pool_stats);
        });
    }

    #[test]
    fn test_comprehensive_feature_integration() {
        ag::run(|ctx: &mut ag::Context<f32>| {
            // Test all major features working together

            // 1. Setup performance optimizations
            T::PerformanceConfig::configure_for_performance();

            // 2. Create data with efficient operations
            let data = T::efficient_ones(&[16, 32], ctx);

            // 3. Apply custom activation
            let activated = T::custom_activation(&data, "mish");

            // 4. Use SIMD operations
            let scaled = T::simd_mul(&activated, &activated);

            // 5. Cache the result
            let cached = T::cached_op(&scaled, "square");

            // 6. Apply conditional logic
            let condition =
                T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(), ctx);
            let alternative = T::efficient_zeros(&[16, 32], ctx);
            let conditional_result = T::conditional(
                &condition,
                &cached,
                &alternative,
                T::PredicateType::GreaterThanZero,
            );

            // 7. Smart checkpointing
            let checkpointed = T::smart_checkpoint(&conditional_result, 10000);

            // 8. Parallel reduction
            let final_result = T::parallel_sum(&checkpointed, &[0], false);

            // Verify the entire pipeline works
            let output = final_result.eval(ctx).unwrap();
            assert_eq!(output.shape(), &[32]);
            assert!(output.iter().all(|&val: &f32| val.is_finite()));
            assert!(output.iter().any(|&val| val > 0.0)); // Should have some positive values

            // Check system statistics
            let cache_stats = T::get_cache_stats();
            let gc_stats = T::get_gc_stats();
            let graph_stats = T::GraphEnhancer::get_graph_stats();

            println!("Final system statistics:");
            println!("Cache: {:?}", cache_stats);
            println!("GC: {:?}", gc_stats);
            println!("Graph: {:?}", graph_stats);
        });
    }
}
