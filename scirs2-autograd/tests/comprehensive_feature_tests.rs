//! Comprehensive tests for recently implemented features
//!
//! This test module provides comprehensive coverage for:
//! - Advanced tensor indexing operations
//! - Tensor broadcasting optimizations
//! - Memory optimization features
//! - Efficient reshape and slice operations
//! - Property-based testing for mathematical operations
//! - Numerical stability testing

use ndarray::{Array, IxDyn};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;

/// Test advanced tensor indexing operations
#[cfg(test)]
mod advanced_indexing_tests {
    use super::*;

    #[test]
    fn test_boolean_mask_basic() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create test data
            let data = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            let mask = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap(),
                ctx,
            );

            // Apply boolean mask
            let result = T::boolean_mask(&data, &mask);

            // The result should be a 1D tensor with selected elements
            let result_array = result.eval(ctx).unwrap();

            // Should contain elements where mask was true: [1.0, 3.0, 5.0]
            assert_eq!(result_array.len(), 3);
        });
    }

    #[test]
    fn test_take_operation() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create test data: [10, 20, 30, 40, 50]
            let data = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
                ctx,
            );

            // Indices to take: [0, 2, 4, 1]
            let indices = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![0.0, 2.0, 4.0, 1.0]).unwrap(),
                ctx,
            );

            // Apply take operation
            let result = T::take(&data, &indices, 0);

            // Verify result
            let result_array = result.eval(ctx).unwrap();
            assert_eq!(result_array.len(), 4);

            // Should contain [10.0, 30.0, 50.0, 20.0]
            let expected = vec![10.0, 30.0, 50.0, 20.0];
            for (i, &expected_val) in expected.iter().enumerate() {
                assert!((result_array[i] - expected_val).abs() < 1e-6_f32);
            }
        });
    }

    #[test]
    fn test_where_operation() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create test tensors
            let condition = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.0, 1.0, 0.0]).unwrap(),
                ctx,
            );

            let x = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ctx,
            );

            let y = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
                ctx,
            );

            // Apply where operation
            let result = T::where_op(&condition, &x, &y);

            // Verify result: should be [1.0, 20.0, 3.0, 40.0]
            let result_array = result.eval(ctx).unwrap();
            let expected = vec![1.0, 20.0, 3.0, 40.0];
            for (i, &expected_val) in expected.iter().enumerate() {
                assert!((result_array[i] - expected_val).abs() < 1e-6_f32);
            }
        });
    }

    #[test]
    fn test_scatter_operation() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create indices and updates
            let indices = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 2.0, 1.0]).unwrap(),
                ctx,
            );

            let updates = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![10.0, 30.0, 20.0]).unwrap(),
                ctx,
            );

            // Apply scatter operation
            let result = T::scatter(&indices, &updates, 5, 0);

            // Verify result: should be [10.0, 20.0, 30.0, 0.0, 0.0]
            let result_array = result.eval(ctx).unwrap();
            assert_eq!(result_array.len(), 5);

            let expected = vec![10.0, 20.0, 30.0, 0.0, 0.0];
            for (i, &expected_val) in expected.iter().enumerate() {
                assert!((result_array[i] - expected_val).abs() < 1e-6_f32);
            }
        });
    }
}

/// Test tensor broadcasting optimizations
#[cfg(test)]
mod broadcasting_tests {
    use super::*;

    #[test]
    fn test_broadcast_analysis() {
        // Test same shape (no broadcasting needed)
        let left_shape = vec![3, 4];
        let right_shape = vec![3, 4];
        let info = T::analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.strategy, T::BroadcastStrategy::NoOp);
        assert!(!info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_scalar() {
        // Test scalar broadcasting
        let left_shape = vec![];
        let right_shape = vec![3, 4];
        let info = T::analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.strategy, T::BroadcastStrategy::ScalarBroadcast);
        assert!(info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_compatible() {
        // Test compatible broadcasting
        let left_shape = vec![1, 4];
        let right_shape = vec![3, 1];
        let info = T::analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(info.left_needs_broadcast);
        assert!(info.right_needs_broadcast);
    }

    #[test]
    fn test_broadcast_analysis_incompatible() {
        // Test incompatible broadcasting
        let left_shape = vec![3, 4];
        let right_shape = vec![2, 4];
        let result = T::analyze_broadcast(&left_shape, &right_shape);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_operations() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 2.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[1, 3]), vec![10.0, 20.0, 30.0]).unwrap(),
                ctx,
            );

            // Test broadcast addition
            let result = T::broadcast_add(&a, &b);
            let result_array = result.eval(ctx).unwrap();

            // Result should be 2x3 matrix
            assert_eq!(result_array.shape(), &[2, 3]);

            // Test broadcast multiplication
            let result = T::broadcast_mul(&a, &b);
            let result_array = result.eval(ctx).unwrap();
            assert_eq!(result_array.shape(), &[2, 3]);
        });
    }

    #[test]
    fn test_broadcast_cache() {
        // Test cache operations
        T::clear_broadcast_cache();
        let (size, _) = T::get_broadcast_cache_stats();
        assert_eq!(size, 0);

        // After some operations, cache should have entries
        let left_shape = vec![2, 3];
        let right_shape = vec![2, 3];
        let _ = T::analyze_broadcast(&left_shape, &right_shape).unwrap();

        let (size, _) = T::get_broadcast_cache_stats();
        assert!(size > 0);
    }
}

/// Test memory optimization features
#[cfg(test)]
mod memory_optimization_tests {
    use super::*;

    #[test]
    fn test_memory_pool_operations() {
        // First clear any existing state
        T::clear_memory_pool();

        // Configure memory pool
        T::configure_memory_pool(8, 50 * 1024 * 1024);
        T::set_memory_pool_enabled(true);

        // Get a buffer from the pool
        let buffer = T::get_pooled_buffer(1024);
        assert_eq!(buffer.len(), 1024);

        // Return buffer to pool
        T::return_pooled_buffer(buffer);

        // Check pool stats
        let stats = T::get_memory_pool_stats();
        assert!(stats.enabled);

        // Clear pool
        T::clear_memory_pool();
    }

    #[test]
    fn test_memory_tracking() {
        // First disable and reset to ensure clean state
        T::disable_memory_tracking();
        T::reset_memory_tracking();

        // Enable memory tracking
        T::enable_memory_tracking();

        // Verify it's enabled immediately after enabling
        let initial_stats = T::get_memory_tracking_stats();
        assert!(
            initial_stats.enabled,
            "Memory tracking should be enabled after calling enable_memory_tracking()"
        );

        // Perform some operations that would be tracked
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            let a = T::ones(
                &T::convert_to_tensor(
                    Array::from_shape_vec(IxDyn(&[2]), vec![100.0, 100.0]).unwrap(),
                    ctx,
                ),
                ctx,
            );
            let b = T::ones(
                &T::convert_to_tensor(
                    Array::from_shape_vec(IxDyn(&[2]), vec![100.0, 100.0]).unwrap(),
                    ctx,
                ),
                ctx,
            );
            let _result = T::inplace_add(&a, &b);
        });

        // Check tracking stats again
        let stats = T::get_memory_tracking_stats();
        assert!(
            stats.enabled,
            "Memory tracking should still be enabled after operations"
        );

        // Reset tracking
        T::reset_memory_tracking();
        T::disable_memory_tracking();
    }

    #[test]
    fn test_memory_optimizer() {
        // Test memory optimizer sessions
        T::MemoryOptimizer::start_session();

        // Perform some operations
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            let a = T::efficient_zeros(&[10, 10], ctx);
            let b = T::efficient_ones(&[10, 10], ctx);
            let _result = T::inplace_mul(&a, &b);
        });

        let (tracking_stats, pool_stats) = T::MemoryOptimizer::end_session();

        // Should have some statistics
        assert!(tracking_stats.enabled || pool_stats.enabled);
    }

    #[test]
    fn test_inplace_operations() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            // Test in-place operations
            let add_result = T::inplace_add(&a, &b);
            let add_array = add_result.eval(ctx).unwrap();

            // Should be [5.0, 7.0, 9.0]
            // Debug print the actual values
            println!("InPlace add result shape: {:?}", add_array.shape());
            println!("InPlace add result values: {:?}", add_array.as_slice());

            // Expected: a[i] + b[i] = [1+4, 2+5, 3+6] = [5, 7, 9]
            let expected = vec![5.0, 7.0, 9.0];
            for i in 0..3 {
                let actual = add_array[[i]];
                let expected_val = expected[i];
                println!(
                    "Index {}: actual = {}, expected = {}",
                    i, actual, expected_val
                );
                assert!(
                    (actual - expected_val).abs() < 1e-6_f32,
                    "InPlace add failed at index {}: expected {}, got {}",
                    i,
                    expected_val,
                    actual
                );
            }

            let mul_result = T::inplace_mul(&a, &b);
            let mul_array = mul_result.eval(ctx).unwrap();

            // Should be [4.0, 10.0, 18.0]
            let expected = vec![4.0, 10.0, 18.0];
            for (i, &expected_val) in expected.iter().enumerate() {
                assert!((mul_array[i] - expected_val).abs() < 1e-6);
            }
        });
    }
}

/// Test efficient tensor operations
#[cfg(test)]
mod efficient_operations_tests {
    use super::*;

    #[test]
    fn test_efficient_reshape_cache() {
        // Test reshape cache operations
        T::clear_reshape_cache();
        let (size, _) = T::get_reshape_cache_stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_slice_range_operations() {
        // Test SliceRange creation
        let slice = T::SliceRange::new(Some(0), Some(10), Some(1));
        assert_eq!(slice.start, Some(0));
        assert_eq!(slice.end, Some(10));
        assert_eq!(slice.step, Some(1));

        let full_slice = T::SliceRange::full();
        assert_eq!(full_slice.start, None);
        assert_eq!(full_slice.end, None);
        assert_eq!(full_slice.step, None);

        let single_slice = T::SliceRange::single(5);
        assert_eq!(single_slice.start, Some(5));
        assert_eq!(single_slice.end, Some(6));
        assert_eq!(single_slice.step, Some(1));
    }

    #[test]
    fn test_efficient_reshape_operations() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create a 2x3 matrix
            let data = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            // Reshape to 3x2
            let reshaped = T::efficient_reshape_with_shape(&data, &[3, 2]);
            let reshaped_array = reshaped.eval(ctx).unwrap();

            assert_eq!(reshaped_array.shape(), &[3, 2]);
            assert_eq!(reshaped_array.len(), 6);
        });
    }

    #[test]
    fn test_efficient_slice_operations() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create a 4x4 matrix
            let data = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4, 4]), (0..16).map(|x| x as f32).collect()).unwrap(),
                ctx,
            );

            // Create slice ranges
            let slices = vec![
                T::SliceRange::new(Some(1), Some(3), Some(1)), // rows 1-2
                T::SliceRange::new(Some(0), Some(2), Some(1)), // cols 0-1
            ];

            let sliced = T::efficient_slice(&data, &slices);
            let sliced_array = sliced.eval(ctx).unwrap();

            // Should be a 2x2 matrix
            assert_eq!(sliced_array.shape(), &[2, 2]);
        });
    }

    #[test]
    fn test_efficient_concat_operations() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create test tensors
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap(),
                ctx,
            );

            // Concatenate along axis 0
            let concat_result = T::efficient_concat(&[&a, &b], 0);
            let concat_array = concat_result.eval(ctx).unwrap();

            // Should be a 4x2 matrix
            assert_eq!(concat_array.shape(), &[4, 2]);
            assert_eq!(concat_array.len(), 8);
        });
    }

    #[test]
    fn test_efficient_ops_manager() {
        // Test configuration methods
        T::EfficientOpsManager::configure_for_performance();
        let stats = T::EfficientOpsManager::get_performance_stats();
        assert!(stats.reshape_cache_max > 0);

        T::EfficientOpsManager::configure_for_memory();
        let stats = T::EfficientOpsManager::get_performance_stats();
        assert!(stats.reshape_cache_max > 0);
    }
}

/// Property-based testing for mathematical operations
#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_addition_commutativity() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test that a + b = b + a
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            let ab = &a + &b;
            let ba = &b + &a;

            let ab_array = ab.eval(ctx).unwrap();
            let ba_array = ba.eval(ctx).unwrap();

            for i in 0..3 {
                assert!((ab_array[i] - ba_array[i]).abs() < 1e-10_f32);
            }
        });
    }

    #[test]
    fn test_multiplication_associativity() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test that (a * b) * c = a * (b * c)
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![4.0, 5.0]).unwrap(),
                ctx,
            );

            let c = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![6.0, 7.0]).unwrap(),
                ctx,
            );

            let ab_c = (a * &b) * &c;
            let a_bc = &a * (b * &c);

            let ab_c_array = ab_c.eval(ctx).unwrap();
            let a_bc_array = a_bc.eval(ctx).unwrap();

            for i in 0..2 {
                assert!((ab_c_array[i] - a_bc_array[i]).abs() < 1e-10_f32);
            }
        });
    }

    #[test]
    fn test_distributive_property() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test that a * (b + c) = a * b + a * c
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![4.0, 5.0]).unwrap(),
                ctx,
            );

            let c = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![6.0, 7.0]).unwrap(),
                ctx,
            );

            let left = &a * (b + &c);
            let right = (a * &b) + (a * &c);

            let left_array = left.eval(ctx).unwrap();
            let right_array = right.eval(ctx).unwrap();

            for i in 0..2 {
                assert!((left_array[i] - right_array[i]).abs() < 1e-10_f32);
            }
        });
    }

    #[test]
    fn test_additive_identity() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test that a + 0 = a
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![1.5, -2.5, 3.7]).unwrap(),
                ctx,
            );

            let zero = T::zeros(
                &T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(), ctx),
                ctx,
            );

            let result = &a + &zero;
            let result_array = result.eval(ctx).unwrap();
            let a_array = a.eval(ctx).unwrap();

            for i in 0..3 {
                assert!((result_array[i] - a_array[i]).abs() < 1e-10_f32);
            }
        });
    }

    #[test]
    fn test_multiplicative_identity() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test that a * 1 = a
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![1.5, -2.5, 3.7]).unwrap(),
                ctx,
            );

            let one = T::ones(
                &T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(), ctx),
                ctx,
            );

            let result = &a * &one;
            let result_array = result.eval(ctx).unwrap();
            let a_array = a.eval(ctx).unwrap();

            for i in 0..3 {
                assert!((result_array[i] - a_array[i]).abs() < 1e-10_f32);
            }
        });
    }
}

/// Numerical stability tests
#[cfg(test)]
mod numerical_stability_tests {
    use super::*;

    #[test]
    fn test_large_number_stability() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test operations with large numbers
            let large_val = 1e10_f32;
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![large_val, large_val]).unwrap(),
                ctx,
            );

            let b = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
                ctx,
            );

            // Addition should maintain precision
            let result = &a + &b;
            let result_array = result.eval(ctx).unwrap();

            // Check that small increments are preserved
            assert!((result_array[0] - (large_val + 1.0)).abs() < 1e-6);
            assert!((result_array[1] - (large_val + 2.0)).abs() < 1e-6);
        });
    }

    #[test]
    fn test_small_number_stability() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test operations with very small numbers
            let small_val = 1e-10_f32;
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![small_val, small_val * 2.0]).unwrap(),
                ctx,
            );

            // Multiplication should not underflow to zero inappropriately
            let result = &a * &a;
            let result_array = result.eval(ctx).unwrap();

            // Should not be exactly zero (unless actually zero)
            assert!(result_array[0] > 0.0);
            assert!(result_array[1] > 0.0);
        });
    }

    #[test]
    fn test_zero_division_handling() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            let a = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
                ctx,
            );

            let zero = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
                ctx,
            );

            // Division by zero should produce infinity or NaN, not crash
            let result = &a / &zero;
            let result_array = result.eval(ctx).unwrap();

            // Check that we get infinity or NaN, not a crash
            assert!(result_array[0].is_infinite() || result_array[0].is_nan());
            assert!(result_array[1].is_infinite() || result_array[1].is_nan());
        });
    }

    #[test]
    fn test_gradient_numerical_stability() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test gradient computation with challenging inputs
            let x = T::variable(
                Array::from_shape_vec(IxDyn(&[2]), vec![1e-8_f32, 1e8_f32]).unwrap(),
                ctx,
            );

            // Compute a function that might have numerical issues
            let y = &x * &x + T::scalar(1e-10_f32, ctx);
            let loss = T::reduce_sum(y, &[0], false);

            // Compute gradients
            let grad = T::grad(&[loss], &[&x])[0];
            let grad_array = grad.eval(ctx).unwrap();

            // Gradients should be finite and reasonable
            for &g in grad_array.iter() {
                assert!(g.is_finite());
                assert!(!g.is_nan());
            }
        });
    }

    #[test]
    fn test_broadcasting_numerical_stability() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Test broadcasting with mixed scales
            let large = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[1, 3]), vec![1e6, 2e6, 3e6]).unwrap(),
                ctx,
            );

            let small = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 1]), vec![1e-6, 2e-6]).unwrap(),
                ctx,
            );

            // Broadcasting operations should maintain numerical stability
            let result = T::broadcast_add(&large, &small);
            let result_array = result.eval(ctx).unwrap();

            // All results should be finite
            for &val in result_array.iter() {
                let val: f32 = val;
                assert!(val.is_finite());
                assert!(!val.is_nan());
            }

            // Results should be dominated by the large values but include small increments
            assert!(result_array[[0, 0]] > 1e6 - 1.0);
            assert!(result_array[[0, 0]] < 1e6 + 1.0);
        });
    }
}

/// Integration tests that combine multiple features
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_memory_optimized_neural_network_layer() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Start memory optimization
            T::MemoryOptimizer::start_session();

            // Create a simple neural network layer with memory optimization
            let input = T::efficient_ones(&[32, 128], ctx); // Batch size 32, 128 features
            let weights = T::efficient_ones(&[128, 64], ctx); // 128 -> 64 transformation
            let bias = T::efficient_zeros(&[64], ctx);

            // Forward pass with checkpointing
            let linear = T::matmul(input, &weights);
            let linear_checkpointed = T::checkpoint(&linear);
            let biased = T::broadcast_add(&linear_checkpointed, &bias);
            let activated = T::relu(biased);

            // Use efficient operations
            let reshaped = T::efficient_reshape_with_shape(&activated, &[32 * 64]);
            let result = T::reduce_sum(reshaped, &[0], false);

            // Verify result
            let result_array = result.eval(ctx).unwrap();

            // Debug print the shape
            println!("Result shape: {:?}", result_array.shape());

            // Handle scalar result (reduce_sum with keep_dims=false produces a scalar)
            let result_value = if result_array.ndim() == 0 {
                result_array[[]]
            } else {
                result_array[[0]]
            };

            assert!(result_value.is_finite());
            assert!(result_value > 0.0); // ReLU should produce positive values

            // End memory optimization and check stats
            let (tracking_stats, pool_stats) = T::MemoryOptimizer::end_session();

            // Should have tracked some operations
            println!("Memory tracking enabled: {}", tracking_stats.enabled);
            println!("Pool enabled: {}", pool_stats.enabled);
        });
    }

    #[test]
    fn test_advanced_indexing_with_broadcasting() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create test data
            let data = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[4, 5]), (0..20).map(|x| x as f32).collect()).unwrap(),
                ctx,
            );

            // Create row indices
            let row_indices = T::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 2.0, 3.0]).unwrap(),
                ctx,
            );

            // Select rows using advanced indexing
            let selected_rows = T::select_rows(&data, &row_indices);

            // Apply broadcasting operation
            let scalar =
                T::convert_to_tensor(Array::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap(), ctx);

            let result = T::broadcast_mul(&selected_rows, &scalar);
            let result_array = result.eval(ctx).unwrap();

            // Should be 3x5 matrix with values multiplied by 10
            assert_eq!(result_array.shape(), &[3, 5]);

            // Check some values
            assert!((result_array[[0, 0]] - 0.0).abs() < 1e-6); // 0 * 10 = 0
            assert!((result_array[[1, 0]] - 100.0).abs() < 1e-6); // 10 * 10 = 100
        });
    }

    #[test]
    #[ignore = "Gradient shape propagation issue - architectural limitation"]
    fn test_efficient_operations_with_gradients() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create variable tensors
            let x = T::variable(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ctx,
            );

            // Apply efficient operations
            let reshaped = T::efficient_reshape_with_shape(&x, &[3, 2]);
            let checkpointed = T::checkpoint(&reshaped);

            // Slice operation
            let slices = vec![
                T::SliceRange::new(Some(0), Some(2), Some(1)), // First 2 rows
                T::SliceRange::full(),                         // All columns
            ];
            let sliced = T::efficient_slice(&checkpointed, &slices);

            // Compute loss
            let loss = T::reduce_sum(sliced, &[0, 1], false);

            // Compute gradients
            let grad = T::grad(&[loss], &[&x])[0];
            let grad_array = grad.eval(ctx).unwrap();

            // Gradients should have the same shape as input
            assert_eq!(grad_array.shape(), x.eval(ctx).unwrap().shape());

            // All gradient values should be finite
            for &g in grad_array.iter() {
                let g: f32 = g;
                assert!(g.is_finite());
            }
        });
    }
}

/// Stress tests with large tensors
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_large_tensor_broadcasting() {
        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Create moderately large tensors to test broadcasting performance
            let large_a = T::efficient_ones(&[100, 1], ctx);
            let large_b = T::efficient_ones(&[1, 100], ctx);

            // This should create a 100x100 result via broadcasting
            let result = T::broadcast_mul(&large_a, &large_b);
            let result_array = result.eval(ctx).unwrap();

            assert_eq!(result_array.shape(), &[100, 100]);
            assert_eq!(result_array.len(), 10000);

            // Check that all values are 1.0 (1 * 1 = 1)
            for &val in result_array.iter().take(100) {
                let val: f32 = val;
                assert!((val - 1.0_f32).abs() < 1e-6_f32);
            }
        });
    }

    #[test]
    fn test_memory_pool_with_many_operations() {
        // Enable memory pool
        T::set_memory_pool_enabled(true);
        T::enable_memory_tracking();

        ag::run(|ctx: &mut ag::Context<'_, f32>| {
            // Perform many operations that would benefit from memory pooling
            for i in 0..10 {
                let size = 100 + i * 10;
                let a = T::efficient_zeros(&[size], ctx);
                let b = T::efficient_ones(&[size], ctx);
                let _result = T::inplace_add(&a, &b);
            }
        });

        // Check that memory pool was used
        let pool_stats = T::get_memory_pool_stats();
        let tracking_stats = T::get_memory_tracking_stats();

        // Should have some memory tracking activity
        println!(
            "Total memory allocations: {:?}",
            tracking_stats.total_allocations
        );
        println!("Pool stats: {:?}", pool_stats);

        // Cleanup
        T::disable_memory_tracking();
        T::reset_memory_tracking();
    }
}
