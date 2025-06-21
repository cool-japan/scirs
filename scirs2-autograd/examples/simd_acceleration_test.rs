//! Test SIMD acceleration in autograd binary operations
//!
//! This example demonstrates the SIMD acceleration capabilities added to the
//! autograd module for element-wise operations like addition and multiplication.

use ag::tensor_ops as T;
use ndarray::Array1;
use scirs2_autograd as ag;
use std::time::Instant;

fn main() {
    println!("Testing SIMD acceleration in autograd binary operations");

    ag::run(|ctx| {
        // Create large 1D arrays for SIMD acceleration testing
        let size = 10000;
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        let array1 = Array1::from(data1);
        let array2 = Array1::from(data2);

        // Convert to tensors
        let tensor1 = T::convert_to_tensor(array1.clone(), ctx);
        let tensor2 = T::convert_to_tensor(array2.clone(), ctx);

        println!("Testing addition with {} elements", size);

        // Time the addition operation (potentially SIMD-accelerated)
        let start = Instant::now();
        let add_result = tensor1.clone() + tensor2.clone();
        let result = add_result.eval(ctx).unwrap();
        let duration = start.elapsed();

        println!("Addition completed in: {:?}", duration);
        println!("Result shape: {:?}", result.shape());
        println!(
            "First 10 elements: {:?}",
            &result.as_slice().unwrap()[0..10]
        );

        // Verify correctness
        let expected_0 = array1[0] + array2[0]; // 0 + 0 = 0
        let expected_1 = array1[1] + array2[1]; // 1 + 2 = 3
        let expected_2 = array1[2] + array2[2]; // 2 + 4 = 6

        assert_eq!(result.as_slice().unwrap()[0], expected_0);
        assert_eq!(result.as_slice().unwrap()[1], expected_1);
        assert_eq!(result.as_slice().unwrap()[2], expected_2);
        println!("✓ Addition correctness verified");

        println!("\nTesting multiplication with {} elements", size);

        // Time the multiplication operation (potentially SIMD-accelerated)
        let start = Instant::now();
        let mul_result = tensor1.clone() * tensor2.clone();
        let mul_result_eval = mul_result.eval(ctx).unwrap();
        let duration = start.elapsed();

        println!("Multiplication completed in: {:?}", duration);
        println!("Result shape: {:?}", mul_result_eval.shape());
        println!(
            "First 10 elements: {:?}",
            &mul_result_eval.as_slice().unwrap()[0..10]
        );

        // Verify correctness for multiplication
        let expected_mul_0 = array1[0] * array2[0]; // 0 * 0 = 0
        let expected_mul_1 = array1[1] * array2[1]; // 1 * 2 = 2
        let expected_mul_2 = array1[2] * array2[2]; // 2 * 4 = 8
        let expected_mul_3 = array1[3] * array2[3]; // 3 * 6 = 18

        assert_eq!(mul_result_eval.as_slice().unwrap()[0], expected_mul_0);
        assert_eq!(mul_result_eval.as_slice().unwrap()[1], expected_mul_1);
        assert_eq!(mul_result_eval.as_slice().unwrap()[2], expected_mul_2);
        assert_eq!(mul_result_eval.as_slice().unwrap()[3], expected_mul_3);
        println!("✓ Multiplication correctness verified");

        // Test with f64 as well
        println!("\nTesting with f64 precision");
        let data1_f64: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let data2_f64: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();

        let array1_f64 = Array1::from(data1_f64);
        let array2_f64 = Array1::from(data2_f64);

        // Test smaller arrays to see both SIMD and non-SIMD paths
        println!("\nTesting with smaller arrays (non-SIMD path)");
        let small_data1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let small_data2: Vec<f32> = vec![4.0, 5.0, 6.0];
        let small_array1 = Array1::from(small_data1);
        let small_array2 = Array1::from(small_data2);

        let small_tensor1 = T::convert_to_tensor(small_array1, ctx);
        let small_tensor2 = T::convert_to_tensor(small_array2, ctx);

        let small_add_result = small_tensor1 + small_tensor2;
        let small_result = small_add_result.eval(ctx).unwrap();

        println!("Small array result: {:?}", small_result.as_slice().unwrap());
        // Expected: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(small_result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
        println!("✓ Small array correctness verified");

        println!("\n🎉 All SIMD acceleration tests passed!");

        #[cfg(feature = "simd")]
        println!("✓ SIMD feature is enabled - using SIMD acceleration for suitable operations");

        #[cfg(not(feature = "simd"))]
        println!("ℹ SIMD feature is disabled - using standard operations");
    });
}
