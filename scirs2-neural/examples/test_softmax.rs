use ndarray::{arr1, arr2};
use scirs2_neural::activations::{Activation, Softmax};

fn main() {
    println!("Testing softmax implementation...\n");

    // Test case 1: Simple 1D array
    let input = arr1(&[1.0, 2.0, 3.0]);
    println!("Input: {:?}", input);

    let softmax = Softmax::new(0);
    let output = softmax.forward(&input.clone().into_dyn()).unwrap();
    println!("Softmax output: {:?}", output);

    // Verify that output sums to 1
    let sum: f64 = output.sum();
    println!("Sum of softmax: {}", sum);
    assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1");

    // Test case 2: 2D array (batch processing)
    println!("\nTest case 2: 2D batch");
    let input_2d = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]]);
    println!("Input 2D:\n{:?}", input_2d);

    // Apply softmax along axis 1 (row-wise)
    let softmax_2d = Softmax::new(1);
    let output_2d = softmax_2d.forward(&input_2d.clone().into_dyn()).unwrap();
    println!("Softmax output 2D:\n{:?}", output_2d);

    // Verify each row sums to 1
    for i in 0..output_2d.shape()[0] {
        let row_sum: f64 = output_2d.slice(ndarray::s![i, ..]).sum();
        println!("Row {} sum: {}", i, row_sum);
        assert!((row_sum - 1.0).abs() < 1e-6, "Each row should sum to 1");
    }

    // Test case 3: Gradient computation
    println!("\nTest case 3: Gradient computation");
    let grad_output = arr1(&[0.1, 0.2, 0.3]).into_dyn();
    let forward_output = softmax.forward(&input.clone().into_dyn()).unwrap();
    let grad_input = softmax.backward(&grad_output, &forward_output).unwrap();
    println!("Gradient input: {:?}", grad_input);

    println!("\nAll tests passed!");
}
