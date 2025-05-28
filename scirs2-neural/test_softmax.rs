use ndarray::arr1;
use scirs2_neural::activations::softmax::softmax;

fn main() {
    println!("Testing softmax implementation...\n");
    
    // Test case 1: Simple 1D array
    let input = arr1(&[1.0, 2.0, 3.0]);
    println!("Input: {:?}", input);
    
    let output = softmax(&input, 0);
    println!("Output: {:?}", output);
    println!("Sum of output: {}", output.sum());
    println!("Expected sum: 1.0");
    
    // Print individual values
    println!("\nDetailed output:");
    for (i, &val) in output.iter().enumerate() {
        println!("  output[{}] = {}", i, val);
    }
    
    // Manually calculate expected values
    let exp_1 = 1.0_f64.exp();
    let exp_2 = 2.0_f64.exp();
    let exp_3 = 3.0_f64.exp();
    let sum_exp = exp_1 + exp_2 + exp_3;
    
    println!("\nManual calculation:");
    println!("  exp(1.0) = {}", exp_1);
    println!("  exp(2.0) = {}", exp_2);
    println!("  exp(3.0) = {}", exp_3);
    println!("  sum of exp = {}", sum_exp);
    println!("  softmax[0] should be = {} / {} = {}", exp_1, sum_exp, exp_1 / sum_exp);
    println!("  softmax[1] should be = {} / {} = {}", exp_2, sum_exp, exp_2 / sum_exp);
    println!("  softmax[2] should be = {} / {} = {}", exp_3, sum_exp, exp_3 / sum_exp);
    
    // Test case 2: Check numerical stability with larger values
    println!("\n\nTest case 2: Larger values");
    let input2 = arr1(&[100.0, 200.0, 300.0]);
    println!("Input: {:?}", input2);
    
    let output2 = softmax(&input2, 0);
    println!("Output: {:?}", output2);
    println!("Sum of output: {}", output2.sum());
    
    // Test case 3: Negative values
    println!("\n\nTest case 3: Negative values");
    let input3 = arr1(&[-1.0, 0.0, 1.0]);
    println!("Input: {:?}", input3);
    
    let output3 = softmax(&input3, 0);
    println!("Output: {:?}", output3);
    println!("Sum of output: {}", output3.sum());
}