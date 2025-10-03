//! Minimal working demo of scirs2-neural functionality

use scirs2_core::ndarray::Array;
use scirs2_neural::prelude::*;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🚀 Scirs2-Neural Minimal Working Demo");

    // Test GELU activation
    println!("\n📊 Testing GELU activation...");
    let gelu = GELU::new();
    let input = Array::from_vec(vec![1.0_f64, -1.0, 2.0, -2.0, 0.0]).into_dyn();
    println!("Input: {input:?}");

    let output = gelu.forward(&input)?;
    println!("GELU output: {output:?}");

    // Test backward pass
    let grad_output = Array::from_vec(vec![1.0_f64; 5]).into_dyn();
    let grad_input = gelu.backward(&grad_output, &input)?;
    println!("GELU gradient: {grad_input:?}");

    // Test Tanh activation
    println!("\n📊 Testing Tanh activation...");
    let tanh = Tanh::new();
    let tanh_output = tanh.forward(&input)?;
    println!("Tanh output: {tanh_output:?}");

    let tanh_grad = tanh.backward(&grad_output, &input)?;
    println!("Tanh gradient: {tanh_grad:?}");

    // Test GELU fast approximation
    println!("\n⚡ Testing GELU fast approximation...");
    let gelu_fast = GELU::fast();
    let fast_output = gelu_fast.forward(&input)?;
    println!("GELU fast output: {fast_output:?}");

    println!("\n✅ All tests completed successfully!");
    println!("🎉 Scirs2-Neural minimal functionality is working!");

    Ok(())
}
