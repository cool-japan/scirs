use scirs2_core::ndarray::{Array, Array1, Array2};
use scirs2_neural::activations_minimal::{Activation, ReLU, Sigmoid, Tanh, GELU};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Activation Functions Demonstration");
    // Create a set of input values
    let x_values: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();
    let x = Array1::from(x_values.clone());
    let x_dyn = x.clone().into_dyn();
    // Initialize all activation functions
    let relu = ReLU::new();
    let leaky_relu = ReLU::leaky(0.1);
    let sigmoid = Sigmoid::new();
    let tanh = Tanh::new();
    let gelu = GELU::new();
    let gelu_fast = GELU::fast();
    // Compute outputs for each activation function
    let relu_output = relu.forward(&x_dyn)?;
    let leaky_relu_output = leaky_relu.forward(&x_dyn)?;
    let sigmoid_output = sigmoid.forward(&x_dyn)?;
    let tanh_output = tanh.forward(&x_dyn)?;
    let gelu_output = gelu.forward(&x_dyn)?;
    let gelu_fast_output = gelu_fast.forward(&x_dyn)?;
    // Note: Swish and Mish are not available in the minimal activation set
    // Print sample values for each activation
    println!("Sample activation values for input x = -2.0, -1.0, 0.0, 1.0, 2.0:");
    let indices = [5, 40, 50, 60, 95]; // Corresponding to x = -2, -1, 0, 1, 2
    println!(
        "| {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |",
        "x", "-2.0", "-1.0", "0.0", "1.0", "2.0"
    );
    println!(
        "|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|",
        "", "", "", "", "", ""
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "ReLU",
        relu_output[[indices[0]]],
        relu_output[[indices[1]]],
        relu_output[[indices[2]]],
        relu_output[[indices[3]]],
        relu_output[[indices[4]]]
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "LeakyReLU",
        leaky_relu_output[[indices[0]]],
        leaky_relu_output[[indices[1]]],
        leaky_relu_output[[indices[2]]],
        leaky_relu_output[[indices[3]]],
        leaky_relu_output[[indices[4]]]
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "Sigmoid",
        sigmoid_output[[indices[0]]],
        sigmoid_output[[indices[1]]],
        sigmoid_output[[indices[2]]],
        sigmoid_output[[indices[3]]],
        sigmoid_output[[indices[4]]]
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "Tanh",
        tanh_output[[indices[0]]],
        tanh_output[[indices[1]]],
        tanh_output[[indices[2]]],
        tanh_output[[indices[3]]],
        tanh_output[[indices[4]]]
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "GELU",
        gelu_output[[indices[0]]],
        gelu_output[[indices[1]]],
        gelu_output[[indices[2]]],
        gelu_output[[indices[3]]],
        gelu_output[[indices[4]]]
    );
    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "GELU Fast",
        gelu_fast_output[[indices[0]]],
        gelu_fast_output[[indices[1]]],
        gelu_fast_output[[indices[2]]],
        gelu_fast_output[[indices[3]]],
        gelu_fast_output[[indices[4]]]
    );
    // Swish and Mish not available in minimal activation set
    // Now test the backward pass with some dummy gradient output
    println!("\nTesting backward pass...");
    // Create a dummy gradient output
    let dummy_grad = Array1::<f64>::ones(x.len()).into_dyn();
    // Compute gradients for each activation function
    let _relu_grad = relu.backward(&dummy_grad, &relu_output)?;
    let _leaky_relu_grad = leaky_relu.backward(&dummy_grad, &leaky_relu_output)?;
    let _sigmoid_grad = sigmoid.backward(&dummy_grad, &sigmoid_output)?;
    let _tanh_grad = tanh.backward(&dummy_grad, &tanh_output)?;
    let _gelu_grad = gelu.backward(&dummy_grad, &gelu_output)?;
    let _gelu_fast_grad = gelu_fast.backward(&dummy_grad, &gelu_fast_output)?;
    // Note: Swish and Mish gradients not available in minimal set
    println!("Backward pass completed successfully.");
    // Test with matrix input instead of vector
    println!("\nTesting with matrix input...");
    // Create a 3x4 matrix
    let mut matrix = Array2::<f64>::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            matrix[[i, j]] = -2.0 + (i as f64 * 4.0 + j as f64) * 0.5;
        }
    }
    // Print input matrix
    println!("Input matrix:");
    for i in 0..3 {
        print!("[ ");
        for j in 0..4 {
            print!("{:6.2} ", matrix[[i, j]]);
        }
        println!("]");
    }
    // Apply GELU activation to the matrix
    let gelu_matrix_output = gelu.forward(&matrix.into_dyn())?;
    // Print output matrix
    println!("\nAfter GELU activation:");
    for i in 0..3 {
        print!("[ ");
        for j in 0..4 {
            print!("{:6.2} ", gelu_matrix_output[[i, j]]);
        }
        println!("]");
    }
    println!("\nActivation functions demonstration completed successfully!");
    // Note about visualization
    println!("\nFor visualization of activation functions:");
    println!("1. You can use external plotting libraries like plotly or matplotlib");
    println!("2. To visualize these functions, you would plot the x_values against");
    println!("   the output values for each activation function");
    println!("3. The data from this example can be exported for plotting as needed");
    // Example of how to access the data for plotting
    println!("\nExample data points for plotting ReLU:");
    for i in 0..5 {
        let idx = i * 20; // Sample every 20th point
        if idx < x_values.len() {
            println!(
                "x: {:.2}, y: {:.6}",
                x_values[idx],
                convert_to_vec(&relu_output)[idx]
            );
        }
    }
    Ok(())
}
#[allow(dead_code)]
fn convert_to_vec<F: Clone>(array: &Array<F, scirs2_core::ndarray::IxDyn>) -> Vec<F> {
    array.iter().cloned().collect()
}
