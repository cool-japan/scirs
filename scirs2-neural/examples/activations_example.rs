use ndarray::{Array, Array1, Array2, Axis};
use plotly::common::Line;
use plotly::layout::{GridPattern, Layout, LayoutGrid};
use plotly::{Plot, Scatter};
use scirs2_neural::activations::{Activation, Mish, ReLU, Sigmoid, Softmax, Swish, Tanh, GELU};

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
    let swish = Swish::new(1.0);
    let mish = Mish::new();

    // Compute outputs for each activation function
    let relu_output = relu.forward(&x_dyn)?;
    let leaky_relu_output = leaky_relu.forward(&x_dyn)?;
    let sigmoid_output = sigmoid.forward(&x_dyn)?;
    let tanh_output = tanh.forward(&x_dyn)?;
    let gelu_output = gelu.forward(&x_dyn)?;
    let gelu_fast_output = gelu_fast.forward(&x_dyn)?;
    let swish_output = swish.forward(&x_dyn)?;
    let mish_output = mish.forward(&x_dyn)?;

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

    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "Swish",
        swish_output[[indices[0]]],
        swish_output[[indices[1]]],
        swish_output[[indices[2]]],
        swish_output[[indices[3]]],
        swish_output[[indices[4]]]
    );

    println!(
        "| {:<10} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} | {:<10.6} |",
        "Mish",
        mish_output[[indices[0]]],
        mish_output[[indices[1]]],
        mish_output[[indices[2]]],
        mish_output[[indices[3]]],
        mish_output[[indices[4]]]
    );

    // Now test the backward pass with some dummy gradient output
    println!("\nTesting backward pass...");

    // Create a dummy gradient output
    let dummy_grad = Array1::<f64>::ones(x.len()).into_dyn();

    // Compute gradients for each activation function
    let relu_grad = relu.backward(&dummy_grad, &relu_output)?;
    let leaky_relu_grad = leaky_relu.backward(&dummy_grad, &leaky_relu_output)?;
    let sigmoid_grad = sigmoid.backward(&dummy_grad, &sigmoid_output)?;
    let tanh_grad = tanh.backward(&dummy_grad, &tanh_output)?;
    let gelu_grad = gelu.backward(&dummy_grad, &gelu_output)?;
    let gelu_fast_grad = gelu_fast.backward(&dummy_grad, &gelu_fast_output)?;
    let swish_grad = swish.backward(&dummy_grad, &swish_output)?;
    let mish_grad = mish.backward(&dummy_grad, &mish_output)?;

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

    // Create plots to visualize the activation functions
    // Only if we're not in a headless environment
    if std::env::var("DISPLAY").is_ok() {
        println!("\nGenerating activation function plots...");

        let mut plot = Plot::new();

        // Add traces for each activation function
        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&relu_output))
                .name("ReLU")
                .line(Line::new().width(2.0)),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&leaky_relu_output))
                .name("Leaky ReLU (α=0.1)")
                .line(Line::new().width(2.0).dash("dash")),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&sigmoid_output))
                .name("Sigmoid")
                .line(Line::new().width(2.0)),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&tanh_output))
                .name("Tanh")
                .line(Line::new().width(2.0)),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&gelu_output))
                .name("GELU")
                .line(Line::new().width(2.0)),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&gelu_fast_output))
                .name("GELU Fast")
                .line(Line::new().width(2.0).dash("dot")),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&swish_output))
                .name("Swish (β=1)")
                .line(Line::new().width(2.0)),
        );

        plot.add_trace(
            Scatter::new(x_values.clone(), convert_to_vec(&mish_output))
                .name("Mish")
                .line(Line::new().width(2.0)),
        );

        // Set plot layout
        plot.set_layout(
            Layout::new()
                .title("Neural Network Activation Functions")
                .x_axis(plotly::layout::Axis::new().title("x"))
                .y_axis(plotly::layout::Axis::new().title("f(x)")),
        );

        // Save the plot to HTML file
        plot.write_html("activations.html");
        println!("Plot saved to activations.html");
    }

    Ok(())
}

fn convert_to_vec<F: Clone>(array: &Array<F, ndarray::IxDyn>) -> Vec<F> {
    array.iter().cloned().collect()
}
