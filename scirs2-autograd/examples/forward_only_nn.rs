use ag::ndarray_ext::ArrayRng;
use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    println!("Forward-only neural network example");

    // Create a variable environment to manage our model parameters
    let mut env = ag::VariableEnvironment::new();

    // Initialize random number generator for weight initialization
    let mut rng = ArrayRng::<f32>::default();

    // Register variables (weights and biases) in the default namespace
    // Input dimension: 2, Hidden dimension: 3, Output dimension: 1
    env.name("w1").set(rng.glorot_uniform(&[2, 3]));
    env.name("b1").set(ag::ndarray_ext::zeros(&[1, 3]));
    env.name("w2").set(rng.glorot_uniform(&[3, 1]));
    env.name("b2").set(ag::ndarray_ext::zeros(&[1, 1]));

    // Generate some toy data (XOR problem)
    let x_data = ag::ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    // Execute the computation graph
    env.run(|ctx| {
        // Define placeholder for input with explicit batch size
        let batch_size = x_data.shape()[0] as isize;
        let x = ctx.placeholder("x", &[batch_size, 2]);

        // Get variables from the context
        let w1 = ctx.variable("w1");
        let b1 = ctx.variable("b1");
        let w2 = ctx.variable("w2");
        let b2 = ctx.variable("b2");

        // Forward pass - use only the most basic operations

        // First layer with ReLU activation
        let xw1 = matmul(x, w1);
        let z1 = add(xw1, b1);
        let h = relu(z1);

        // Second layer with sigmoid activation
        let hw2 = matmul(h, w2);
        let z2 = add(hw2, b2);
        let pred = sigmoid(z2);

        // Create feeder with input data
        let x_dyn = x_data.clone().into_dyn();
        let feeder = ag::Feeder::new().push(x, x_dyn.view());

        // Evaluate and print predictions
        let predictions = ctx.evaluator().push(&pred).set_feeder(feeder).run()[0]
            .clone()
            .unwrap();

        println!("\nPredictions:");
        println!("Shape of predictions: {:?}", predictions.shape());
        if !predictions.is_empty() {
            println!("Input    | Prediction");
            println!("---------|----------");
            for i in 0..4 {
                // Only print if predictions has valid dimensions
                if predictions.ndim() == 2 {
                    println!(
                        "{:.0}, {:.0}    | {:.6}",
                        x_data[[i, 0]],
                        x_data[[i, 1]],
                        predictions[[i, 0]]
                    );
                }
            }
        }
    });
}
