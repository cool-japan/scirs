use ag::ndarray_ext::ArrayRng;
use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    println!("Simplified neural network example");

    // Create a variable environment to manage our model parameters
    let mut env = ag::VariableEnvironment::new();

    // Initialize random number generator for weight initialization
    let mut rng = ArrayRng::<f32>::default();

    // Register variables (weights) in the default namespace
    // We'll avoid bias terms to simplify the graph
    // Input dimension: 2, Hidden dimension: 3, Output dimension: 1
    env.name("w1").set(rng.glorot_uniform(&[2, 3]));
    env.name("w2").set(rng.glorot_uniform(&[3, 1]));

    // Generate some toy data (XOR problem)
    let x_data = ag::ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    // Execute the computation graph
    env.run(|ctx| {
        // Define placeholder for input with explicit batch size
        let batch_size = x_data.shape()[0] as isize;
        let x = ctx.placeholder("x", &[batch_size, 2]);

        // Get variables from the context
        let w1 = ctx.variable("w1");
        let w2 = ctx.variable("w2");

        // Forward pass
        // First layer with ReLU activation
        let h = relu(matmul(x, w1));
        // Output layer with sigmoid activation
        let pred = sigmoid(matmul(h, w2));

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

            // Safe way to print predictions
            let num_rows = predictions.shape().get(0).cloned().unwrap_or(0);
            let num_cols = if predictions.ndim() > 1 {
                predictions.shape()[1]
            } else {
                1
            };

            for i in 0..std::cmp::min(4, num_rows) {
                if predictions.ndim() == 2 && num_cols > 0 {
                    println!(
                        "{:.0}, {:.0}    | {:.6}",
                        x_data[[i, 0]],
                        x_data[[i, 1]],
                        predictions[[i, 0]]
                    );
                } else {
                    println!(
                        "{:.0}, {:.0}    | Unable to get prediction (dimension mismatch)",
                        x_data[[i, 0]],
                        x_data[[i, 1]]
                    );
                }
            }
        }
    });
}
