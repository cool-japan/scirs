use ag::ndarray_ext::ArrayRng;
use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    println!("Creating a minimal neural network for binary classification");

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
    let y_data = ag::ndarray::array![[0.0], [1.0], [1.0], [0.0]];

    // Clone for later use
    let x_data_eval = x_data.clone();
    let y_data_eval = y_data.clone();

    // Training loop
    let num_epochs = 1000;
    for epoch in 0..num_epochs {
        // Execute the computation graph
        let loss = env.run(|ctx| {
            // Define placeholders for input and output with explicit batch size
            let batch_size = x_data.shape()[0] as isize;
            let x = ctx.placeholder("x", &[batch_size, 2]);
            let y = ctx.placeholder("y", &[batch_size, 1]);

            // Get variables from the context
            let w1 = ctx.variable("w1");
            let b1 = ctx.variable("b1");
            let w2 = ctx.variable("w2");
            let b2 = ctx.variable("b2");

            // Forward pass
            // First layer with ReLU activation
            let h = relu(add(matmul(x, w1), b1));
            // Output layer with sigmoid activation
            let logits = sigmoid(add(matmul(h, w2), b2));

            // Binary cross-entropy loss
            // Manual implementation with careful handling for numerical stability
            let epsilon = 1e-7;
            let one_minus_epsilon = 1.0 - epsilon;
            let clipped_logits = clip(logits, epsilon, one_minus_epsilon);

            // Create tensors of ones with the same shape as y and clipped_logits
            let ones_y = ones(&[batch_size, 1], ctx);
            let ones_logits = ones(&[batch_size, 1], ctx);

            let loss = neg(mean_all(add(
                mul(y, ln(clipped_logits)),
                mul(sub(ones_y, y), ln(sub(ones_logits, clipped_logits))),
            )));

            // Create feeder to provide input data
            let x_dyn = x_data.clone().into_dyn();
            let y_dyn = y_data.clone().into_dyn();
            let feeder = ag::Feeder::new()
                .push(x, x_dyn.view())
                .push(y, y_dyn.view());

            // Evaluate and return the loss
            ctx.evaluator().push(&loss).set_feeder(feeder.clone()).run()[0]
                .clone()
                .unwrap()[[]]
        });

        // Print progress every 100 epochs
        if epoch % 100 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }
    }

    // Evaluate the trained model
    env.run(|ctx| {
        // Get variables from the context
        let w1 = ctx.variable("w1");
        let b1 = ctx.variable("b1");
        let w2 = ctx.variable("w2");
        let b2 = ctx.variable("b2");

        // Define placeholder for input with explicit batch size
        let batch_size = x_data_eval.shape()[0] as isize;
        let x = ctx.placeholder("x", &[batch_size, 2]);

        // Forward pass
        let h = relu(add(matmul(x, w1), b1));
        let pred = sigmoid(add(matmul(h, w2), b2));

        // Create feeder with input data
        let x_eval_dyn = x_data_eval.clone().into_dyn();
        let feeder = ag::Feeder::new().push(x, x_eval_dyn.view());

        // Evaluate and print predictions
        let predictions = ctx.evaluator().push(&pred).set_feeder(feeder).run()[0]
            .clone()
            .unwrap();

        println!("\nPredictions:");
        println!("Shape of predictions: {:?}", predictions.shape());
        if !predictions.is_empty() {
            println!("Input    | Target | Prediction");
            println!("---------|--------|----------");
            for i in 0..4 {
                // Only print if predictions has valid dimensions
                if predictions.ndim() == 2 {
                    println!(
                        "{:.0}, {:.0}    | {:.0}      | {:.6}",
                        x_data_eval[[i, 0]],
                        x_data_eval[[i, 1]],
                        y_data_eval[[i, 0]],
                        predictions[[i, 0]]
                    );
                }
            }
        }
    });
}
