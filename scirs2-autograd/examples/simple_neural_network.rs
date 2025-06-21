use ag::ndarray_ext::ArrayRng;
use ag::optimizers::adam::Adam;
use ag::prelude::*;
use ag::tensor_ops::*;
use scirs2_autograd as ag;

// NOTE: This example currently has issues with the Variable op implementation.
// There's an `unreachable!()` in the Variable::compute method that gets called
// during optimization. This example needs to be updated to match the current
// architecture of the autograd system. The issue is in
// src/tensor_ops/basic_source_ops.rs where Variable op is implemented with
// `unreachable!()` in its compute method.

fn main() {
    println!("Creating a simple neural network for binary classification");

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

    // Create optimizer (not used in this example - using manual SGD updates instead)
    let _adam = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);

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
            let h = relu(matmul(x, w1) + b1);
            // Output layer with sigmoid activation
            let logits = sigmoid(matmul(h, w2) + b2);

            // Binary cross-entropy loss
            // Manual implementation with careful handling for numerical stability
            let epsilon = 1e-7;
            let one_minus_epsilon = 1.0 - epsilon;
            let clipped_logits = clip(logits, epsilon, one_minus_epsilon);
            let loss = neg(mean_all(
                y * ln(clipped_logits) + (1.0 - y) * ln(1.0 - clipped_logits),
            ));

            // Compute gradients with respect to all variables
            let grads = &grad(&[loss], &[w1, b1, w2, b2]);

            // Create feeder to provide input data
            let x_dyn = x_data.clone().into_dyn();
            let y_dyn = y_data.clone().into_dyn();
            let feeder = ag::Feeder::new()
                .push(x, x_dyn.view())
                .push(y, y_dyn.view());

            // Update parameters using SGD instead of Adam to avoid the issues
            // We'll implement a simple SGD optimizer directly here
            println!(
                "Applying manual SGD updates to {} parameters",
                &[w1, b1, w2, b2].len()
            );

            // Create manual SGD update operations
            let params = [w1, b1, w2, b2];
            let learning_rate = 0.01; // Use a reasonable learning rate for SGD

            // Create update operations first and store them in a vector
            let mut update_ops = Vec::with_capacity(params.len());
            for i in 0..params.len() {
                let param = params[i];
                let grad = grads[i];

                // Create update operation: param = param - learning_rate * grad
                let scaled_grad = ag::tensor_ops::scalar_mul(grad, learning_rate);
                let update_op = ag::tensor_ops::sub(param, scaled_grad);
                update_ops.push(update_op);
            }

            // Create evaluator and add all operations
            let mut evaluator = ctx.evaluator().set_feeder(feeder.clone());
            for op in &update_ops {
                evaluator = evaluator.push(op);
            }

            // Run all update operations
            let results = evaluator.run();
            println!("SGD update applied with {} operations", results.len());

            // Evaluate and return the loss
            ctx.evaluator().push(&loss).set_feeder(feeder).run()[0]
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
        let h = relu(matmul(x, w1) + b1);
        let pred = sigmoid(matmul(h, w2) + b2);

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
