//! Example demonstrating the Sharpness-Aware Minimization (SAM) optimizer
//!
//! This example shows how the SAM optimizer can be used to improve
//! generalization by minimizing both loss value and loss sharpness.
//! We compare SAM against standard optimizers on a linear regression task.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use scirs2_optim::optimizers::{Adam, Optimizer, SAM, SGD};
use std::time::Instant;

fn main() {
    println!("Sharpness-Aware Minimization (SAM) Optimizer Example");
    println!("===================================================\n");

    // Generate synthetic data for linear regression with noise
    let n_samples = 100;
    let n_features = 5;

    // True weights with some specific pattern to test generalization
    let true_weights = Array1::<f64>::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);

    // Generate input features
    let x_train = Array2::<f64>::random((n_samples, n_features), Normal::new(0.0, 1.0).unwrap());

    // Generate target values: y = X * w + noise
    let noise = Array1::<f64>::random(n_samples, Normal::new(0.0, 0.1).unwrap());
    let y_train = x_train.dot(&true_weights) + &noise;

    // Generate separate test set (with same distribution but different samples)
    let x_test = Array2::<f64>::random((n_samples, n_features), Normal::new(0.0, 1.0).unwrap());
    let y_test = x_test.dot(&true_weights); // No noise in test targets to measure true generalization

    println!("Linear Regression Task:");
    println!("  - {} samples, {} features", n_samples, n_features);
    println!("  - True weights: {:?}", true_weights);
    println!();

    // Parameters for optimization
    let lr = 0.03;
    let n_iterations = 200;

    // 1. Train with SGD
    println!("Training with SGD:");
    let start_time = Instant::now();
    let sgd = SGD::new(lr);
    let sgd_result = train_linear_regression_standard(
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        n_iterations,
        sgd,
        "SGD",
    );
    let sgd_time = start_time.elapsed();

    // 2. Train with Adam
    println!("\nTraining with Adam:");
    let start_time = Instant::now();
    let adam = Adam::new(lr);
    let adam_result = train_linear_regression_standard(
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        n_iterations,
        adam,
        "Adam",
    );
    let adam_time = start_time.elapsed();

    // 3. Train with SAM(SGD)
    println!("\nTraining with SAM(SGD):");
    let start_time = Instant::now();
    let sgd = SGD::new(lr);
    let sam_sgd_result = train_linear_regression_sam(
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        n_iterations,
        sgd,
        0.05,
        false,
        "SAM(SGD)",
    );
    let sam_sgd_time = start_time.elapsed();

    // 4. Train with SAM-A(SGD) [Adaptive SAM]
    println!("\nTraining with SAM-A(SGD) [Adaptive]:");
    let start_time = Instant::now();
    let sgd = SGD::new(lr);
    let sam_a_sgd_result = train_linear_regression_sam(
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        n_iterations,
        sgd,
        0.05,
        true,
        "SAM-A(SGD)",
    );
    let sam_a_sgd_time = start_time.elapsed();

    // 5. Train with SAM(Adam)
    println!("\nTraining with SAM(Adam):");
    let start_time = Instant::now();
    let adam = Adam::new(lr);
    let sam_adam_result = train_linear_regression_sam(
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        n_iterations,
        adam,
        0.05,
        false,
        "SAM(Adam)",
    );
    let sam_adam_time = start_time.elapsed();

    // Compare final results
    println!("\nFinal Results Comparison:");
    println!("  - True weights:     {:?}", true_weights);
    println!("  - SGD weights:      {:?}", sgd_result.weights);
    println!("  - Adam weights:     {:?}", adam_result.weights);
    println!("  - SAM(SGD):         {:?}", sam_sgd_result.weights);
    println!("  - SAM-A(SGD):       {:?}", sam_a_sgd_result.weights);
    println!("  - SAM(Adam):        {:?}", sam_adam_result.weights);

    // Print training metrics
    println!("\nTraining Time and Final Losses:");
    println!(
        "{:<12} {:<12} {:<12} {:<15} {:<15}",
        "Optimizer", "Train Time", "Train Loss", "Test Loss", "Weight Error"
    );
    println!(
        "{:-<12} {:-<12} {:-<12} {:-<15} {:-<15}",
        "", "", "", "", ""
    );

    println!(
        "{:<12} {:<12.3?} {:<12.6} {:<15.6} {:<15.6}",
        "SGD", sgd_time, sgd_result.train_loss, sgd_result.test_loss, sgd_result.weight_error
    );
    println!(
        "{:<12} {:<12.3?} {:<12.6} {:<15.6} {:<15.6}",
        "Adam", adam_time, adam_result.train_loss, adam_result.test_loss, adam_result.weight_error
    );
    println!(
        "{:<12} {:<12.3?} {:<12.6} {:<15.6} {:<15.6}",
        "SAM(SGD)",
        sam_sgd_time,
        sam_sgd_result.train_loss,
        sam_sgd_result.test_loss,
        sam_sgd_result.weight_error
    );
    println!(
        "{:<12} {:<12.3?} {:<12.6} {:<15.6} {:<15.6}",
        "SAM-A(SGD)",
        sam_a_sgd_time,
        sam_a_sgd_result.train_loss,
        sam_a_sgd_result.test_loss,
        sam_a_sgd_result.weight_error
    );
    println!(
        "{:<12} {:<12.3?} {:<12.6} {:<15.6} {:<15.6}",
        "SAM(Adam)",
        sam_adam_time,
        sam_adam_result.train_loss,
        sam_adam_result.test_loss,
        sam_adam_result.weight_error
    );

    // Calculate improvement percentages for test loss
    let sgd_improvement =
        (sgd_result.test_loss - sam_sgd_result.test_loss) / sgd_result.test_loss * 100.0;
    let adam_improvement =
        (adam_result.test_loss - sam_adam_result.test_loss) / adam_result.test_loss * 100.0;

    println!("\nSAM Improvement on Test Loss:");
    println!("  - SAM(SGD) vs SGD:      {:.2}%", sgd_improvement);
    println!("  - SAM(Adam) vs Adam:    {:.2}%", adam_improvement);

    println!("\nConclusions:");
    println!("  - SAM typically improves generalization performance (test loss)");
    println!("  - SAM methods may have slightly higher training loss but better test loss");
    println!(
        "  - Adaptive SAM (SAM-A) can provide additional benefits by adapting to parameter scales"
    );
    println!(
        "  - The performance improvement comes at the cost of approximately double training time"
    );
    println!("  - SAM is most effective when there is a gap between training and test performance");
}

/// Results structure to track metrics
struct TrainingResult {
    weights: Array1<f64>,
    train_loss: f64,
    test_loss: f64,
    weight_error: f64,
}

/// Trains a linear regression model using standard optimizers
fn train_linear_regression_standard<O>(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    x_test: &Array2<f64>,
    y_test: &Array1<f64>,
    n_iterations: usize,
    mut optimizer: O,
    name: &str,
) -> TrainingResult
where
    O: Optimizer<f64, ndarray::Ix1>,
{
    // Initialize weights to zeros
    let mut weights = Array1::<f64>::zeros(x_train.dim().1);

    // Training loop
    for i in 0..n_iterations {
        // Forward pass: predict y = X * w
        let y_pred = x_train.dot(&weights);

        // Compute MSE loss
        let diff = &y_pred - y_train;
        let loss = (&diff * &diff).sum() / (y_train.len() as f64);

        // Compute gradients: grad = 2 * X^T * (X*w - y) / n
        let gradients = x_train.t().dot(&diff) * (2.0 / (y_train.len() as f64));

        // Update weights
        weights = optimizer.step(&weights, &gradients).unwrap();

        // Print progress at intervals
        if i == 0 || i == n_iterations - 1 || (i + 1) % 50 == 0 {
            println!(
                "  {} - Iteration {}/{}: train loss = {:.6}",
                name,
                i + 1,
                n_iterations,
                loss
            );
        }
    }

    // Compute final metrics
    let train_pred = x_train.dot(&weights);
    let train_diff = &train_pred - y_train;
    let train_loss = (&train_diff * &train_diff).sum() / (y_train.len() as f64);

    let test_pred = x_test.dot(&weights);
    let test_diff = &test_pred - y_test;
    let test_loss = (&test_diff * &test_diff).sum() / (y_test.len() as f64);

    let weight_diff = &weights - &Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    let weight_error = (&weight_diff * &weight_diff).sum().sqrt();

    TrainingResult {
        weights,
        train_loss,
        test_loss,
        weight_error,
    }
}

/// Trains a linear regression model using SAM optimizer
fn train_linear_regression_sam<O>(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    x_test: &Array2<f64>,
    y_test: &Array1<f64>,
    n_iterations: usize,
    inner_optimizer: O,
    rho: f64,
    adaptive: bool,
    name: &str,
) -> TrainingResult
where
    O: Optimizer<f64, ndarray::Ix1> + Clone,
{
    // Initialize weights to zeros
    let mut weights = Array1::<f64>::zeros(x_train.dim().1);

    // Create SAM optimizer
    let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::with_config(inner_optimizer, rho, adaptive);

    // Training loop
    for i in 0..n_iterations {
        // Step 1: Forward pass with current weights
        let y_pred = x_train.dot(&weights);

        // Compute MSE loss
        let diff = &y_pred - y_train;
        let loss = (&diff * &diff).sum() / (y_train.len() as f64);

        // Compute gradients: grad = 2 * X^T * (X*w - y) / n
        let gradients = x_train.t().dot(&diff) * (2.0 / (y_train.len() as f64));

        // Step 2: SAM first step - get perturbed parameters
        let (perturbed_weights, _) = optimizer.first_step(&weights, &gradients).unwrap();

        // Step 3: Forward pass with perturbed weights
        let y_pred_perturbed = x_train.dot(&perturbed_weights);

        // Compute MSE loss at perturbed point
        let diff_perturbed = &y_pred_perturbed - y_train;
        let loss_perturbed = (&diff_perturbed * &diff_perturbed).sum() / (y_train.len() as f64);

        // Compute gradients at perturbed point
        let gradients_perturbed = x_train.t().dot(&diff_perturbed) * (2.0 / (y_train.len() as f64));

        // Step 4: SAM second step - update original weights using gradients at perturbed point
        weights = optimizer
            .second_step(&weights, &gradients_perturbed)
            .unwrap();

        // Print progress at intervals
        if i == 0 || i == n_iterations - 1 || (i + 1) % 50 == 0 {
            println!(
                "  {} - Iteration {}/{}: train loss = {:.6}, perturbed loss = {:.6}",
                name,
                i + 1,
                n_iterations,
                loss,
                loss_perturbed
            );
        }
    }

    // Compute final metrics
    let train_pred = x_train.dot(&weights);
    let train_diff = &train_pred - y_train;
    let train_loss = (&train_diff * &train_diff).sum() / (y_train.len() as f64);

    let test_pred = x_test.dot(&weights);
    let test_diff = &test_pred - y_test;
    let test_loss = (&test_diff * &test_diff).sum() / (y_test.len() as f64);

    let weight_diff = &weights - &Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    let weight_error = (&weight_diff * &weight_diff).sum().sqrt();

    TrainingResult {
        weights,
        train_loss,
        test_loss,
        weight_error,
    }
}
