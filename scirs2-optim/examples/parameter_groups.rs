//! Example demonstrating parameter groups with different learning rates
//!
//! This example shows how to use different learning rates for different
//! parts of a model, which is common in transfer learning and fine-tuning.

use ndarray::{Array1, Array2};
use scirs2_optim::{
    optimizers::GroupedAdam,
    parameter_groups::{GroupedOptimizer, ParameterGroupConfig},
};

/// Simulate a neural network with feature extractor and classifier layers
struct SimpleNetwork {
    // Feature extractor layers (pretrained, should train slowly)
    feature_layer1: Array2<f64>,
    feature_layer2: Array2<f64>,

    // Classifier layers (new, should train faster)
    classifier_layer1: Array2<f64>,
    classifier_layer2: Array2<f64>,
    output_layer: Array1<f64>,
}

impl SimpleNetwork {
    fn new() -> Self {
        Self {
            feature_layer1: Array2::from_shape_vec((10, 5), vec![0.1; 50]).unwrap(),
            feature_layer2: Array2::from_shape_vec((8, 10), vec![0.1; 80]).unwrap(),
            classifier_layer1: Array2::from_shape_vec((6, 8), vec![0.1; 48]).unwrap(),
            classifier_layer2: Array2::from_shape_vec((4, 6), vec![0.1; 24]).unwrap(),
            output_layer: Array1::from_vec(vec![0.1; 4]),
        }
    }

    /// Get parameters as vectors for each group
    fn get_feature_params(&self) -> Vec<Array2<f64>> {
        vec![self.feature_layer1.clone(), self.feature_layer2.clone()]
    }

    fn get_classifier_params(&self) -> Vec<Array2<f64>> {
        vec![
            self.classifier_layer1.clone(),
            self.classifier_layer2.clone(),
        ]
    }

    fn get_output_params(&self) -> Vec<Array1<f64>> {
        vec![self.output_layer.clone()]
    }
}

/// Simulate gradients for demonstration
fn compute_gradients_2d(params: &[Array2<f64>]) -> Vec<Array2<f64>> {
    params
        .iter()
        .map(|p| Array2::from_shape_vec(p.dim(), vec![0.01; p.len()]).unwrap())
        .collect()
}

fn compute_gradients_1d(params: &[Array1<f64>]) -> Vec<Array1<f64>> {
    params
        .iter()
        .map(|p| Array1::from_vec(vec![0.01; p.len()]))
        .collect()
}

fn main() {
    println!("Parameter Groups Example");
    println!("========================\n");

    // Create a simple network
    let network = SimpleNetwork::new();

    // Create grouped optimizer with different learning rates
    let mut optimizer_2d = GroupedAdam::new(0.001); // Default learning rate
    let mut optimizer_1d = GroupedAdam::new(0.001);

    // Feature extractor group (pretrained layers - slow learning)
    let feature_config = ParameterGroupConfig::new()
        .with_learning_rate(0.0001)  // 10x slower than default
        .with_weight_decay(0.0001);

    let feature_params = network.get_feature_params();
    let feature_group = optimizer_2d
        .add_group(feature_params.clone(), feature_config)
        .unwrap();

    // Classifier group (new layers - normal learning)
    let classifier_config = ParameterGroupConfig::new()
        .with_learning_rate(0.001)  // Normal learning rate
        .with_weight_decay(0.0);

    let classifier_params = network.get_classifier_params();
    let classifier_group = optimizer_2d
        .add_group(classifier_params.clone(), classifier_config)
        .unwrap();

    // Output layer group (new layer - fast learning)
    let output_config = ParameterGroupConfig::new()
        .with_learning_rate(0.01)  // 10x faster than default
        .with_weight_decay(0.0);

    let output_params = network.get_output_params();
    let output_group = optimizer_1d
        .add_group(output_params.clone(), output_config)
        .unwrap();

    println!("Created parameter groups:");
    println!(
        "- Feature extractor: {} parameters, LR = 0.0001",
        feature_params.iter().map(|p| p.len()).sum::<usize>()
    );
    println!(
        "- Classifier: {} parameters, LR = 0.001",
        classifier_params.iter().map(|p| p.len()).sum::<usize>()
    );
    println!(
        "- Output layer: {} parameters, LR = 0.01\n",
        output_params.iter().map(|p| p.len()).sum::<usize>()
    );

    // Training loop simulation
    println!("Training simulation:");
    for epoch in 0..5 {
        println!("\nEpoch {}:", epoch);

        // Compute gradients for each group
        let feature_grads = compute_gradients_2d(&feature_params);
        let classifier_grads = compute_gradients_2d(&classifier_params);
        let output_grads = compute_gradients_1d(&output_params);

        // Update feature extractor (slow)
        let updated_features = optimizer_2d
            .step_group(feature_group, &feature_grads)
            .unwrap();
        let feature_change: f64 = updated_features
            .iter()
            .zip(feature_params.iter())
            .map(|(new, old)| (new - old).mapv(|x| x.abs()).sum())
            .sum();

        // Update classifier (normal)
        let updated_classifier = optimizer_2d
            .step_group(classifier_group, &classifier_grads)
            .unwrap();
        let classifier_change: f64 = updated_classifier
            .iter()
            .zip(classifier_params.iter())
            .map(|(new, old)| (new - old).mapv(|x| x.abs()).sum())
            .sum();

        // Update output layer (fast)
        let updated_output = optimizer_1d
            .step_group(output_group, &output_grads)
            .unwrap();
        let output_change: f64 = updated_output
            .iter()
            .zip(output_params.iter())
            .map(|(new, old)| (new - old).mapv(|x| x.abs()).sum())
            .sum();

        println!("  Feature extractor change: {:.6}", feature_change);
        println!("  Classifier change: {:.6}", classifier_change);
        println!("  Output layer change: {:.6}", output_change);

        // Verify relative learning rates
        if epoch > 0 {
            println!(
                "  Ratio (classifier/feature): {:.1}x",
                classifier_change / feature_change
            );
            println!(
                "  Ratio (output/classifier): {:.1}x",
                output_change / classifier_change
            );
        }
    }

    // Demonstrate dynamic learning rate adjustment
    println!("\n\nDynamic learning rate adjustment:");

    // Reduce learning rate for classifier after a few epochs
    optimizer_2d
        .set_group_learning_rate(classifier_group, 0.0001)
        .unwrap();
    println!("Reduced classifier learning rate to 0.0001");

    // One more epoch with adjusted rates
    let feature_grads = compute_gradients_2d(&feature_params);
    let classifier_grads = compute_gradients_2d(&classifier_params);

    let updated_features = optimizer_2d
        .step_group(feature_group, &feature_grads)
        .unwrap();
    let updated_classifier = optimizer_2d
        .step_group(classifier_group, &classifier_grads)
        .unwrap();

    let feature_change: f64 = updated_features
        .iter()
        .zip(feature_params.iter())
        .map(|(new, old)| (new - old).mapv(|x| x.abs()).sum())
        .sum();

    let classifier_change: f64 = updated_classifier
        .iter()
        .zip(classifier_params.iter())
        .map(|(new, old)| (new - old).mapv(|x| x.abs()).sum())
        .sum();

    println!("\nAfter adjustment:");
    println!("  Feature extractor change: {:.6}", feature_change);
    println!("  Classifier change: {:.6}", classifier_change);
    println!(
        "  Now they should be similar: ratio = {:.2}x",
        classifier_change / feature_change
    );

    // Show parameter group information
    println!("\n\nParameter group summary:");
    for (i, group) in optimizer_2d.groups().iter().enumerate() {
        println!(
            "Group {}: {} parameters, LR = {:?}",
            i,
            group.num_params(),
            group.config.learning_rate
        );
    }
}
