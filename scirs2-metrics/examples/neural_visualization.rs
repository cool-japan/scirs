//! Example of visualizing neural network metrics
//!
//! This example shows how to use visualizations for neural network metrics.
//! To run this example, enable the 'neural_common' feature:
//!
//! ```bash
//! cargo run --example neural_visualization --features neural_common
//! ```

use std::error::Error;

#[cfg(feature = "neural_common")]
use scirs2_metrics::integration::neural;

#[cfg(feature = "neural_common")]
use std::collections::HashMap;

#[cfg(feature = "neural_common")]
use scirs2_core::ndarray::Array1;

#[cfg(feature = "neural_common")]
use scirs2_core::random::Rng;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "neural_common"))]
    {
        println!("This example requires the 'neural_common' feature to be enabled.");
        println!("Run with: cargo run --example neural_visualization --features neural_common");
        Ok(())
    }

    #[cfg(feature = "neural_common")]
    {
        println!("Neural Network Metrics Visualization Example");
        println!("-----------------------------------------");

        // Generate synthetic training history data
        let num_epochs = 10;
        let metrics = vec!["loss".to_string(), "accuracy".to_string()];

        let mut history = Vec::with_capacity(num_epochs);
        let mut val_history = Vec::with_capacity(num_epochs);

        let mut rng = scirs2_core::random::rng();

        // Generate improving metrics over epochs
        for epoch in 0..num_epochs {
            let progress = epoch as f64 / (num_epochs - 1) as f64;

            // Training metrics - loss decreases, accuracy increases
            let mut epoch_metrics = HashMap::new();
            epoch_metrics.insert(
                "loss".to_string(),
                1.0 - 0.8 * progress + rng.random::<f64>() * 0.1,
            );
            epoch_metrics.insert(
                "accuracy".to_string(),
                0.5 + 0.4 * progress + rng.random::<f64>() * 0.05,
            );
            history.push(epoch_metrics);

            // Validation metrics - slightly worse than training
            let mut val_epoch_metrics = HashMap::new();
            val_epoch_metrics.insert(
                "loss".to_string(),
                1.1 - 0.7 * progress + rng.random::<f64>() * 0.15,
            );
            val_epoch_metrics.insert(
                "accuracy".to_string(),
                0.45 + 0.35 * progress + rng.random::<f64>() * 0.08,
            );
            val_history.push(val_epoch_metrics);
        }

        // Create training history visualization
        let training_viz = neural::training_history_visualization(
            metrics.clone(),
            history.clone(),
            Some(val_history.clone()),
        );

        // Get the visualization data and metadata
        let viz_data = training_viz.prepare_data()?;
        let viz_metadata = training_viz.get_metadata();

        println!("Created training history visualization:");
        println!("  Title: {}", viz_metadata.title);
        println!("  Plot type: {:?}", viz_metadata.plot_type);
        println!("  X label: {}", viz_metadata.x_label);
        println!("  Y label: {}", viz_metadata.y_label);
        println!(
            "  Data series: {}",
            viz_data
                .series_names
                .as_ref()
                .map(|names| names.join(", "))
                .unwrap_or_else(|| "None".to_string())
        );

        // Generate binary classification data for ROC and PR curves
        let n_samples = 100;
        let y_true = Array1::from_vec(
            (0..n_samples)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
        );
        let mut y_score = Array1::zeros(n_samples);

        for i in 0..n_samples {
            if y_true[i] > 0.5 {
                // For positive samples, generate scores mostly > 0.5
                y_score[i] = 0.7 + rng.random::<f64>() * 0.3;
                // Add some false negatives
                if rng.random::<f64>() < 0.2 {
                    y_score[i] = rng.random::<f64>() * 0.5;
                }
            } else {
                // For negative samples, generate scores mostly < 0.5
                y_score[i] = rng.random::<f64>() * 0.5;
                // Add some false positives
                if rng.random::<f64>() < 0.1 {
                    y_score[i] = 0.5 + rng.random::<f64>() * 0.5;
                }
            }
        }

        // Create ROC curve visualization
        let roc_viz = neural::neural_roc_curve_visualization(
            &y_true.clone().into_dyn(),
            &y_score.clone().into_dyn(),
            Some(0.85), // Example AUC value
        )?;

        // Get the visualization data and metadata
        let roc_data = roc_viz.prepare_data()?;
        let roc_metadata = roc_viz.get_metadata();

        println!("\nCreated ROC curve visualization:");
        println!("  Title: {}", roc_metadata.title);
        println!("  Plot type: {:?}", roc_metadata.plot_type);
        println!("  X label: {}", roc_metadata.x_label);
        println!("  Y label: {}", roc_metadata.y_label);
        println!(
            "  Data series: {}",
            roc_data
                .series_names
                .as_ref()
                .map(|names| names.join(", "))
                .unwrap_or_else(|| "None".to_string())
        );

        // Create Precision-Recall curve visualization
        let pr_viz = neural::neural_precision_recall_curve_visualization(
            &y_true.clone().into_dyn(),
            &y_score.clone().into_dyn(),
            Some(0.75), // Example Average Precision value
        )?;

        // Get the visualization data and metadata
        let pr_data = pr_viz.prepare_data()?;
        let pr_metadata = pr_viz.get_metadata();

        println!("\nCreated Precision-Recall curve visualization:");
        println!("  Title: {}", pr_metadata.title);
        println!("  Plot type: {:?}", pr_metadata.plot_type);
        println!("  X label: {}", pr_metadata.x_label);
        println!("  Y label: {}", pr_metadata.y_label);
        println!(
            "  Data series: {}",
            pr_data
                .series_names
                .as_ref()
                .map(|names| names.join(", "))
                .unwrap_or_else(|| "None".to_string())
        );

        // Create multiclass data for confusion matrix
        let classes = 3;
        let n_samples = 50;

        let mut y_true_multi = Array1::zeros(n_samples);
        let mut y_pred_multi = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // True class
            let true_class = i % classes;
            y_true_multi[i] = true_class as f64;

            // Predicted class - mostly correct, some errors
            let pred_class = if rng.random::<f64>() < 0.7 {
                true_class // Correct prediction
            } else {
                (true_class + 1 + (rng.random::<f64>() * (classes as f64 - 2.0)) as usize) % classes
                // Error
            };
            y_pred_multi[i] = pred_class as f64;
        }

        // Create confusion matrix visualization
        let cm_viz = neural::neural_confusion_matrix_visualization(
            &y_true_multi.clone().into_dyn(),
            &y_pred_multi.clone().into_dyn(),
            Some(vec![
                "Class A".to_string(),
                "Class B".to_string(),
                "Class C".to_string(),
            ]),
            false,
        )?;

        // Get the visualization data and metadata
        let cm_data = cm_viz.prepare_data()?;
        let cm_metadata = cm_viz.get_metadata();

        println!("\nCreated confusion matrix visualization:");
        println!("  Title: {}", cm_metadata.title);
        println!("  Plot type: {:?}", cm_metadata.plot_type);
        println!(
            "  Data series: {}",
            cm_data
                .series_names
                .as_ref()
                .map(|names| names.join(", "))
                .unwrap_or_else(|| "None".to_string())
        );

        println!("\nAll visualizations created successfully!");

        Ok(())
    }
}
