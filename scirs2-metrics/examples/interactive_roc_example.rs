//! Example demonstrating the interactive ROC curve visualization
//!
//! This example shows how to create an interactive ROC curve with threshold adjustment,
//! which allows exploring different classification thresholds and their corresponding
//! performance metrics.

use scirs2_core::ndarray::array;
use scirs2_metrics::{
    classification::curves::roc_curve,
    visualization::{
        backends::{default_interactive_backend, PlotlyInteractiveBackendInterface},
        helpers, InteractiveOptions, VisualizationOptions,
    },
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating interactive ROC curve visualization example...");

    // Create binary classification data
    let y_true = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let y_score = array![
        0.1, 0.2, 0.3, 0.4, 0.55, 0.58, 0.6, 0.65, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.99
    ];

    println!("Computing ROC curve...");
    // Compute ROC curve
    let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score)?;

    // Print ROC curve points
    println!("ROC curve points: {} points", fpr.len());
    println!("First 5 points:");
    for i in 0..5.min(fpr.len()) {
        println!(
            "  FPR: {:.3}, TPR: {:.3}, Threshold: {:.3}",
            fpr[i], tpr[i], thresholds[i]
        );
    }

    // Create interactive options
    let interactive_options = InteractiveOptions {
        width: 900,
        height: 700,
        show_threshold_slider: true,
        show_metric_values: true,
        show_confusion_matrix: true,
        custom_layout: std::collections::HashMap::new(),
    };

    // Create an interactive ROC curve visualizer directly from labels and scores
    println!("Creating interactive ROC curve visualizer from labels and scores...");
    let roc_viz = helpers::visualize_interactive_roc_from_labels(
        y_true.view(),
        y_score.view(),
        None, // No specific positive label
        Some(interactive_options.clone()),
    )?;

    // Alternatively, create from pre-computed ROC curve values
    println!("Creating interactive ROC curve visualizer from pre-computed values...");
    let _roc_viz2 = helpers::visualize_interactive_roc_curve(
        fpr.view(),
        tpr.view(),
        Some(thresholds.view()),
        Some(0.94), // AUC value
        Some(interactive_options),
    );

    // Get visualization data and metadata
    let viz_data = roc_viz.prepare_data()?;
    let viz_metadata = roc_viz.get_metadata();

    // Get the interactive backend
    let backend = default_interactive_backend();

    // Save to HTML file
    println!("Saving interactive ROC curve to interactive_roc_curve.html...");
    backend.save_interactive_roc(
        &viz_data,
        &viz_metadata,
        &VisualizationOptions::default()
            .with_width(900)
            .with_height(700),
        "interactive_roc_curve.html",
    )?;

    println!("Interactive ROC curve saved to interactive_roc_curve.html");
    println!(
        "You can open this file in a web browser to interact with the ROC curve visualization."
    );
    println!("Features include:");
    println!("  - Adjusting the threshold to see different operating points");
    println!("  - Viewing performance metrics at the selected threshold");
    println!("  - Interactive legend that allows showing/hiding elements");
    println!();
    println!("Example complete! Check the output files for the interactive visualization.");

    Ok(())
}
