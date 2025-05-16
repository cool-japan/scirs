//! Examples demonstrating various visualization capabilities
//!
//! This example demonstrates how to use the visualization module to create various plots
//! for metrics results, such as ROC curves, confusion matrices, etc.

use ndarray::{array, Array2};
use scirs2_metrics::{
    classification::confusion_matrix,
    classification::curves::{calibration_curve, precision_recall_curve, roc_curve},
    visualization::{
        backends, helpers, ColorMap, PlotType, VisualizationData, VisualizationMetadata,
        VisualizationOptions,
    },
};

// Helper function to print what would be rendered
fn print_visualization_info(title: &str, plot_type: &PlotType, filename: &str) {
    println!(
        "Would render a {:?} plot titled '{}' to {}",
        plot_type, title, filename
    );
}

fn main() {
    // Example 1: Confusion Matrix Visualization
    println!("\n===== Example 1: Confusion Matrix Visualization =====");

    // Create a confusion matrix
    let y_true = array![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let y_pred = array![0, 2, 1, 0, 0, 2, 1, 1, 2];

    let (cm, _) = confusion_matrix(&y_true, &y_pred, None).unwrap();
    println!("Confusion Matrix:\n{:?}", cm);

    // Create a visualizer for the confusion matrix
    let cm_f64 = cm.mapv(|x| x as f64);
    let cm_viz = helpers::visualize_confusion_matrix(
        cm_f64.view(),
        Some(vec![
            "Class 0".to_string(),
            "Class 1".to_string(),
            "Class 2".to_string(),
        ]),
        false,
    );

    // Print what would be rendered
    let metadata = cm_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "confusion_matrix.png");

    // Example 2: ROC Curve Visualization
    println!("\n===== Example 2: ROC Curve Visualization =====");

    // Create binary classification data
    let y_true_binary = array![0, 1, 1, 0, 1, 0, 1, 0, 1];
    let y_score = array![0.1, 0.8, 0.7, 0.3, 0.9, 0.2, 0.6, 0.3, 0.8];

    // Compute ROC curve
    let (fpr, tpr, thresholds) = roc_curve(&y_true_binary, &y_score).unwrap();
    println!("ROC Curve Points: {} points", fpr.len());

    // Create a visualizer for the ROC curve
    let roc_viz = helpers::visualize_roc_curve(
        fpr.view(),
        tpr.view(),
        Some(thresholds.view()),
        Some(0.85), // Example AUC value
    );

    // Print what would be rendered
    let metadata = roc_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "roc_curve.png");

    // Example 3: Precision-Recall Curve Visualization
    println!("\n===== Example 3: Precision-Recall Curve Visualization =====");

    // Compute precision-recall curve
    let (precision, recall, pr_thresholds) =
        precision_recall_curve(&y_true_binary, &y_score).unwrap();
    println!("Precision-Recall Curve Points: {} points", precision.len());

    // Create a visualizer for the precision-recall curve
    let pr_viz = helpers::visualize_precision_recall_curve(
        precision.view(),
        recall.view(),
        Some(pr_thresholds.view()),
        Some(0.75), // Example average precision
    );

    // Print what would be rendered
    let metadata = pr_viz.get_metadata();
    print_visualization_info(
        &metadata.title,
        &metadata.plot_type,
        "precision_recall_curve.png",
    );

    // Example 4: Calibration Curve Visualization
    println!("\n===== Example 4: Calibration Curve Visualization =====");

    // Compute calibration curve
    let (prob_true, prob_pred, _) = calibration_curve(&y_true_binary, &y_score, Some(5)).unwrap();
    println!("Calibration Curve Points: {} points", prob_true.len());

    // Create a visualizer for the calibration curve
    let cal_viz =
        helpers::visualize_calibration_curve(prob_true.view(), prob_pred.view(), 5, "uniform");

    // Print what would be rendered
    let metadata = cal_viz.get_metadata();
    print_visualization_info(
        &metadata.title,
        &metadata.plot_type,
        "calibration_curve.png",
    );

    // Example 5: Learning Curve Visualization
    println!("\n===== Example 5: Learning Curve Visualization =====");

    // Create learning curve data
    let train_sizes = vec![10, 30, 50, 100, 200];
    let train_scores = vec![
        vec![0.6, 0.62, 0.64],  // 10 samples
        vec![0.7, 0.72, 0.74],  // 30 samples
        vec![0.75, 0.77, 0.79], // 50 samples
        vec![0.8, 0.82, 0.84],  // 100 samples
        vec![0.85, 0.87, 0.89], // 200 samples
    ];
    let val_scores = vec![
        vec![0.5, 0.52, 0.54],  // 10 samples
        vec![0.6, 0.62, 0.64],  // 30 samples
        vec![0.65, 0.67, 0.69], // 50 samples
        vec![0.7, 0.72, 0.74],  // 100 samples
        vec![0.75, 0.77, 0.79], // 200 samples
    ];

    // Create a visualizer for the learning curve
    let lc_viz =
        helpers::visualize_learning_curve(train_sizes, train_scores, val_scores, "Accuracy")
            .unwrap();

    // Print what would be rendered
    let metadata = lc_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "learning_curve.png");

    // Example 6: Generic Metric Visualization
    println!("\n===== Example 6: Generic Metric Visualization =====");

    // Create some metric data (e.g., loss over epochs)
    let epochs = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let loss = array![1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23];

    // Create a generic metric visualizer
    let metric_viz = helpers::visualize_metric(
        epochs.view(),
        loss.view(),
        "Training Loss",
        "Epoch",
        "Loss",
        PlotType::Line,
    );

    // Print what would be rendered
    let metadata = metric_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "training_loss.png");

    // Example 7: Multi-Curve Visualization
    println!("\n===== Example 7: Multi-Curve Visualization =====");

    // Create multi-curve data (e.g., comparing models)
    let epochs = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let model1_acc = array![0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 0.77, 0.78, 0.79];
    let model2_acc = array![0.55, 0.63, 0.68, 0.72, 0.75, 0.77, 0.79, 0.81, 0.82, 0.83];
    let model3_acc = array![0.45, 0.55, 0.62, 0.67, 0.7, 0.73, 0.75, 0.76, 0.78, 0.8];

    // Create a multi-curve visualizer
    let multi_viz = helpers::visualize_multi_curve(
        epochs.view(),
        vec![model1_acc.view(), model2_acc.view(), model3_acc.view()],
        vec![
            "Model 1".to_string(),
            "Model 2".to_string(),
            "Model 3".to_string(),
        ],
        "Model Comparison",
        "Epoch",
        "Accuracy",
    );

    // Print what would be rendered
    let metadata = multi_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "model_comparison.png");

    // Example 8: Heatmap Visualization
    println!("\n===== Example 8: Heatmap Visualization =====");

    // Create a correlation matrix (for example)
    let correlation_matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 0.8, 0.3, 0.8, 1.0, 0.5, 0.3, 0.5, 1.0]).unwrap();

    // Create a heatmap visualizer
    let heatmap_viz = helpers::visualize_heatmap(
        correlation_matrix.view(),
        Some(vec![
            "Feature 1".to_string(),
            "Feature 2".to_string(),
            "Feature 3".to_string(),
        ]),
        Some(vec![
            "Feature 1".to_string(),
            "Feature 2".to_string(),
            "Feature 3".to_string(),
        ]),
        "Feature Correlation Matrix",
        Some(ColorMap::BlueRed),
    );

    // Print what would be rendered
    let metadata = heatmap_viz.get_metadata();
    print_visualization_info(
        &metadata.title,
        &metadata.plot_type,
        "correlation_matrix.png",
    );

    // Example 9: Histogram Visualization
    println!("\n===== Example 9: Histogram Visualization =====");

    // Create some data for a histogram
    let values = array![
        1.2, 1.5, 2.1, 2.3, 2.5, 2.7, 3.0, 3.1, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.5,
        4.7, 4.9, 5.0, 5.5, 6.0
    ];

    // Create a histogram visualizer
    let hist_viz = helpers::visualize_histogram(
        values.view(),
        10,
        "Value Distribution",
        "Value",
        Some("Count".to_string()),
    );

    // Print what would be rendered
    let metadata = hist_viz.get_metadata();
    print_visualization_info(&metadata.title, &metadata.plot_type, "histogram.png");

    // Example 10: Custom Visualization with Options
    println!("\n===== Example 10: Custom Visualization with Options =====");

    // Create custom data
    let x = (0..100).map(|i| i as f64 / 10.0).collect::<Vec<f64>>();
    let y = x.iter().map(|&val| val.sin()).collect::<Vec<f64>>();

    // Create visualization data and metadata
    let mut data = VisualizationData::new();
    data.x = x;
    data.y = y;

    let metadata = VisualizationMetadata::line_plot("Sine Wave", "X", "sin(X)");

    // Create custom options
    let _options = VisualizationOptions::new()
        .with_width(1200)
        .with_height(800)
        .with_dpi(150)
        .with_grid(true)
        .with_line_width(2.0)
        .with_font_sizes(Some(16.0), Some(14.0), Some(12.0))
        .with_color_palette("Set1");

    // Get the default backend
    let _backend = backends::default_backend();

    // Print what would be rendered
    println!(
        "Would render a {:?} plot titled '{}' to sine_wave.png with custom options",
        metadata.plot_type, metadata.title
    );

    println!("\nAll visualization examples completed!");
}
