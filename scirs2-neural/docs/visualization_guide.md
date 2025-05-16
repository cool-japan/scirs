# Visualization Guide for Neural Network Training

This document provides usage examples for the visualization tools available in the scirs2-neural crate.

## Overview

The visualization module in scirs2-neural provides:

1. ASCII plotting for visualizing training metrics in the terminal
2. CSV export for detailed analysis in external tools
3. Learning rate scheduling utilities
4. Training history analysis for detecting issues

## Basic Usage

### ASCII Plotting

```rust
use scirs2_neural::utils::{ascii_plot, PlotOptions};
use std::collections::HashMap;

// Create a simple history with training and validation loss
let mut history = HashMap::new();
history.insert("train_loss".to_string(), vec![0.8, 0.6, 0.4, 0.3, 0.25]);
history.insert("val_loss".to_string(), vec![0.85, 0.7, 0.5, 0.4, 0.38]);

// Generate an ASCII plot
let plot = ascii_plot(
    &history,
    Some("Training Progress"),
    Some(PlotOptions::default()),
).unwrap();

// Print the plot to the terminal
println!("{}", plot);
```

### CSV Export

```rust
use scirs2_neural::utils::export_history_to_csv;
use std::collections::HashMap;

// Create training history
let mut history = HashMap::new();
history.insert("train_loss".to_string(), vec![0.8, 0.6, 0.4, 0.3, 0.25]);
history.insert("val_loss".to_string(), vec![0.85, 0.7, 0.5, 0.4, 0.38]);
history.insert("accuracy".to_string(), vec![0.6, 0.7, 0.8, 0.85, 0.88]);

// Export to CSV file
export_history_to_csv(&history, "training_history.csv").unwrap();
```

### Learning Rate Scheduling

```rust
use scirs2_neural::utils::LearningRateSchedule;

// Create a step decay learning rate schedule
let schedule = LearningRateSchedule::StepDecay {
    initial_lr: 0.001,
    decay_factor: 0.5,
    step_size: 3,
};

// Get learning rate for a specific epoch
let lr_epoch_5 = schedule.get_learning_rate(5);

// Generate learning rates for all epochs
let num_epochs = 10;
let learning_rates = schedule.generate_schedule(num_epochs);

// Print learning rates
for (epoch, &lr) in learning_rates.iter().enumerate() {
    println!("Epoch {}: Learning rate = {:.6}", epoch + 1, lr);
}
```

### Training History Analysis

```rust
use scirs2_neural::utils::analyze_training_history;
use std::collections::HashMap;

// Create training history
let mut history = HashMap::new();
history.insert("train_loss".to_string(), vec![0.8, 0.6, 0.4, 0.3, 0.25]);
history.insert("val_loss".to_string(), vec![0.85, 0.7, 0.55, 0.52, 0.5]);
history.insert("accuracy".to_string(), vec![0.6, 0.7, 0.8, 0.82, 0.83]);

// Analyze training history
let analysis = analyze_training_history(&history);

// Print analysis results
println!("Training Analysis:");
for issue in analysis {
    println!("  {}", issue);
}
```

## Using the VisualizationCallback

The VisualizationCallback provides automatic visualization during training:

```rust
use scirs2_neural::callbacks::VisualizationCallback;
use scirs2_neural::utils::PlotOptions;

// Create visualization callback with default options
let vis_callback = VisualizationCallback::new(1) // Update every epoch
    .with_save_path("./outputs/training_plot.txt")
    .with_tracked_metrics(vec![
        "train_loss".to_string(),
        "val_loss".to_string(),
        "accuracy".to_string(),
    ]);

// Add to your training callbacks
let mut callbacks: Vec<Box<dyn Callback<f32>>> = vec![
    // Other callbacks
    Box::new(vis_callback),
];

// Train your model with these callbacks
```

## Customizing Plots

You can customize your plots with the PlotOptions struct:

```rust
use scirs2_neural::utils::{ascii_plot, PlotOptions};

let custom_options = PlotOptions {
    width: 100,                // Width of the plot
    height: 30,                // Height of the plot
    max_x_ticks: 10,           // Number of x-axis ticks
    max_y_ticks: 6,            // Number of y-axis ticks
    line_char: '·',            // Character for grid lines
    point_char: '█',           // Character for data points
    background_char: ' ',      // Background character
    show_grid: true,           // Show grid lines
    show_legend: true,         // Show legend
};

// Use custom options when generating plot
let plot = ascii_plot(&history, Some("Custom Training Plot"), Some(custom_options)).unwrap();
```

## Complete Training Loop Example

See the `training_loop_example.rs` file in the examples directory for a complete working example of a training loop with visualization features.

## Best Practices

1. **Monitor Multiple Metrics**: Track both losses and accuracy metrics to get a complete picture of training.

2. **Use Early Stopping**: Combine visualizations with early stopping to prevent overfitting.

3. **Export History**: Always export your training history to CSV for further analysis.

4. **Analyze Trends**: Use the `analyze_training_history` function to identify potential issues.

5. **Schedule Learning Rates**: Implement learning rate scheduling to improve convergence.

6. **Customize Visualization**: Use PlotOptions to customize visualizations for better readability.

7. **Regular Evaluation**: Visualize training progress frequently to catch problems early.