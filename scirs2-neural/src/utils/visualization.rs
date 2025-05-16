use crate::error::{NeuralError, Result};
use ndarray::Array1;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Represents ASCII plotting options
#[derive(Clone, Debug)]
pub struct PlotOptions {
    /// Width of the plot in characters
    pub width: usize,
    /// Height of the plot in characters
    pub height: usize,
    /// Maximum number of ticks on x-axis
    pub max_x_ticks: usize,
    /// Maximum number of ticks on y-axis  
    pub max_y_ticks: usize,
    /// Character to use for the plot line
    pub line_char: char,
    /// Character to use for plot points
    pub point_char: char,
    /// Character to use for the plot background
    pub background_char: char,
    /// Whether to show a grid
    pub show_grid: bool,
    /// Whether to show a legend
    pub show_legend: bool,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            width: 80,
            height: 20,
            max_x_ticks: 10,
            max_y_ticks: 5,
            line_char: '─',
            point_char: '●',
            background_char: ' ',
            show_grid: true,
            show_legend: true,
        }
    }
}

/// Simple ASCII plotting function for training metrics visualization
///
/// # Arguments
///
/// * `data` - Map of series names to data arrays
/// * `title` - Optional title for the plot
/// * `options` - Optional plot options
///
/// # Returns
///
/// * `Result<String>` - The ASCII plot as a string
pub fn ascii_plot<F: num_traits::Float + std::fmt::Display + std::fmt::Debug>(
    data: &HashMap<String, Vec<F>>,
    title: Option<&str>,
    options: Option<PlotOptions>,
) -> Result<String> {
    let options = options.unwrap_or_default();
    let width = options.width;
    let height = options.height;

    if data.is_empty() {
        return Err(NeuralError::ValidationError("No data to plot".to_string()));
    }

    // Find the global min and max for y-axis scaling
    let mut min_y = F::infinity();
    let mut max_y = F::neg_infinity();
    let mut max_len = 0;

    for values in data.values() {
        if values.is_empty() {
            continue;
        }

        max_len = max_len.max(values.len());

        for &v in values {
            if v.is_finite() {
                min_y = min_y.min(v);
                max_y = max_y.max(v);
            }
        }
    }

    if max_len == 0 {
        return Err(NeuralError::ValidationError(
            "All data series are empty".to_string(),
        ));
    }

    if !min_y.is_finite() || !max_y.is_finite() {
        return Err(NeuralError::ValidationError(
            "Data contains non-finite values".to_string(),
        ));
    }

    // Add a small margin to the y-range
    let y_range = max_y - min_y;
    let margin = y_range * F::from(0.05).unwrap();
    min_y = min_y - margin;
    max_y = max_y + margin;

    // If min and max are the same, create a small range
    if (max_y - min_y).abs() < F::epsilon() {
        min_y = min_y - F::from(0.5).unwrap();
        max_y = max_y + F::from(0.5).unwrap();
    }

    // Create the plot canvas
    let mut plot = vec![vec![options.background_char; width]; height];

    // Draw grid if enabled
    if options.show_grid {
        for (y, row) in plot.iter_mut().enumerate().take(height) {
            for (x, cell) in row.iter_mut().enumerate().take(width) {
                if x % (width / options.max_x_ticks.max(1)) == 0
                    && y % (height / options.max_y_ticks.max(1)) == 0
                {
                    *cell = '·';
                }
            }
        }
    }

    // Draw axes
    for row in plot.iter_mut().take(height) {
        row[0] = '│';
    }

    for x in 0..width {
        plot[height - 1][x] = '─';
    }

    plot[height - 1][0] = '└';

    // Plot each series with different symbols
    let symbols = ['●', '■', '▲', '◆', '★', '✖', '◎'];

    let mut result = String::with_capacity(height * (width + 2) + 100);

    // Add title if provided
    if let Some(title) = title {
        let title_padding = (width - title.len()) / 2;
        result.push_str(&" ".repeat(title_padding));
        result.push_str(title);
        result.push('\n');
        result.push('\n');
    }

    let mut legend_entries = Vec::new();

    for (i, (name, values)) in data.iter().enumerate() {
        let symbol = symbols[i % symbols.len()];

        if values.is_empty() {
            continue;
        }

        // Store legend entry
        legend_entries.push((name, symbol));

        // Plot the series
        for (x_idx, &y_val) in values.iter().enumerate() {
            if !y_val.is_finite() {
                continue;
            }

            let x = ((x_idx as f64) / (max_len as f64 - 1.0) * (width as f64 - 2.0)).round()
                as usize
                + 1;

            if x >= width {
                continue;
            }

            let y_norm = ((y_val - min_y) / (max_y - min_y)).to_f64().unwrap();
            let y = height - (y_norm * (height as f64 - 2.0)).round() as usize - 1;

            if y < height {
                plot[y][x] = symbol;
            }
        }
    }

    // Render the plot
    let y_ticks = (0..options.max_y_ticks.min(height))
        .map(|i| {
            let val = max_y
                - F::from(i as f64 / (options.max_y_ticks as f64 - 1.0)).unwrap() * (max_y - min_y);
            format!("{:.2}", val)
        })
        .collect::<Vec<_>>();

    let max_y_tick_width = y_ticks.iter().map(|t| t.len()).max().unwrap_or(0);

    for y in 0..height {
        // Add y-axis ticks for specific rows
        if y % (height / options.max_y_ticks.max(1)) == 0 && y < y_ticks.len() {
            let tick = &y_ticks[y];
            result.push_str(&format!("{:>width$} ", tick, width = max_y_tick_width));
        } else {
            result.push_str(&" ".repeat(max_y_tick_width + 1));
        }

        // Add the plot row
        for x in 0..width {
            result.push(plot[y][x]);
        }

        result.push('\n');
    }

    // Add x-axis labels
    result.push_str(&" ".repeat(max_y_tick_width + 1));
    for i in 0..options.max_x_ticks {
        let _x = i * width / options.max_x_ticks;
        let epoch = (i as f64 * (max_len as f64 - 1.0) / (options.max_x_ticks as f64 - 1.0)).round()
            as usize;

        let tick = format!("{}", epoch);
        let padding = width / options.max_x_ticks - tick.len();
        let left_padding = padding / 2;
        let right_padding = padding - left_padding;

        result.push_str(&" ".repeat(left_padding));
        result.push_str(&tick);
        result.push_str(&" ".repeat(right_padding));
    }

    result.push('\n');

    // Add legend if enabled
    if options.show_legend && !legend_entries.is_empty() {
        result.push('\n');
        result.push_str("Legend: ");

        for (i, (name, symbol)) in legend_entries.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }
            result.push_str(&format!("{} {}", symbol, name));
        }

        result.push('\n');
    }

    Ok(result)
}

/// Export training history to a CSV file
///
/// # Arguments
///
/// * `history` - Map of metric names to values
/// * `filepath` - Path to save the CSV file
///
/// # Returns
///
/// * `Result<()>` - Result of the operation
pub fn export_history_to_csv<F: std::fmt::Display>(
    history: &HashMap<String, Vec<F>>,
    filepath: impl AsRef<Path>,
) -> Result<()> {
    let mut file = File::create(filepath)
        .map_err(|e| NeuralError::IOError(format!("Failed to create CSV file: {}", e)))?;

    // Find the maximum array length
    let max_len = history.values().map(|v| v.len()).max().unwrap_or(0);

    // Write header
    let mut header = String::from("epoch");

    // Get sorted keys for consistent column order
    let mut keys: Vec<&String> = history.keys().collect();
    keys.sort();

    for key in keys.iter() {
        header.push_str(&format!(",{}", key));
    }
    header.push('\n');

    file.write_all(header.as_bytes())
        .map_err(|e| NeuralError::IOError(format!("Failed to write CSV header: {}", e)))?;

    // Write data rows
    for i in 0..max_len {
        let mut row = i.to_string();

        // Ensure columns match the header order using the same sorted keys
        for key in keys.iter() {
            row.push(',');
            if let Some(values) = history.get(*key) {
                if i < values.len() {
                    row.push_str(&format!("{}", values[i]));
                }
            }
        }

        row.push('\n');

        file.write_all(row.as_bytes())
            .map_err(|e| NeuralError::IOError(format!("Failed to write CSV row: {}", e)))?;
    }

    Ok(())
}

/// Simple utility to generate a learning rate schedule
pub enum LearningRateSchedule<F: num_traits::Float> {
    /// Constant learning rate
    Constant(F),
    /// Step decay learning rate
    StepDecay {
        /// Initial learning rate
        initial_lr: F,
        /// Decay factor
        decay_factor: F,
        /// Epochs per step
        step_size: usize,
    },
    /// Exponential decay learning rate
    ExponentialDecay {
        /// Initial learning rate
        initial_lr: F,
        /// Decay factor
        decay_factor: F,
    },
    /// Custom learning rate schedule function
    Custom(Box<dyn Fn(usize) -> F>),
}

impl<F: num_traits::Float> LearningRateSchedule<F> {
    /// Get the learning rate for a given epoch
    pub fn get_learning_rate(&self, epoch: usize) -> F {
        match self {
            Self::Constant(lr) => *lr,
            Self::StepDecay {
                initial_lr,
                decay_factor,
                step_size,
            } => {
                let num_steps = epoch / step_size;
                *initial_lr * (*decay_factor).powi(num_steps as i32)
            }
            Self::ExponentialDecay {
                initial_lr,
                decay_factor,
            } => *initial_lr * (*decay_factor).powi(epoch as i32),
            Self::Custom(f) => f(epoch),
        }
    }

    /// Generate the learning rate schedule for all epochs
    ///
    /// # Arguments
    ///
    /// * `num_epochs` - Number of epochs
    ///
    /// # Returns
    ///
    /// * `Array1<F>` - Learning rate for each epoch
    pub fn generate_schedule(&self, num_epochs: usize) -> Array1<F> {
        Array1::from_shape_fn(num_epochs, |i| self.get_learning_rate(i))
    }
}

/// Analyze training history to find potential issues
///
/// # Arguments
///
/// * `history` - Map of metric names to values
///
/// # Returns
///
/// * `Vec<String>` - List of potential issues and suggestions
pub fn analyze_training_history<F: num_traits::Float + std::fmt::Display>(
    history: &HashMap<String, Vec<F>>,
) -> Vec<String> {
    let mut issues = Vec::new();

    // Check if we have training and validation loss
    if let (Some(train_loss), Some(val_loss)) = (history.get("train_loss"), history.get("val_loss"))
    {
        if train_loss.len() < 2 || val_loss.len() < 2 {
            return vec!["Not enough epochs to analyze training history.".to_string()];
        }

        // Check for overfitting
        let last_train = train_loss.last().unwrap();
        let last_val = val_loss.last().unwrap();

        if last_val.to_f64().unwrap() > last_train.to_f64().unwrap() * 1.1 {
            issues.push("Potential overfitting: validation loss is significantly higher than training loss.".to_string());
            issues.push("  - Try adding regularization (L1, L2, dropout)".to_string());
            issues.push("  - Consider data augmentation".to_string());
            issues.push("  - Try reducing model complexity".to_string());
        }

        // Check for underfitting
        let last_train_float = last_train.to_f64().unwrap();
        if last_train_float > 0.1 {
            issues.push("Potential underfitting: training loss is still high.".to_string());
            issues.push("  - Try increasing model complexity".to_string());
            issues.push("  - Train for more epochs".to_string());
            issues.push("  - Try different optimization algorithms or learning rates".to_string());
        }

        // Check for unstable training
        let mut fluctuations = 0;
        for i in 1..train_loss.len() {
            if train_loss[i] > train_loss[i - 1] {
                fluctuations += 1;
            }
        }

        let fluctuation_rate = fluctuations as f64 / (train_loss.len() as f64 - 1.0);
        if fluctuation_rate > 0.3 {
            issues.push("Unstable training: loss values fluctuate frequently.".to_string());
            issues.push("  - Try reducing learning rate".to_string());
            issues.push(
                "  - Use a different optimizer (Adam usually helps stabilize training)".to_string(),
            );
            issues.push("  - Try gradient clipping".to_string());
        }

        // Check for plateauing
        if train_loss.len() >= 4 {
            // Ensure we have enough data points for this analysis
            let first_half_improvement = train_loss[train_loss.len() / 2].to_f64().unwrap()
                - train_loss[0].to_f64().unwrap();
            let second_half_improvement = train_loss.last().unwrap().to_f64().unwrap()
                - train_loss[train_loss.len() / 2].to_f64().unwrap();

            if second_half_improvement.abs() < first_half_improvement.abs() * 0.2 {
                issues.push("Training plateau: little improvement in later epochs.".to_string());
                issues.push("  - Try learning rate scheduling".to_string());
                issues.push("  - Use early stopping to avoid wasting computation".to_string());
                issues.push("  - Consider a different optimizer or model architecture".to_string());
            }
        }

        // Check for divergent validation loss
        let mut val_increasing_count = 0;
        for i in 1..val_loss.len().min(5) {
            // Look at the last 5 epochs or less
            if val_loss[val_loss.len() - i] > val_loss[val_loss.len() - i - 1] {
                val_increasing_count += 1;
            }
        }

        if val_increasing_count >= 3 && val_loss.len() >= 5 {
            issues.push(
                "Validation loss is increasing in recent epochs, indicating overfitting."
                    .to_string(),
            );
            issues.push("  - Consider stopping training now to prevent overfitting".to_string());
            issues.push("  - Increase regularization strength".to_string());
            issues.push("  - Reduce model complexity".to_string());
        }
    }

    // Check accuracy trends if available
    if let Some(accuracy) = history.get("accuracy") {
        if accuracy.len() >= 3 {
            let last_accuracy = accuracy.last().unwrap().to_f64().unwrap();

            // Check if accuracy is high
            if last_accuracy > 0.95 {
                issues.push("Model has achieved very high accuracy (>95%).".to_string());
                issues.push(
                    "  - Consider stopping training or validating on more challenging data"
                        .to_string(),
                );
            }

            // Check for accuracy plateaus
            if accuracy.len() >= 5 {
                let recent_change = (accuracy.last().unwrap().to_f64().unwrap()
                    - accuracy[accuracy.len() - 5].to_f64().unwrap())
                .abs();

                if recent_change < 0.01 {
                    issues.push(
                        "Accuracy has plateaued with minimal improvement in recent epochs."
                            .to_string(),
                    );
                    issues.push("  - Try adjusting learning rate".to_string());
                    issues.push("  - Consider stopping training to avoid overfitting".to_string());
                }
            }
        }
    }

    if issues.is_empty() {
        issues.push("No significant issues detected in the training process.".to_string());
    }

    issues
}
