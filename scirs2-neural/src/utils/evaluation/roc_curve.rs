//! ROC curve for binary classification problems
//!
//! This module provides tools for computing, analyzing, and visualizing
//! Receiver Operating Characteristic (ROC) curves for binary classifiers.

use crate::error::{NeuralError, Result};
use crate::utils::colors::{
    colored_metric_cell, colorize, stylize, Color, ColorOptions, Style, RESET,
};
use crate::utils::evaluation::helpers::draw_line_with_coords;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::fmt::{Debug, Display};

/// ROC curve data structure for binary classification evaluation
///
/// This struct represents a Receiver Operating Characteristic (ROC) curve,
/// which plots the True Positive Rate (TPR) against the False Positive Rate (FPR)
/// at various classification thresholds. It also calculates the Area Under the
/// Curve (AUC), a common metric for binary classification performance.
pub struct ROCCurve<F: Float + Debug + Display> {
    /// False positive rates at different thresholds
    pub fpr: Array1<F>,
    /// True positive rates at different thresholds
    pub tpr: Array1<F>,
    /// Classification thresholds
    pub thresholds: Array1<F>,
    /// Area Under the ROC Curve (AUC)
    pub auc: F,
}

impl<F: Float + Debug + Display> ROCCurve<F> {
    /// Compute ROC curve and AUC from binary classification scores
    ///
    /// # Arguments
    ///
    /// * `y_true` - True binary labels (0 or 1)
    /// * `y_score` - Predicted probabilities or decision function
    ///
    /// # Returns
    ///
    /// * `Result<ROCCurve<F>>` - ROC curve data
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, ArrayView1};
    /// use scirs2_neural::utils::evaluation::ROCCurve;
    ///
    /// // Create some example data
    /// let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0]);
    /// let y_score = Array1::from_vec(vec![0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.3]);
    ///
    /// // Compute ROC curve
    /// let roc = ROCCurve::<f64>::new(&y_true.view(), &y_score.view()).unwrap();
    ///
    /// // AUC should be > 0.5 for a model better than random guessing
    /// assert!(roc.auc > 0.5);
    /// ```
    pub fn new(y_true: &ArrayView1<usize>, y_score: &ArrayView1<F>) -> Result<Self> {
        if y_true.len() != y_score.len() {
            return Err(NeuralError::ValidationError(
                "Labels and scores must have the same length".to_string(),
            ));
        }

        // Check if y_true contains only binary values (0 or 1)
        for &label in y_true.iter() {
            if label != 0 && label != 1 {
                return Err(NeuralError::ValidationError(
                    "Labels must be binary (0 or 1)".to_string(),
                ));
            }
        }

        // Sort scores and corresponding labels in descending order
        let mut score_label_pairs: Vec<(F, usize)> = y_score
            .iter()
            .zip(y_true.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Count positives and negatives
        let n_pos = y_true.iter().filter(|&&label| label == 1).count();
        let n_neg = y_true.len() - n_pos;

        if n_pos == 0 || n_neg == 0 {
            return Err(NeuralError::ValidationError(
                "Both positive and negative samples are required".to_string(),
            ));
        }

        // Initialize arrays for ROC curve
        let n_thresholds = score_label_pairs.len() + 1;
        let mut fpr = Array1::zeros(n_thresholds);
        let mut tpr = Array1::zeros(n_thresholds);
        let mut thresholds = Array1::zeros(n_thresholds);

        // Set the first point (0,0) and last threshold to infinity
        thresholds[0] = F::infinity();

        // Compute ROC curve
        let mut tp = 0;
        let mut fp = 0;

        for i in 0..score_label_pairs.len() {
            let (score, label) = score_label_pairs[i];

            // Update counts
            if label == 1 {
                tp += 1;
            } else {
                fp += 1;
            }

            // Set threshold for this point
            thresholds[i + 1] = score;

            // Compute rates
            tpr[i + 1] = F::from(tp).unwrap() / F::from(n_pos).unwrap();
            fpr[i + 1] = F::from(fp).unwrap() / F::from(n_neg).unwrap();
        }

        // Compute AUC using trapezoidal rule
        let mut auc = F::zero();
        for i in 0..fpr.len() - 1 {
            auc = auc + (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) * F::from(0.5).unwrap();
        }

        Ok(ROCCurve {
            fpr,
            tpr,
            thresholds,
            auc,
        })
    }

    /// Create an ASCII line plot of the ROC curve
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the plot
    /// * `width` - Width of the plot
    /// * `height` - Height of the plot
    ///
    /// # Returns
    ///
    /// * `String` - ASCII line plot
    pub fn to_ascii(&self, title: Option<&str>, width: usize, height: usize) -> String {
        self.to_ascii_with_options(title, width, height, &ColorOptions::default())
    }

    /// Create an ASCII line plot of the ROC curve with color options
    ///
    /// This method provides a customizable visualization of the ROC curve
    /// with controls for colors and styling.
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the plot
    /// * `width` - Width of the plot
    /// * `height` - Height of the plot
    /// * `color_options` - Color options for visualization
    ///
    /// # Returns
    ///
    /// * `String` - ASCII line plot with colors
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, ArrayView1};
    /// use scirs2_neural::utils::evaluation::ROCCurve;
    /// use scirs2_neural::utils::colors::ColorOptions;
    ///
    /// // Create test data
    /// let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0]);
    /// let y_score = Array1::from_vec(vec![0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.3]);
    ///
    /// // Compute ROC curve
    /// let roc = ROCCurve::<f64>::new(&y_true.view(), &y_score.view()).unwrap();
    ///
    /// // Create ROC curve visualization
    /// let options = ColorOptions::default();
    /// let plot = roc.to_ascii_with_options(Some("Model Performance"), 50, 20, &options);
    ///
    /// // Visualization will show the curve with the AUC value
    /// assert!(plot.contains("AUC ="));
    /// ```
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        width: usize,
        height: usize,
        color_options: &ColorOptions,
    ) -> String {
        // Pre-allocate result string with estimated capacity
        let mut result = String::with_capacity(width * height * 2);

        // Add title and AUC with coloring if enabled
        if let Some(title_text) = title {
            if color_options.enabled {
                let styled_title = stylize(title_text, Style::Bold);
                let auc_value = self.auc.to_f64().unwrap_or(0.0);
                let colored_auc =
                    colored_metric_cell(format!("{:.3}", self.auc), auc_value, color_options);
                result.push_str(&format!("{} (AUC = {})\n\n", styled_title, colored_auc));
            } else {
                result.push_str(&format!("{} (AUC = {:.3})\n\n", title_text, self.auc));
            }
        } else if color_options.enabled {
            let styled_title = stylize("ROC Curve", Style::Bold);
            let auc_value = self.auc.to_f64().unwrap_or(0.0);
            let colored_auc =
                colored_metric_cell(format!("{:.3}", self.auc), auc_value, color_options);
            result.push_str(&format!("{} (AUC = {})\n\n", styled_title, colored_auc));
        } else {
            result.push_str(&format!("ROC Curve (AUC = {:.3})\n\n", self.auc));
        }

        // Create a 2D grid for the plot
        let mut grid = vec![vec![' '; width]; height];

        // Draw the diagonal (random classifier line)
        for i in 0..std::cmp::min(width, height) {
            let x = i;
            let y = height - 1 - i * (height - 1) / (width - 1);
            if x < width && y < height {
                grid[y][x] = '.';
            }
        }

        // Convert ROC curve points to line segments
        let mut prev_x = 0;
        let mut prev_y = height - 1; // Start at (0,0) in ROC space

        for i in 1..self.fpr.len() {
            let x = (self.fpr[i].to_f64().unwrap() * (width - 1) as f64).round() as usize;
            let y =
                height - 1 - (self.tpr[i].to_f64().unwrap() * (height - 1) as f64).round() as usize;

            // Draw line segments between points
            if x != prev_x || y != prev_y {
                for (line_x, line_y) in
                    draw_line_with_coords(prev_x, prev_y, x, y, Some(width), Some(height))
                {
                    grid[line_y][line_x] = '●';
                }

                prev_x = x;
                prev_y = y;
            }
        }

        // Draw the grid
        for (y, row) in grid.iter().enumerate() {
            // Y-axis labels with styling
            if y == height - 1 {
                if color_options.enabled {
                    result.push_str(&format!("{}0.0{} |", Color::BrightCyan.fg_code(), RESET));
                } else {
                    result.push_str("0.0 |");
                }
            } else if y == 0 {
                if color_options.enabled {
                    result.push_str(&format!("{}1.0{} |", Color::BrightCyan.fg_code(), RESET));
                } else {
                    result.push_str("1.0 |");
                }
            } else if y == height / 2 {
                if color_options.enabled {
                    result.push_str(&format!("{}0.5{} |", Color::BrightCyan.fg_code(), RESET));
                } else {
                    result.push_str("0.5 |");
                }
            } else {
                result.push_str("    |");
            }

            // Grid content with coloring
            for char in row.iter().take(width) {
                if color_options.enabled {
                    match char {
                        '●' => {
                            // Color the ROC curve points
                            result.push_str(&colorize("●", Color::BrightGreen));
                        }
                        '.' => {
                            // Color the diagonal line
                            result.push_str(&colorize(".", Color::BrightBlack));
                        }
                        _ => result.push(*char),
                    }
                } else {
                    result.push(*char);
                }
            }

            result.push('\n');
        }

        // X-axis
        result.push_str("    +");
        result.push_str(&"-".repeat(width));
        result.push('\n');

        // X-axis labels with styling
        result.push_str("     ");
        if color_options.enabled {
            result.push_str(&colorize("0.0", Color::BrightCyan));
            result.push_str(&" ".repeat(width - 6));
            result.push_str(&colorize("1.0", Color::BrightCyan));
        } else {
            result.push_str("0.0");
            result.push_str(&" ".repeat(width - 6));
            result.push_str("1.0");
        }
        result.push('\n');

        // Axis labels with styling
        if color_options.enabled {
            result.push_str(&format!(
                "     {}\n",
                stylize("False Positive Rate (FPR)", Style::Bold)
            ));
        } else {
            result.push_str("     False Positive Rate (FPR)\n");
        }

        // Add legend if colors are enabled
        if color_options.enabled {
            result.push('\n');
            result.push_str(&format!(
                "     {} ROC curve     {} Random classifier\n",
                colorize("●", Color::BrightGreen),
                colorize(".", Color::BrightBlack)
            ));
        }

        result
    }
}
