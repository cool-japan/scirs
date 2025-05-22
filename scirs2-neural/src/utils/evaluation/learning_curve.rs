//! Learning curve for model performance analysis
//!
//! This module provides the LearningCurve data structure for visualizing model
//! performance across different training set sizes, comparing training and
//! validation metrics.

use crate::error::{NeuralError, Result};
use crate::utils::colors::{colorize, stylize, Color, ColorOptions, Style};
use crate::utils::evaluation::helpers::draw_line_with_coords;
use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

// Removed problematic type alias - use trait bounds directly in implementations

/// Learning curve data structure for visualizing model performance
///
/// This structure represents learning curves that show how model performance
/// changes as the training set size increases, comparing training and validation
/// metrics to help diagnose overfitting, underfitting, and other training issues.
pub struct LearningCurve<F: Float + Debug + Display> {
    /// Training set sizes used for evaluation
    pub train_sizes: Array1<usize>,
    /// Training scores for each size and fold (rows=sizes, cols=folds)
    pub train_scores: Array2<F>,
    /// Validation scores for each size and fold (rows=sizes, cols=folds)
    pub val_scores: Array2<F>,
    /// Mean training scores across folds
    pub train_mean: Array1<F>,
    /// Standard deviation of training scores
    pub train_std: Array1<F>,
    /// Mean validation scores across folds
    pub val_mean: Array1<F>,
    /// Standard deviation of validation scores
    pub val_std: Array1<F>,
}

impl<F: Float + Debug + Display + FromPrimitive> LearningCurve<F> {
    /// Create a new learning curve from training and validation scores
    ///
    /// # Arguments
    ///
    /// * `train_sizes` - Array of training set sizes
    /// * `train_scores` - 2D array of training scores (rows=sizes, cols=cv folds)
    /// * `val_scores` - 2D array of validation scores (rows=sizes, cols=cv folds)
    ///
    /// # Returns
    ///
    /// * `Result<LearningCurve<F>>` - Learning curve data
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use scirs2_neural::utils::evaluation::LearningCurve;
    ///
    /// // Create sample data
    /// let train_sizes = Array1::from_vec(vec![100, 200, 300, 400, 500]);
    /// let train_scores = Array2::from_shape_vec((5, 3), vec![
    ///     0.6, 0.62, 0.58,    // 100 samples, 3 folds
    ///     0.7, 0.72, 0.68,    // 200 samples, 3 folds
    ///     0.8, 0.78, 0.79,    // 300 samples, 3 folds
    ///     0.85, 0.83, 0.84,   // 400 samples, 3 folds
    ///     0.87, 0.88, 0.86,   // 500 samples, 3 folds
    /// ]).unwrap();
    /// let val_scores = Array2::from_shape_vec((5, 3), vec![
    ///     0.55, 0.53, 0.54,   // 100 samples, 3 folds
    ///     0.65, 0.63, 0.64,   // 200 samples, 3 folds
    ///     0.75, 0.73, 0.74,   // 300 samples, 3 folds
    ///     0.76, 0.74, 0.75,   // 400 samples, 3 folds
    ///     0.77, 0.76, 0.76,   // 500 samples, 3 folds
    /// ]).unwrap();
    ///
    /// // Create learning curve
    /// let curve = LearningCurve::<f64>::new(train_sizes, train_scores, val_scores).unwrap();
    /// ```
    pub fn new(
        train_sizes: Array1<usize>,
        train_scores: Array2<F>,
        val_scores: Array2<F>,
    ) -> Result<Self> {
        let n_sizes = train_sizes.len();

        if train_scores.shape()[0] != n_sizes || val_scores.shape()[0] != n_sizes {
            return Err(NeuralError::ValidationError(
                "Number of scores must match number of training sizes".to_string(),
            ));
        }

        if train_scores.shape()[1] != val_scores.shape()[1] {
            return Err(NeuralError::ValidationError(
                "Training and validation scores must have the same number of CV folds".to_string(),
            ));
        }

        // Compute means and standard deviations
        let train_mean = train_scores.mean_axis(Axis(1)).unwrap();
        let val_mean = val_scores.mean_axis(Axis(1)).unwrap();

        // Compute standard deviations using helper function
        let train_std = compute_std(&train_scores, &train_mean, n_sizes);
        let val_std = compute_std(&val_scores, &val_mean, n_sizes);

        Ok(LearningCurve {
            train_sizes,
            train_scores,
            val_scores,
            train_mean,
            train_std,
            val_mean,
            val_std,
        })
    }

    /// Create an ASCII line plot of the learning curve
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the plot
    /// * `width` - Width of the plot
    /// * `height` - Height of the plot
    /// * `metric_name` - Name of the metric (e.g., "Accuracy")
    ///
    /// # Returns
    ///
    /// * `String` - ASCII line plot
    pub fn to_ascii(
        &self,
        title: Option<&str>,
        width: usize,
        height: usize,
        metric_name: &str,
    ) -> String {
        self.to_ascii_with_options(title, width, height, metric_name, &ColorOptions::default())
    }

    /// Create an ASCII line plot of the learning curve with customizable colors
    ///
    /// This method allows fine-grained control over the color scheme using the
    /// provided ColorOptions parameter.
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the plot
    /// * `width` - Width of the plot
    /// * `height` - Height of the plot
    /// * `metric_name` - Name of the metric (e.g., "Accuracy")
    /// * `color_options` - Color options for visualization
    ///
    /// # Returns
    ///
    /// * `String` - ASCII line plot with colors
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        width: usize,
        height: usize,
        metric_name: &str,
        color_options: &ColorOptions,
    ) -> String {
        // Pre-allocate result string with estimated capacity
        let mut result = String::with_capacity(width * height * 2);

        // Add title with styling if provided
        if let Some(title_text) = title {
            if color_options.enabled {
                result.push_str(&format!("{}\n\n", stylize(title_text, Style::Bold)));
            } else {
                result.push_str(&format!("{}\n\n", title_text));
            }
        } else if color_options.enabled {
            let title = format!("Learning Curve ({})", stylize(metric_name, Style::Bold));
            result.push_str(&format!("{}\n\n", stylize(title, Style::Bold)));
        } else {
            result.push_str(&format!("Learning Curve ({})\n\n", metric_name));
        }

        // Find min and max values for y-axis scaling
        let min_score = self
            .val_mean
            .iter()
            .fold(F::infinity(), |acc, &v| if v < acc { v } else { acc });

        let max_score =
            self.train_mean
                .iter()
                .fold(F::neg_infinity(), |acc, &v| if v > acc { v } else { acc });

        // Add a small margin to the y-range
        let y_margin = F::from(0.1).unwrap() * (max_score - min_score);
        let y_min = min_score - y_margin;
        let y_max = max_score + y_margin;

        // Create a 2D grid for the plot
        let mut grid = vec![vec![' '; width]; height];
        let mut grid_markers = vec![vec![(false, false); width]; height]; // Track which points are training vs. validation

        // Function to map a value to a y-coordinate
        let y_coord = |value: F| -> usize {
            let norm = (value - y_min) / (y_max - y_min);
            let y = height - 1 - (norm.to_f64().unwrap() * (height - 1) as f64).round() as usize;
            std::cmp::min(y, height - 1)
        };

        // Function to map a training size to an x-coordinate
        let x_coord = |size_idx: usize| -> usize {
            ((size_idx as f64) / ((self.train_sizes.len() - 1) as f64) * (width - 1) as f64).round()
                as usize
        };

        // Draw training curve and mark as training points
        for i in 0..self.train_sizes.len() - 1 {
            let x1 = x_coord(i);
            let y1 = y_coord(self.train_mean[i]);
            let x2 = x_coord(i + 1);
            let y2 = y_coord(self.train_mean[i + 1]);

            // Draw a line between points and mark as training points
            for (x, y) in draw_line_with_coords(x1, y1, x2, y2, Some(width), Some(height)) {
                grid[y][x] = '●';
                grid_markers[y][x].0 = true; // Mark as training point
            }
        }

        // Draw validation curve and mark as validation points
        for i in 0..self.train_sizes.len() - 1 {
            let x1 = x_coord(i);
            let y1 = y_coord(self.val_mean[i]);
            let x2 = x_coord(i + 1);
            let y2 = y_coord(self.val_mean[i + 1]);

            // Draw a line between points and mark as validation points
            for (x, y) in draw_line_with_coords(x1, y1, x2, y2, Some(width), Some(height)) {
                grid[y][x] = '○';
                grid_markers[y][x].1 = true; // Mark as validation point
            }
        }

        // Draw the grid
        for y in 0..height {
            // Y-axis labels with styling
            if y == 0 {
                if color_options.enabled {
                    let value = format!("{:.2}", y_max);
                    result.push_str(&format!("{} |", colorize(value, Color::BrightCyan)));
                } else {
                    result.push_str(&format!("{:.2} |", y_max));
                }
            } else if y == height - 1 {
                if color_options.enabled {
                    let value = format!("{:.2}", y_min);
                    result.push_str(&format!("{} |", colorize(value, Color::BrightCyan)));
                } else {
                    result.push_str(&format!("{:.2} |", y_min));
                }
            } else if y == height / 2 {
                let mid = y_min + (y_max - y_min) * F::from(0.5).unwrap();
                if color_options.enabled {
                    let value = format!("{:.2}", mid);
                    result.push_str(&format!("{} |", colorize(value, Color::BrightCyan)));
                } else {
                    result.push_str(&format!("{:.2} |", mid));
                }
            } else {
                result.push_str("      |");
            }

            // Grid content with coloring
            for x in 0..width {
                let char = grid[y][x];
                let (is_train, is_val) = grid_markers[y][x];

                if color_options.enabled {
                    if is_train {
                        // Training point
                        result.push_str(&colorize("●", Color::BrightBlue));
                    } else if is_val {
                        // Validation point
                        result.push_str(&colorize("○", Color::BrightGreen));
                    } else {
                        result.push(char);
                    }
                } else {
                    result.push(char);
                }
            }

            result.push('\n');
        }

        // X-axis
        result.push_str("      +");
        result.push_str(&"-".repeat(width));
        result.push('\n');

        // X-axis labels with styling
        result.push_str("       ");

        // Put a few size labels along the x-axis
        let n_labels = std::cmp::min(5, self.train_sizes.len());
        for i in 0..n_labels {
            let idx = i * (self.train_sizes.len() - 1) / (n_labels - 1);
            let size = self.train_sizes[idx];
            let label = format!("{}", size);
            let x = x_coord(idx);

            // Position the label with styling
            if i == 0 {
                if color_options.enabled {
                    result.push_str(&colorize(label, Color::BrightCyan));
                } else {
                    result.push_str(&label);
                }
            } else {
                let prev_end = result.len();
                let spaces = x.saturating_sub(prev_end - 7);
                result.push_str(&" ".repeat(spaces));
                if color_options.enabled {
                    result.push_str(&colorize(label, Color::BrightCyan));
                } else {
                    result.push_str(&label);
                }
            }
        }

        result.push('\n');

        // X-axis title with styling
        if color_options.enabled {
            result.push_str(&format!(
                "       {}\n\n",
                stylize("Training Set Size", Style::Bold)
            ));
        } else {
            result.push_str("       Training Set Size\n\n");
        }

        // Add legend with colors
        if color_options.enabled {
            result.push_str(&format!(
                "       {} Training score   {} Validation score\n",
                colorize("●", Color::BrightBlue),
                colorize("○", Color::BrightGreen)
            ));
        } else {
            result.push_str("       ● Training score   ○ Validation score\n");
        }

        result
    }
}

/// Helper function to compute standard deviation for scores
fn compute_std<F: Float + Debug + Display + FromPrimitive>(
    scores: &Array2<F>,
    mean: &Array1<F>,
    n_sizes: usize,
) -> Array1<F> {
    let mut std_arr = Array1::zeros(n_sizes);
    let n_folds = scores.shape()[1];

    for i in 0..n_sizes {
        let mut sum_sq_diff = F::zero();
        for j in 0..n_folds {
            let diff = scores[[i, j]] - mean[i];
            sum_sq_diff = sum_sq_diff + diff * diff;
        }
        std_arr[i] = (sum_sq_diff / F::from(n_folds).unwrap()).sqrt();
    }

    std_arr
}
