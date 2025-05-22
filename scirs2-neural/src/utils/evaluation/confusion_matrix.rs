//! Confusion matrix for classification problems

use crate::error::{NeuralError, Result};
use crate::utils::colors::{
    colored_metric_cell, colorize, colorize_and_style, gradient_color, heatmap_cell,
    heatmap_color_legend, stylize, Color, ColorOptions, Style,
};
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// Confusion matrix for classification problems
#[derive(Debug, Clone)]
pub struct ConfusionMatrix<F: Float + Debug + Display> {
    /// The raw confusion matrix data
    pub matrix: Array2<F>,
    /// Class labels (optional)
    pub labels: Option<Vec<String>>,
    /// Number of classes
    pub num_classes: usize,
}

impl<F: Float + Debug + Display> ConfusionMatrix<F> {
    /// Create a new confusion matrix from predictions and true labels
    ///
    /// # Arguments
    ///
    /// * `y_true` - True class labels as integers
    /// * `y_pred` - Predicted class labels as integers
    /// * `num_classes` - Number of classes (if None, determined from data)
    /// * `labels` - Optional class labels as strings
    ///
    /// # Returns
    ///
    /// * `Result<ConfusionMatrix<F>>` - The confusion matrix
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_neural::utils::evaluation::ConfusionMatrix;
    /// use ndarray::Array1;
    ///
    /// let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 0]);
    /// let y_pred = Array1::from_vec(vec![0, 1, 1, 0, 1, 2, 0]);
    /// let cm = ConfusionMatrix::<f32>::new(&y_true.view(), &y_pred.view(), None, None).unwrap();
    /// ```
    pub fn new(
        y_true: &ArrayView1<usize>,
        y_pred: &ArrayView1<usize>,
        num_classes: Option<usize>,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        if y_true.len() != y_pred.len() {
            return Err(NeuralError::ValidationError(
                "Predictions and true labels must have the same length".to_string(),
            ));
        }

        // Determine number of classes
        let n_classes = num_classes.unwrap_or_else(|| {
            let max_true = y_true.iter().max().copied().unwrap_or(0);
            let max_pred = y_pred.iter().max().copied().unwrap_or(0);
            std::cmp::max(max_true, max_pred) + 1
        });

        // Initialize confusion matrix with zeros
        let mut matrix = Array2::zeros((n_classes, n_classes));

        // Fill confusion matrix
        for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
            if *true_label < n_classes && *pred_label < n_classes {
                matrix[[*true_label, *pred_label]] = matrix[[*true_label, *pred_label]] + F::one();
            } else {
                return Err(NeuralError::ValidationError(format!(
                    "Class index out of bounds: true={}, pred={}, n_classes={}",
                    true_label, pred_label, n_classes
                )));
            }
        }

        // Validate labels if provided
        let validated_labels = if let Some(label_vec) = labels {
            if label_vec.len() != n_classes {
                return Err(NeuralError::ValidationError(format!(
                    "Number of labels ({}) does not match number of classes ({})",
                    label_vec.len(),
                    n_classes
                )));
            }
            Some(label_vec)
        } else {
            None
        };

        Ok(ConfusionMatrix {
            matrix,
            labels: validated_labels,
            num_classes: n_classes,
        })
    }

    /// Create a confusion matrix from raw matrix data
    ///
    /// # Arguments
    ///
    /// * `matrix` - Raw confusion matrix data
    /// * `labels` - Optional class labels
    ///
    /// # Returns
    ///
    /// * `Result<ConfusionMatrix<F>>` - The confusion matrix
    pub fn from_matrix(matrix: Array2<F>, labels: Option<Vec<String>>) -> Result<Self> {
        let shape = matrix.shape();
        if shape[0] != shape[1] {
            return Err(NeuralError::ValidationError(
                "Confusion matrix must be square".to_string(),
            ));
        }

        let n_classes = shape[0];

        // Validate labels if provided
        let validated_labels = if let Some(label_vec) = labels {
            if label_vec.len() != n_classes {
                return Err(NeuralError::ValidationError(format!(
                    "Number of labels ({}) does not match matrix size ({})",
                    label_vec.len(),
                    n_classes
                )));
            }
            Some(label_vec)
        } else {
            None
        };

        Ok(ConfusionMatrix {
            matrix,
            labels: validated_labels,
            num_classes: n_classes,
        })
    }

    /// Get the normalized confusion matrix (rows sum to 1)
    pub fn normalized(&self) -> Array2<F> {
        let mut norm_matrix = self.matrix.clone();

        // Normalize rows to sum to 1
        for row in 0..self.num_classes {
            let row_sum = self.matrix.row(row).sum();
            if row_sum > F::zero() {
                for col in 0..self.num_classes {
                    norm_matrix[[row, col]] = self.matrix[[row, col]] / row_sum;
                }
            }
        }

        norm_matrix
    }

    /// Calculate accuracy from the confusion matrix
    pub fn accuracy(&self) -> F {
        let total: F = self.matrix.sum();
        if total > F::zero() {
            let diagonal_sum: F = (0..self.num_classes)
                .map(|i| self.matrix[[i, i]])
                .fold(F::zero(), |acc, x| acc + x);
            diagonal_sum / total
        } else {
            F::zero()
        }
    }

    /// Calculate precision for each class
    pub fn precision(&self) -> Array1<F> {
        let mut precision = Array1::zeros(self.num_classes);

        for i in 0..self.num_classes {
            let col_sum = self.matrix.column(i).sum();
            if col_sum > F::zero() {
                precision[i] = self.matrix[[i, i]] / col_sum;
            }
        }

        precision
    }

    /// Calculate recall for each class
    pub fn recall(&self) -> Array1<F> {
        let mut recall = Array1::zeros(self.num_classes);

        for i in 0..self.num_classes {
            let row_sum = self.matrix.row(i).sum();
            if row_sum > F::zero() {
                recall[i] = self.matrix[[i, i]] / row_sum;
            }
        }

        recall
    }

    /// Calculate F1 score for each class
    pub fn f1_score(&self) -> Array1<F> {
        let precision = self.precision();
        let recall = self.recall();
        let mut f1 = Array1::zeros(self.num_classes);

        for i in 0..self.num_classes {
            let denom = precision[i] + recall[i];
            if denom > F::zero() {
                f1[i] = F::from(2.0).unwrap() * precision[i] * recall[i] / denom;
            }
        }

        f1
    }

    /// Calculate macro-averaged F1 score
    pub fn macro_f1(&self) -> F {
        let f1 = self.f1_score();
        let sum = f1.sum();
        sum / F::from(self.num_classes).unwrap()
    }

    /// Get class-wise metrics as a HashMap
    pub fn class_metrics(&self) -> HashMap<String, Vec<F>> {
        let precision = self.precision();
        let recall = self.recall();
        let f1 = self.f1_score();

        let mut metrics = HashMap::new();
        metrics.insert("precision".to_string(), precision.to_vec());
        metrics.insert("recall".to_string(), recall.to_vec());
        metrics.insert("f1".to_string(), f1.to_vec());

        metrics
    }

    /// Convert the confusion matrix to an ASCII representation
    pub fn to_ascii(&self, title: Option<&str>, normalized: bool) -> String {
        self.to_ascii_with_options(title, normalized, &ColorOptions::default())
    }

    /// Convert the confusion matrix to an ASCII representation with color options
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        normalized: bool,
        color_options: &ColorOptions,
    ) -> String {
        let matrix = if normalized {
            self.normalized()
        } else {
            self.matrix.clone()
        };

        let mut result = String::new();

        // Add title if provided
        if let Some(title_text) = title {
            if color_options.enabled {
                result.push_str(&stylize(title_text, Style::Bold));
            } else {
                result.push_str(title_text);
            }
            result.push_str("\n\n");
        }

        // Get class labels
        let labels: Vec<String> = match &self.labels {
            Some(label_vec) => label_vec.clone(),
            None => (0..self.num_classes).map(|i| i.to_string()).collect(),
        };

        // Determine column widths
        let label_width = labels.iter().map(|l| l.len()).max().unwrap_or(2).max(5);

        let cell_width = if normalized {
            6 // Width for normalized values (0.XX)
        } else {
            matrix
                .iter()
                .map(|&v| format!("{:.0}", v).len())
                .max()
                .unwrap_or(2)
                .max(5)
        };

        // Header row with class labels
        if color_options.enabled {
            result.push_str(&format!(
                "{:<width$} |",
                stylize("Pred→", Style::Bold),
                width = label_width + 8
            ));
        } else {
            result.push_str(&format!("{:<width$} |", "Pred→", width = label_width));
        }

        for label in &labels {
            if color_options.enabled {
                let styled_label = stylize(label, Style::Bold);
                result.push_str(&format!(
                    " {:<width$} |",
                    styled_label,
                    width = cell_width + 8
                ));
            } else {
                result.push_str(&format!(" {:<width$} |", label, width = cell_width));
            }
        }

        if color_options.enabled {
            result.push_str(&format!(" {}\n", stylize("Recall", Style::Bold)));
        } else {
            result.push_str(" Recall\n");
        }

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push_str(&"-".repeat(8));
        result.push('\n');

        // Data rows
        let recall = self.recall();
        for i in 0..self.num_classes {
            // Row label
            if color_options.enabled {
                result.push_str(&format!(
                    "{:<width$} |",
                    stylize(&labels[i], Style::Bold),
                    width = label_width + 8
                ));
            } else {
                result.push_str(&format!("{:<width$} |", labels[i], width = label_width));
            }

            for j in 0..self.num_classes {
                let value = matrix[[i, j]];
                let formatted = if normalized {
                    format!("{:.3}", value)
                } else {
                    format!("{:.0}", value)
                };

                // Color cells based on value (if enabled)
                if i == j {
                    // Diagonal cells (true positives)
                    if color_options.enabled {
                        // Get normalized value for coloring
                        let norm_value = if normalized {
                            value.to_f64().unwrap_or(0.0)
                        } else {
                            // For non-normalized matrices, normalize by row sum
                            let row_sum = matrix.row(i).sum().to_f64().unwrap_or(1.0);
                            if row_sum > 0.0 {
                                value.to_f64().unwrap_or(0.0) / row_sum
                            } else {
                                0.0
                            }
                        };

                        // Use gradient colors based on value
                        if let Some(color) = gradient_color(norm_value, color_options) {
                            // Apply both bold style and color
                            let colored_value = colorize(stylize(&formatted, Style::Bold), color);
                            result.push_str(&format!(
                                " {:<width$} |",
                                colored_value,
                                width = cell_width + 9
                            ));
                        } else {
                            // Just use bold if no color
                            result.push_str(&format!(
                                " {:<width$} |",
                                stylize(&formatted, Style::Bold),
                                width = cell_width + 8
                            ));
                        }
                    } else {
                        // No color, just bold
                        result.push_str(&format!(
                            " \x1b[1m{:<width$}\x1b[0m |",
                            formatted,
                            width = cell_width
                        ));
                    }
                } else if color_options.enabled && normalized {
                    // Color non-diagonal cells by intensity
                    let norm_value = value.to_f64().unwrap_or(0.0);
                    if norm_value > 0.1 {
                        result.push_str(&format!(
                            " {:<width$} |",
                            colorize(&formatted, Color::BrightRed),
                            width = cell_width + 9
                        ));
                    } else {
                        result.push_str(&format!(" {:<width$} |", formatted, width = cell_width));
                    }
                } else {
                    // No color for non-diagonal cells
                    result.push_str(&format!(" {:<width$} |", formatted, width = cell_width));
                }
            }

            // Add recall for this class with coloring
            if color_options.enabled {
                let recall_val = recall[i].to_f64().unwrap_or(0.0);
                let colored_recall =
                    colored_metric_cell(format!("{:.3}", recall[i]), recall_val, color_options);
                result.push_str(&format!(" {}\n", colored_recall));
            } else {
                result.push_str(&format!(" {:.3}\n", recall[i]));
            }
        }

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push_str(&"-".repeat(8));
        result.push('\n');

        // Precision row
        let precision = self.precision();
        if color_options.enabled {
            result.push_str(&format!(
                "{:<width$} |",
                stylize("Precision", Style::Bold),
                width = label_width + 8
            ));
        } else {
            result.push_str(&format!("{:<width$} |", "Precision", width = label_width));
        }

        for j in 0..self.num_classes {
            if color_options.enabled {
                let precision_val = precision[j].to_f64().unwrap_or(0.0);
                let colored_precision = colored_metric_cell(
                    format!("{:.3}", precision[j]),
                    precision_val,
                    color_options,
                );
                result.push_str(&format!(" {} |", colored_precision));
            } else {
                result.push_str(&format!(" {:.3} |", precision[j]));
            }
        }

        // Add overall accuracy
        let accuracy = self.accuracy();
        if color_options.enabled {
            let accuracy_val = accuracy.to_f64().unwrap_or(0.0);
            let colored_accuracy =
                colored_metric_cell(format!("{:.3}", accuracy), accuracy_val, color_options);
            result.push_str(&format!(" {}\n", colored_accuracy));
        } else {
            result.push_str(&format!(" {:.3}\n", accuracy));
        }

        // Add F1 scores
        let f1 = self.f1_score();
        if color_options.enabled {
            result.push_str(&format!(
                "{:<width$} |",
                stylize("F1-score", Style::Bold),
                width = label_width + 8
            ));
        } else {
            result.push_str(&format!("{:<width$} |", "F1-score", width = label_width));
        }

        for j in 0..self.num_classes {
            if color_options.enabled {
                let f1_val = f1[j].to_f64().unwrap_or(0.0);
                let colored_f1 =
                    colored_metric_cell(format!("{:.3}", f1[j]), f1_val, color_options);
                result.push_str(&format!(" {} |", colored_f1));
            } else {
                result.push_str(&format!(" {:.3} |", f1[j]));
            }
        }

        // Add macro F1
        let macro_f1 = self.macro_f1();
        if color_options.enabled {
            let macro_f1_val = macro_f1.to_f64().unwrap_or(0.0);
            let colored_macro_f1 =
                colored_metric_cell(format!("{:.3}", macro_f1), macro_f1_val, color_options);
            result.push_str(&format!(" {}\n", colored_macro_f1));
        } else {
            result.push_str(&format!(" {:.3}\n", macro_f1));
        }

        result
    }

    /// Convert the confusion matrix to a heatmap visualization
    ///
    /// This creates a colorful heatmap visualization of the confusion matrix
    /// where cell colors represent the intensity of values using a detailed color gradient.
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the heatmap
    /// * `normalized` - Whether to normalize the matrix (row values sum to 1)
    ///
    /// # Returns
    ///
    /// * `String` - ASCII heatmap representation
    pub fn to_heatmap(&self, title: Option<&str>, normalized: bool) -> String {
        self.to_heatmap_with_options(title, normalized, &ColorOptions::default())
    }

    /// Create a confusion matrix heatmap that focuses on misclassification patterns
    ///
    /// This visualization is specialized to highlight where the model makes mistakes,
    /// with emphasis on the off-diagonal elements to help identify error patterns.
    /// The key features of this visualization are:
    ///
    /// - Diagonal elements (correct classifications) are de-emphasized with dim styling
    /// - Off-diagonal elements (errors) are highlighted with a color gradient
    /// - Colors are normalized relative to the maximum off-diagonal value
    /// - A specialized legend explains error intensity levels
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the heatmap
    ///
    /// # Returns
    ///
    /// * `String` - ASCII error pattern heatmap
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_neural::utils::evaluation::ConfusionMatrix;
    /// use ndarray::Array1;
    ///
    /// // Create some example data
    /// let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0]);
    /// let y_pred = Array1::from_vec(vec![0, 1, 1, 0, 1, 2, 1, 1, 0, 0]);
    ///
    /// let class_labels = vec!["Class A".to_string(), "Class B".to_string(), "Class C".to_string()];
    /// let cm = ConfusionMatrix::<f32>::new(&y_true.view(), &y_pred.view(), None, Some(class_labels)).unwrap();
    ///
    /// // Generate the error pattern heatmap
    /// let error_viz = cm.error_heatmap(Some("Misclassification Analysis"));
    /// println!("{}", error_viz);
    /// ```
    pub fn error_heatmap(&self, title: Option<&str>) -> String {
        // Always use normalized values for error heatmap
        let normalized = true;

        // Create custom color options for error visualization
        let color_options = ColorOptions {
            use_bright: true,
            ..Default::default()
        };

        // Create an error-focused visualization of the confusion matrix
        let matrix = if normalized {
            self.normalized()
        } else {
            self.matrix.clone()
        };

        let mut result = String::new();

        // Add title if provided
        if let Some(title_text) = title {
            if color_options.enabled {
                result.push_str(&stylize(title_text, Style::Bold));
            } else {
                result.push_str(title_text);
            }
            result.push('\n');
            result.push('\n');
        }

        // Get class labels
        let labels: Vec<String> = match &self.labels {
            Some(label_vec) => label_vec.clone(),
            None => (0..self.num_classes).map(|i| i.to_string()).collect(),
        };

        // Determine column widths
        let label_width = labels.iter().map(|l| l.len()).max().unwrap_or(2).max(5);

        let cell_width = if normalized {
            6 // Width for normalized values (0.XX)
        } else {
            matrix
                .iter()
                .map(|&v| format!("{:.0}", v).len())
                .max()
                .unwrap_or(2)
                .max(5)
        };

        // Find the maximum off-diagonal value for normalization
        let mut max_off_diag = F::zero();
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                if i != j && matrix[[i, j]] > max_off_diag {
                    max_off_diag = matrix[[i, j]];
                }
            }
        }

        // If there are no off-diagonal values (perfect classifier), use max value
        if max_off_diag == F::zero() {
            max_off_diag = matrix
                .iter()
                .fold(F::zero(), |acc, &v| if v > acc { v } else { acc });
        }

        // Header with error emphasis title
        if color_options.enabled {
            result.push_str(&format!(
                "{:<width$} |",
                stylize("True↓ / Pred→", Style::Bold),
                width = label_width + 8
            ));
        } else {
            result.push_str(&format!(
                "{:<width$} |",
                "True↓ / Pred→",
                width = label_width
            ));
        }

        for label in &labels {
            if color_options.enabled {
                let styled_label = stylize(label, Style::Bold);
                result.push_str(&format!(
                    " {:<width$} |",
                    styled_label,
                    width = cell_width + 8
                ));
            } else {
                result.push_str(&format!(" {:<width$} |", label, width = cell_width));
            }
        }
        result.push('\n');

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push('\n');

        // Data rows - using error-focused coloring
        for i in 0..self.num_classes {
            // Row label
            if color_options.enabled {
                result.push_str(&format!(
                    "{:<width$} |",
                    stylize(&labels[i], Style::Bold),
                    width = label_width + 8
                ));
            } else {
                result.push_str(&format!("{:<width$} |", labels[i], width = label_width));
            }

            // Format and color each cell - but emphasize errors (off-diagonal)
            for j in 0..self.num_classes {
                let value = matrix[[i, j]];
                let formatted = if normalized {
                    format!("{:.3}", value)
                } else {
                    format!("{:.0}", value)
                };

                if i == j {
                    // Diagonal elements (correct classifications) - de-emphasize
                    if color_options.enabled {
                        // Dim style for diagonal elements
                        result.push_str(&format!(
                            " {:<width$} |",
                            colorize_and_style(&formatted, None, None, Some(Style::Dim)),
                            width = cell_width + 8
                        ));
                    } else {
                        result.push_str(&format!(" {:<width$} |", formatted, width = cell_width));
                    }
                } else {
                    // Off-diagonal elements (errors) - emphasize with color gradient
                    let norm_value = if max_off_diag > F::zero() {
                        (value / max_off_diag).to_f64().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    if color_options.enabled && norm_value > 0.0 {
                        // Use a specialized color scheme for errors
                        let error_color = if norm_value < 0.25 {
                            Color::BrightBlue
                        } else if norm_value < 0.5 {
                            Color::BrightCyan
                        } else if norm_value < 0.75 {
                            Color::BrightRed
                        } else {
                            Color::BrightMagenta
                        };

                        // Bold style for the most significant errors
                        if norm_value > 0.5 {
                            result.push_str(&format!(
                                " {:<width$} |",
                                colorize_and_style(
                                    &formatted,
                                    Some(error_color),
                                    None,
                                    Some(Style::Bold)
                                ),
                                width = cell_width + 8
                            ));
                        } else {
                            result.push_str(&format!(
                                " {:<width$} |",
                                colorize(&formatted, error_color),
                                width = cell_width + 8
                            ));
                        }
                    } else {
                        result.push_str(&format!(" {:<width$} |", formatted, width = cell_width));
                    }
                }
            }

            result.push('\n');
        }

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push('\n');

        // Add specialized error heatmap legend
        if color_options.enabled {
            result.push('\n');
            let mut legend = String::from("Error Pattern Legend: ");

            // Custom legend showing error intensity levels
            let error_levels = [
                (Color::BrightBlue, "Low Error (0-25%)"),
                (Color::BrightCyan, "Moderate Error (25-50%)"),
                (Color::BrightRed, "High Error (50-75%)"),
                (Color::BrightMagenta, "Critical Error (75-100%)"),
            ];

            for (i, (color, label)) in error_levels.iter().enumerate() {
                if i > 0 {
                    legend.push(' ');
                }
                legend.push_str(&format!("{} {}", colorize("■", *color), label));
            }

            // Add note about diagonal elements
            legend.push_str(&format!(
                " {} Correct Classification",
                colorize_and_style("■", None, None, Some(Style::Dim))
            ));

            result.push_str(&legend);
            result.push('\n');
        }

        result
    }

    /// Convert the confusion matrix to a heatmap visualization with customizable options
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the heatmap
    /// * `normalized` - Whether to normalize the matrix (row values sum to 1)
    /// * `color_options` - Color options for visualization
    ///
    /// # Returns
    ///
    /// * `String` - ASCII heatmap representation with colors
    pub fn to_heatmap_with_options(
        &self,
        title: Option<&str>,
        normalized: bool,
        color_options: &ColorOptions,
    ) -> String {
        let matrix = if normalized {
            self.normalized()
        } else {
            self.matrix.clone()
        };

        let mut result = String::new();

        // Add title if provided
        if let Some(title_text) = title {
            if color_options.enabled {
                result.push_str(&stylize(title_text, Style::Bold));
            } else {
                result.push_str(title_text);
            }
            result.push_str("\n\n");
        }

        // Get class labels
        let labels: Vec<String> = match &self.labels {
            Some(label_vec) => label_vec.clone(),
            None => (0..self.num_classes).map(|i| i.to_string()).collect(),
        };

        // Determine column widths
        let label_width = labels.iter().map(|l| l.len()).max().unwrap_or(2).max(5);

        let cell_width = if normalized {
            6 // Width for normalized values (0.XX)
        } else {
            matrix
                .iter()
                .map(|&v| format!("{:.0}", v).len())
                .max()
                .unwrap_or(2)
                .max(5)
        };

        // Find the maximum value for normalization
        let max_value = if normalized {
            F::one() // Normalized values are already between 0 and 1
        } else {
            matrix
                .iter()
                .fold(F::zero(), |acc, &v| if v > acc { v } else { acc })
        };

        // Header row with class labels
        if color_options.enabled {
            result.push_str(&format!(
                "{:<width$} |",
                stylize("True↓", Style::Bold),
                width = label_width + 8
            ));
        } else {
            result.push_str(&format!("{:<width$} |", "True↓", width = label_width));
        }

        for label in &labels {
            if color_options.enabled {
                let styled_label = stylize(label, Style::Bold);
                result.push_str(&format!(
                    " {:<width$} |",
                    styled_label,
                    width = cell_width + 8
                ));
            } else {
                result.push_str(&format!(" {:<width$} |", label, width = cell_width));
            }
        }
        result.push('\n');

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push('\n');

        // Data rows - using heatmap coloring
        for i in 0..self.num_classes {
            // Row label
            if color_options.enabled {
                result.push_str(&format!(
                    "{:<width$} |",
                    stylize(&labels[i], Style::Bold),
                    width = label_width + 8
                ));
            } else {
                result.push_str(&format!("{:<width$} |", labels[i], width = label_width));
            }

            // Format and color each cell
            for j in 0..self.num_classes {
                let value = matrix[[i, j]];
                let formatted = if normalized {
                    format!("{:.3}", value)
                } else {
                    format!("{:.0}", value)
                };

                // Get normalized value for coloring
                let norm_value = if normalized {
                    value.to_f64().unwrap_or(0.0)
                } else if max_value > F::zero() {
                    (value / max_value).to_f64().unwrap_or(0.0)
                } else {
                    0.0
                };

                // Apply heatmap coloring
                if color_options.enabled {
                    let heatmap_value = heatmap_cell(&formatted, norm_value, color_options);

                    // Add extra space for ANSI color codes
                    result.push_str(&format!(
                        " {:<width$} |",
                        heatmap_value,
                        width = cell_width + 9
                    ));
                } else {
                    result.push_str(&format!(" {:<width$} |", formatted, width = cell_width));
                }
            }

            result.push('\n');
        }

        // Separator
        result.push_str(&"-".repeat(label_width + 2));
        for _ in 0..self.num_classes {
            result.push_str(&format!("{}-", "-".repeat(cell_width + 2)));
        }
        result.push('\n');

        // Add heatmap legend
        if color_options.enabled {
            if let Some(legend) = heatmap_color_legend(color_options) {
                result.push('\n');
                result.push_str(&legend);
                result.push('\n');
            }
        }

        result
    }
}
