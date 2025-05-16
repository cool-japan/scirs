use crate::error::{NeuralError, Result};
use crate::utils::colors::{
    colored_metric_cell, colorize, colorize_and_style, gradient_color, heatmap_cell,
    heatmap_color_legend, stylize, Color, ColorOptions, Style, RESET,
};
use ndarray::{Array1, Array2, ArrayView1, Axis};
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

/// Feature importance visualization for machine learning models
pub struct FeatureImportance<F: Float + Debug + Display> {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Importance scores
    pub importance: Array1<F>,
}

impl<F: Float + Debug + Display> FeatureImportance<F> {
    /// Create a new feature importance visualization
    ///
    /// # Arguments
    ///
    /// * `feature_names` - Names of features
    /// * `importance` - Importance scores
    ///
    /// # Returns
    ///
    /// * `Result<FeatureImportance<F>>` - The feature importance visualization
    pub fn new(feature_names: Vec<String>, importance: Array1<F>) -> Result<Self> {
        if feature_names.len() != importance.len() {
            return Err(NeuralError::ValidationError(
                "Number of feature names must match number of importance scores".to_string(),
            ));
        }

        Ok(FeatureImportance {
            feature_names,
            importance,
        })
    }

    /// Get the top-k most important features
    ///
    /// # Arguments
    ///
    /// * `k` - Number of top features to return
    ///
    /// # Returns
    ///
    /// * `(Vec<String>, Array1<F>)` - Feature names and importance scores
    pub fn top_k(&self, k: usize) -> (Vec<String>, Array1<F>) {
        let k = std::cmp::min(k, self.feature_names.len());
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();

        // Sort indices by importance (descending)
        indices.sort_by(|&a, &b| {
            self.importance[b]
                .partial_cmp(&self.importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top-k features
        let top_indices = indices[..k].to_vec();
        let top_names = top_indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();
        let top_importance = Array1::from_iter(top_indices.iter().map(|&i| self.importance[i]));

        (top_names, top_importance)
    }

    /// Create an ASCII bar chart of feature importance
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the chart
    /// * `width` - Width of the bar chart
    /// * `k` - Number of top features to include (None for all)
    ///
    /// # Returns
    ///
    /// * `String` - ASCII bar chart
    pub fn to_ascii(&self, title: Option<&str>, width: usize, k: Option<usize>) -> String {
        self.to_ascii_with_options(title, width, k, &ColorOptions::default())
    }

    /// Create an ASCII bar chart of feature importance with color options
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the chart
    /// * `width` - Width of the bar chart
    /// * `k` - Number of top features to include (None for all)
    /// * `color_options` - Color options for visualization
    ///
    /// # Returns
    ///
    /// * `String` - ASCII bar chart
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        width: usize,
        k: Option<usize>,
        color_options: &ColorOptions,
    ) -> String {
        let (features, importance) = if let Some(top_k) = k {
            self.top_k(top_k)
        } else {
            (self.feature_names.clone(), self.importance.clone())
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

        // Determine maximum importance for scaling
        let max_importance = importance
            .iter()
            .fold(F::zero(), |acc, &v| if v > acc { v } else { acc });

        // Get maximum feature name length for alignment
        let max_name_len = features
            .iter()
            .map(|name| name.len())
            .max()
            .unwrap_or(10)
            .max(10);

        // Determine available width for bars
        let bar_area_width = width.saturating_sub(max_name_len + 10);

        // Create a sorted index
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.sort_by(|&a, &b| {
            importance[b]
                .partial_cmp(&importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Draw the bar chart
        for &idx in &indices {
            let feature_name = &features[idx];
            let imp = importance[idx];

            // Scale the importance to fit in the available width
            let bar_length = if max_importance > F::zero() {
                let ratio = (imp / max_importance).to_f64().unwrap_or(0.0);
                (ratio * bar_area_width as f64).round() as usize
            } else {
                0
            };

            // Format feature name with styling
            let formatted_name = if color_options.enabled {
                stylize(feature_name, Style::Bold).to_string()
            } else {
                feature_name.clone()
            };

            // Format importance value with coloring
            let formatted_imp = if color_options.enabled {
                let normalized_imp = (imp / max_importance).to_f64().unwrap_or(0.0);
                colored_metric_cell(format!("{:<5.3}", imp), normalized_imp, color_options)
            } else {
                format!("{:<5.3}", imp)
            };

            // Format the bar with coloring
            let bar = if color_options.enabled {
                let normalized_imp = (imp / max_importance).to_f64().unwrap_or(0.0);
                if let Some(color) = gradient_color(normalized_imp, color_options) {
                    colorize("█".repeat(bar_length), color)
                } else {
                    "█".repeat(bar_length)
                }
            } else {
                "█".repeat(bar_length)
            };

            // Format the line
            if color_options.enabled {
                // When using colors, we need to adjust width calculations for ANSI escapes
                result.push_str(&format!(
                    "{:<width$} | {} |{}|\n",
                    formatted_name,
                    formatted_imp,
                    bar,
                    width = max_name_len + 9 // Add extra space for ANSI codes
                ));
            } else {
                result.push_str(&format!(
                    "{:<width$} | {} |{}|\n",
                    formatted_name,
                    formatted_imp,
                    bar,
                    width = max_name_len
                ));
            }
        }

        result
    }
}

/// ROC curve data structure
pub struct ROCCurve<F: Float + Debug + Display> {
    /// False positive rates
    pub fpr: Array1<F>,
    /// True positive rates
    pub tpr: Array1<F>,
    /// Thresholds
    pub thresholds: Array1<F>,
    /// Area under the curve
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
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        width: usize,
        height: usize,
        color_options: &ColorOptions,
    ) -> String {
        let mut result = String::new();

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
            grid[y][x] = '.';
        }

        // Map ROC curve points to grid coordinates
        for i in 0..self.fpr.len() {
            let x = (self.fpr[i].to_f64().unwrap() * (width - 1) as f64).round() as usize;
            let y =
                height - 1 - (self.tpr[i].to_f64().unwrap() * (height - 1) as f64).round() as usize;

            if x < width && y < height {
                grid[y][x] = '●';
            }
        }

        // Draw the grid
        for (y, row) in grid.iter().enumerate() {
            // Y-axis labels with styling
            if y == height - 1 {
                if color_options.enabled {
                    result.push_str(&format!(
                        "{}0.0{} |",
                        if color_options.enabled {
                            Color::BrightCyan.fg_code()
                        } else {
                            ""
                        },
                        RESET
                    ));
                } else {
                    result.push_str("0.0 |");
                }
            } else if y == 0 {
                if color_options.enabled {
                    result.push_str(&format!(
                        "{}1.0{} |",
                        if color_options.enabled {
                            Color::BrightCyan.fg_code()
                        } else {
                            ""
                        },
                        RESET
                    ));
                } else {
                    result.push_str("1.0 |");
                }
            } else if y == height / 2 {
                if color_options.enabled {
                    result.push_str(&format!(
                        "{}0.5{} |",
                        if color_options.enabled {
                            Color::BrightCyan.fg_code()
                        } else {
                            ""
                        },
                        RESET
                    ));
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

/// Learning curve data structure
pub struct LearningCurve<F: Float + Debug + Display> {
    /// Training set sizes
    pub train_sizes: Array1<usize>,
    /// Training scores for each size
    pub train_scores: Array2<F>,
    /// Validation scores for each size
    pub val_scores: Array2<F>,
    /// Mean training scores
    pub train_mean: Array1<F>,
    /// Standard deviation of training scores
    pub train_std: Array1<F>,
    /// Mean validation scores
    pub val_mean: Array1<F>,
    /// Standard deviation of validation scores
    pub val_std: Array1<F>,
}

impl<F: Float + Debug + Display + num_traits::FromPrimitive> LearningCurve<F> {
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

        // Compute standard deviations
        let mut train_std = Array1::zeros(n_sizes);
        let mut val_std = Array1::zeros(n_sizes);

        for i in 0..n_sizes {
            let n_folds = train_scores.shape()[1];

            // Training scores std
            let mut sum_sq_diff = F::zero();
            for j in 0..n_folds {
                let diff = train_scores[[i, j]] - train_mean[i];
                sum_sq_diff = sum_sq_diff + diff * diff;
            }
            train_std[i] = (sum_sq_diff / F::from(n_folds).unwrap()).sqrt();

            // Validation scores std
            let mut sum_sq_diff = F::zero();
            for j in 0..n_folds {
                let diff = val_scores[[i, j]] - val_mean[i];
                sum_sq_diff = sum_sq_diff + diff * diff;
            }
            val_std[i] = (sum_sq_diff / F::from(n_folds).unwrap()).sqrt();
        }

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

    /// Create an ASCII line plot of the learning curve with color options
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
        let mut result = String::new();

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
            for (x, y) in draw_line_with_coords(x1, y1, x2, y2) {
                if x < width && y < height {
                    grid[y][x] = '●';
                    grid_markers[y][x].0 = true; // Mark as training point
                }
            }
        }

        // Draw validation curve and mark as validation points
        for i in 0..self.train_sizes.len() - 1 {
            let x1 = x_coord(i);
            let y1 = y_coord(self.val_mean[i]);
            let x2 = x_coord(i + 1);
            let y2 = y_coord(self.val_mean[i + 1]);

            // Draw a line between points and mark as validation points
            for (x, y) in draw_line_with_coords(x1, y1, x2, y2) {
                if x < width && y < height {
                    grid[y][x] = '○';
                    grid_markers[y][x].1 = true; // Mark as validation point
                }
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

// Removed unused draw_line function since it's been replaced by draw_line_with_coords

/// Helper function to draw a line between two points and return coordinates
/// Returns a vector of (x, y) coordinates on the line
fn draw_line_with_coords(x1: usize, y1: usize, x2: usize, y2: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();

    // Simple Bresenham-like algorithm
    let dx = (x2 as isize - x1 as isize).abs();
    let dy = (y2 as isize - y1 as isize).abs();

    let sx = if x1 < x2 { 1isize } else { -1isize };
    let sy = if y1 < y2 { 1isize } else { -1isize };

    let mut err = dx - dy;
    let mut x = x1 as isize;
    let mut y = y1 as isize;

    while x != x2 as isize || y != y2 as isize {
        if x >= 0 && y >= 0 {
            coords.push((x as usize, y as usize));
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }

    // Add the endpoint
    coords.push((x2, y2));

    coords
}
