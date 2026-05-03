//! Terminal visualization utilities for training metrics
//!
//! Provides ASCII/Unicode sparkline plots for rendering metric histories
//! directly in a terminal during or after neural network training.

use std::collections::HashMap;

use crate::error::{NeuralError, Result};

/// Options controlling the dimensions of the generated plot.
#[derive(Debug, Clone)]
pub struct PlotOptions {
    /// Width of the plot in columns (one column per data point, clamped to `data.len()`).
    pub width: usize,
    /// Height of the plot in rows (number of character rows in the vertical bar).
    pub height: usize,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            width: 80,
            height: 20,
        }
    }
}

/// Unicode block elements used to build vertical bars.
///
/// Index 0 → empty cell (space), index 8 → full-block `█`.
const BLOCK_CHARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render a single data series as a multi-row ASCII/Unicode bar chart.
///
/// Each column represents one data point; the column height is drawn using
/// Unicode block characters (`▁` … `█`).  The function returns a
/// `height`-line string (lines joined by `'\n'`) where the top row is the
/// highest value and the bottom row is the lowest.
///
/// # Arguments
/// * `data`   – Time-series values to render.
/// * `width`  – Number of columns in the output (capped at `data.len()`).
/// * `height` – Number of rows in the output.
///
/// # Returns
/// A multi-line `String`.  Returns an empty string when `data` is empty or
/// `height` is zero.
pub fn ascii_plot(data: &[f64], width: usize, height: usize) -> String {
    if data.is_empty() || height == 0 {
        return String::new();
    }

    // Determine the number of columns actually rendered.
    let n_cols = width.min(data.len()).max(1);

    // Downsample by averaging buckets when data is longer than width.
    let sampled: Vec<f64> = (0..n_cols)
        .map(|col| {
            let start = col * data.len() / n_cols;
            let end = ((col + 1) * data.len() / n_cols)
                .max(start + 1)
                .min(data.len());
            let slice = &data[start..end];
            slice.iter().copied().sum::<f64>() / (slice.len() as f64)
        })
        .collect();

    let min_val = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    // Map each sampled value to a bar height in [0, height * 8] sub-steps
    // (8 sub-steps per row for the 8 block-character levels).
    let total_steps = height * 8;
    let bar_steps: Vec<usize> = sampled
        .iter()
        .map(|&v| {
            if range < 1e-12 {
                total_steps / 2
            } else {
                (((v - min_val) / range) * total_steps as f64).round() as usize
            }
        })
        .collect();

    // Build `height` rows from top (row 0) to bottom (row height-1).
    // Row `r` represents the band covering sub-steps [height*8 - (r+1)*8, height*8 - r*8).
    let mut rows: Vec<String> = Vec::with_capacity(height);
    for row in 0..height {
        // The band this row covers (in sub-steps from the bottom).
        let band_bottom = (height - 1 - row) * 8;
        let band_top = band_bottom + 8; // exclusive

        let row_str: String = bar_steps
            .iter()
            .map(|&steps| {
                if steps >= band_top {
                    // Column fills this row completely.
                    BLOCK_CHARS[8]
                } else if steps > band_bottom {
                    // Column partially fills this row.
                    let partial = steps - band_bottom; // 1..7
                    BLOCK_CHARS[partial]
                } else {
                    // Column does not reach this row.
                    ' '
                }
            })
            .collect();

        rows.push(row_str);
    }

    rows.join("\n")
}

/// Render multiple named metric series as a labelled ASCII plot.
///
/// Each metric in `history` is rendered as its own labelled sparkline
/// using [`ascii_plot`].  The result is a single `String` that may span
/// many lines.
///
/// # Arguments
/// * `history` – Map from metric name to ordered sequence of `f64` values.
/// * `title`   – Optional title printed at the top.
/// * `options` – Optional plot dimensions; falls back to [`PlotOptions::default`].
///
/// # Errors
/// Returns `NeuralError::ComputationError` when the conversion from the
/// internal float type to `f64` fails for any value.
pub fn plot_metrics(
    history: &HashMap<String, Vec<f64>>,
    title: Option<&str>,
    options: Option<PlotOptions>,
) -> Result<String> {
    let opts = options.unwrap_or_default();
    let mut output = String::new();

    if let Some(t) = title {
        let border = "─".repeat(opts.width.min(80));
        output.push_str(&format!("┌{border}┐\n"));
        output.push_str(&format!("│ {t:<width$}│\n", width = opts.width.min(80) - 1));
        output.push_str(&format!("└{border}┘\n"));
    }

    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = history.keys().collect();
    keys.sort();

    for key in keys {
        let values = &history[key];
        if values.is_empty() {
            continue;
        }
        output.push_str(&format!("  {key}:\n"));
        let plot = ascii_plot(values, opts.width, opts.height.clamp(4, 10));
        for line in plot.lines() {
            output.push_str(&format!("  │{line}│\n"));
        }
        output.push('\n');
    }

    Ok(output)
}

/// Convert a `Vec<F: Float>` to `Vec<f64>`, silently dropping values that
/// cannot be converted (returns `NeuralError` if *all* conversions fail).
pub fn float_vec_to_f64<F: scirs2_core::numeric::Float>(values: &[F]) -> Result<Vec<f64>> {
    let converted: Vec<f64> = values.iter().filter_map(|v| v.to_f64()).collect();
    if converted.is_empty() && !values.is_empty() {
        return Err(NeuralError::ComputationError(
            "Could not convert any metric values to f64 for visualization".to_string(),
        ));
    }
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_ascii_plot_sinusoid() {
        let data: Vec<f64> = (0..32).map(|i| (i as f64 * PI / 16.0).sin()).collect();

        let plot = ascii_plot(&data, 32, 8);

        // Must produce exactly 8 lines.
        assert_eq!(plot.lines().count(), 8, "expected 8 rows");

        // Must contain at least one Unicode block character.
        let has_block = plot
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'));
        assert!(has_block, "plot should contain block characters");

        // Must not be empty.
        assert!(!plot.is_empty());
    }

    #[test]
    fn test_ascii_plot_empty_returns_empty() {
        assert_eq!(ascii_plot(&[], 10, 5), "");
    }

    #[test]
    fn test_ascii_plot_zero_height_returns_empty() {
        assert_eq!(ascii_plot(&[1.0, 2.0, 3.0], 10, 0), "");
    }

    #[test]
    fn test_ascii_plot_constant_series() {
        // Constant data: no range → middle fill.
        let data = vec![5.0_f64; 20];
        let plot = ascii_plot(&data, 20, 4);
        assert_eq!(plot.lines().count(), 4);
    }

    #[test]
    fn test_plot_metrics_produces_labelled_output() {
        let mut history: HashMap<String, Vec<f64>> = HashMap::new();
        history.insert("train_loss".to_string(), vec![1.0, 0.8, 0.6, 0.5, 0.4]);
        history.insert("val_loss".to_string(), vec![1.1, 0.9, 0.7, 0.6, 0.5]);

        let result = plot_metrics(
            &history,
            Some("Test Metrics"),
            Some(PlotOptions {
                width: 40,
                height: 6,
            }),
        );
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert!(output.contains("train_loss"));
        assert!(output.contains("val_loss"));
        assert!(output.contains("Test Metrics"));
    }
}
