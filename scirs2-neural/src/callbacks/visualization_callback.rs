use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use crate::utils::visualization::{ascii_plot, PlotOptions};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Callback for visualizing training metrics in real-time
pub struct VisualizationCallback<F: Float + Debug + ScalarOperand> {
    /// Frequency of visualization (in epochs)
    pub frequency: usize,
    /// Whether to show plots during training
    pub show_plots: bool,
    /// Optional path to save plots
    pub save_path: Option<PathBuf>,
    /// Tracked metrics for visualization
    pub tracked_metrics: Vec<String>,
    /// Plot options
    pub plot_options: PlotOptions,
    /// Current epoch history
    epoch_history: HashMap<String, Vec<F>>,
}
impl<F: Float + Debug + ScalarOperand> VisualizationCallback<F> {
    /// Create a new visualization callback
    ///
    /// # Arguments
    /// * `frequency` - How often to visualize metrics (in epochs)
    pub fn new(frequency: usize) -> Self {
        Self {
            frequency,
            show_plots: true,
            save_path: None,
            tracked_metrics: vec!["train_loss".to_string(), "val_loss".to_string()],
            plot_options: PlotOptions::default(),
            epoch_history: HashMap::new(),
        }
    }
    /// Set the save path for plots
    pub fn with_save_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.save_path = Some(path.into());
        self
    /// Set whether to show plots during training
    pub fn with_show_plots(mut self, showplots: bool) -> Self {
        self.show_plots = show_plots;
    /// Set tracked metrics
    pub fn with_tracked_metrics(mut self, metrics: Vec<String>) -> Self {
        self.tracked_metrics = metrics;
    /// Set plot options
    pub fn with_plot_options(mut self, options: PlotOptions) -> Self {
        self.plot_options = options;
impl<F: Float + Debug + ScalarOperand + std::fmt::Display> Callback<F>
    for VisualizationCallback<F>
{
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        match timing {
            CallbackTiming::BeforeTraining => {
                // Initialize history with empty vectors for tracked metrics
                for metric in &self.tracked_metrics {
                    self.epoch_history.insert(metric.clone(), Vec::new());
                }
            }
            CallbackTiming::AfterEpoch => {
                // Update the history from the context
                if let Some(train_loss) = context.epoch_loss {
                    if let Some(values) = self.epoch_history.get_mut("train_loss") {
                        values.push(train_loss);
                    }
                if let Some(val_loss) = context.val_loss {
                    if let Some(values) = self.epoch_history.get_mut("val_loss") {
                        values.push(val_loss);
                // Add metrics if available
                if !context.metrics.is_empty() {
                    // Handle multiple metrics
                    let tracked_metrics_count = self.tracked_metrics.len();
                    // We have at least two elements reserved for train_loss and val_loss
                    let metric_offset = 2;
                    for (i, &metric_value) in context.metrics.iter().enumerate() {
                        let metric_name = if i + metric_offset < tracked_metrics_count {
                            // Use the predefined metric name if available
                            self.tracked_metrics[i + metric_offset].clone()
                        } else {
                            // Otherwise use a generic name
                            format!("metric_{}", i)
                        };
                        if let Some(values) = self.epoch_history.get_mut(&metric_name) {
                            values.push(metric_value);
                            self.epoch_history.insert(metric_name, vec![metric_value]);
                        }
                // Display visualization if frequency matches
                if context.epoch % self.frequency == 0
                    && self.show_plots
                    && !self.epoch_history.is_empty()
                {
                    if let Ok(plot) = ascii_plot(
                        &self.epoch_history,
                        Some("Training Metrics"),
                        Some(self.plot_options.clone()),
                    ) {
                        println!("\n{}", plot);
            CallbackTiming::AfterTraining => {
                // Display final visualization
                if self.show_plots && !self.epoch_history.is_empty() {
                        Some("Final Training Metrics"),
                // Save final visualization if save_path is provided
                if let Some(save_path) = &self.save_path {
                        if let Err(e) = std::fs::write(save_path, plot) {
                            eprintln!("Failed to save plot to {}: {}", save_path.display(), e);
                            println!("Plot saved to {}", save_path.display());
            _ => {}
        Ok(())
#[cfg(test)]
mod tests {
    use super::*;
    // No imports needed for this test
    #[test]
    fn test_visualization_callback_creation() {
        let callback = VisualizationCallback::<f32>::new(1);
        assert_eq!(callback.frequency, 1);
        assert!(callback.show_plots);
        assert!(callback.save_path.is_none());
        assert_eq!(
            callback.tracked_metrics,
            vec!["train_loss".to_string(), "val_loss".to_string()]
        );
        assert!(callback.epoch_history.is_empty());
        // Test with options
        let callback = VisualizationCallback::<f32>::new(2)
            .with_save_path("test.txt")
            .with_show_plots(false)
            .with_tracked_metrics(vec![
                "train_loss".to_string(),
                "val_loss".to_string(),
                "accuracy".to_string(),
            ]);
        assert_eq!(callback.frequency, 2);
        assert!(!callback.show_plots);
        assert!(callback.save_path.is_some());
            vec![
                "accuracy".to_string()
            ]
