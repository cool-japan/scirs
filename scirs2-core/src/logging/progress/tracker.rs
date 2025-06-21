//! Enhanced Progress Tracker
//!
//! Provides rich progress tracking with multiple visualization styles, statistical analysis,
//! and adaptive update rates.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::renderer::ProgressRenderer;
use super::statistics::ProgressStats;

/// Progress visualization style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgressStyle {
    /// Simple percentage text (e.g., "45%")
    Percentage,
    /// Basic ASCII progress bar
    Bar,
    /// Unicode block characters for smoother bar
    BlockBar,
    /// Spinner with percentage
    Spinner,
    /// Detailed bar with additional statistics
    DetailedBar,
}

/// Progress tracker configuration
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    /// The style of progress visualization
    pub style: ProgressStyle,
    /// Width of the progress bar in characters
    pub width: usize,
    /// Whether to show ETA
    pub show_eta: bool,
    /// Whether to show detailed statistics
    pub show_statistics: bool,
    /// Whether to show transfer rate (items/second)
    pub show_speed: bool,
    /// Whether to use adaptive update rate
    pub adaptive_rate: bool,
    /// Minimum update interval
    pub min_update_interval: Duration,
    /// Maximum update interval
    pub max_update_interval: Duration,
    /// The template for progress display
    pub template: Option<String>,
    /// Custom symbols for the progress bar
    pub symbols: Option<ProgressSymbols>,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            style: ProgressStyle::BlockBar,
            width: 40,
            show_eta: true,
            show_statistics: true,
            show_speed: true,
            adaptive_rate: true,
            min_update_interval: Duration::from_millis(100),
            max_update_interval: Duration::from_secs(1),
            template: None,
            symbols: None,
        }
    }
}

/// Custom symbols for progress visualization
#[derive(Debug, Clone)]
pub struct ProgressSymbols {
    /// Start of progress bar
    pub start: String,
    /// End of progress bar
    pub end: String,
    /// Filled section of progress bar
    pub fill: String,
    /// Empty section of progress bar
    pub empty: String,
    /// Spinner frames
    pub spinner: Vec<String>,
}

impl Default for ProgressSymbols {
    fn default() -> Self {
        Self {
            start: "[".to_string(),
            end: "]".to_string(),
            fill: "=".to_string(),
            empty: " ".to_string(),
            spinner: vec![
                "-".to_string(),
                "\\".to_string(),
                "|".to_string(),
                "/".to_string(),
            ],
        }
    }
}

impl ProgressSymbols {
    /// Create symbols for block-style progress bar
    pub fn blocks() -> Self {
        Self {
            start: "│".to_string(),
            end: "│".to_string(),
            fill: "█".to_string(),
            empty: " ".to_string(),
            spinner: vec![
                "⠋".to_string(),
                "⠙".to_string(),
                "⠹".to_string(),
                "⠸".to_string(),
                "⠼".to_string(),
                "⠴".to_string(),
                "⠦".to_string(),
                "⠧".to_string(),
                "⠇".to_string(),
                "⠏".to_string(),
            ],
        }
    }
}

/// Enhanced progress tracker
pub struct EnhancedProgressTracker {
    /// The progress description
    pub description: String,
    /// Progress configuration
    pub config: ProgressConfig,
    /// Progress statistics
    stats: Arc<Mutex<ProgressStats>>,
    /// Start time
    start_time: Instant,
    /// Is the progress tracking active?
    active: bool,
    /// Should the progress bar be hidden?
    hidden: bool,
    /// Renderer for the progress bar
    renderer: ProgressRenderer,
}

impl EnhancedProgressTracker {
    /// Create a new progress tracker
    pub fn new(description: &str, total: u64) -> Self {
        let config = ProgressConfig::default();
        let stats = Arc::new(Mutex::new(ProgressStats::new(total)));
        let renderer = ProgressRenderer::new();

        Self {
            description: description.to_string(),
            config,
            stats,
            start_time: Instant::now(),
            active: false,
            hidden: false,
            renderer,
        }
    }

    /// Configure the progress tracker
    pub fn with_config(mut self, config: ProgressConfig) -> Self {
        self.config = config;
        self
    }

    /// Use a specific progress style
    pub const fn with_style(mut self, style: ProgressStyle) -> Self {
        self.config.style = style;
        self
    }

    /// Set custom symbols
    pub fn with_symbols(mut self, symbols: ProgressSymbols) -> Self {
        self.config.symbols = Some(symbols);
        self
    }

    /// Show or hide ETA
    pub const fn with_eta(mut self, show: bool) -> Self {
        self.config.show_eta = show;
        self
    }

    /// Show or hide statistics
    pub const fn with_statistics(mut self, show: bool) -> Self {
        self.config.show_statistics = show;
        self
    }

    /// Start tracking progress
    pub fn start(&mut self) {
        self.active = true;
        self.start_time = Instant::now();

        // Initialize terminal if needed
        if !self.hidden {
            self.renderer.init();
            self.render();
        }
    }

    /// Update progress with current count
    pub fn update(&mut self, processed: u64) {
        if !self.active {
            return;
        }

        let now = Instant::now();

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.update(processed, now);
        }

        // Determine if we should render an update
        let should_render = if self.config.adaptive_rate {
            self.should_update_adaptive()
        } else {
            self.should_update_fixed()
        };

        if should_render && !self.hidden {
            self.render();
        }
    }

    /// Increment progress by a specified amount
    pub fn increment(&mut self, amount: u64) {
        if !self.active {
            return;
        }

        let processed = {
            let stats = self.stats.lock().unwrap();
            stats.processed + amount
        };

        self.update(processed);
    }

    /// Finish progress tracking
    pub fn finish(&mut self) {
        if !self.active {
            return;
        }

        self.active = false;

        // Set processed to total
        {
            let mut stats = self.stats.lock().unwrap();
            let total = stats.total;
            stats.processed = total;
            stats.percentage = 100.0;
            stats.eta = Duration::from_secs(0);
            stats.update(total, Instant::now());
        }

        // Final render
        if !self.hidden {
            self.render();
            self.renderer.finalize();
        }
    }

    /// Hide the progress bar (useful for non-interactive environments)
    pub fn hide(&mut self) {
        self.hidden = true;
    }

    /// Show the progress bar
    pub fn show(&mut self) {
        self.hidden = false;

        if self.active {
            self.render();
        }
    }

    /// Get the current progress statistics
    pub fn stats(&self) -> ProgressStats {
        self.stats.lock().unwrap().clone()
    }

    /// Determine if we should update based on fixed interval
    fn should_update_fixed(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        let elapsed = stats.last_update.elapsed();
        elapsed >= self.config.min_update_interval
    }

    /// Determine if we should update based on adaptive interval
    fn should_update_adaptive(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        let elapsed = stats.last_update.elapsed();

        // Always update if we've exceeded the maximum interval
        if elapsed >= self.config.max_update_interval {
            return true;
        }

        // Always update if we've exceeded the minimum interval and progress has changed significantly
        if elapsed >= self.config.min_update_interval {
            // Calculate how much progress has been made
            let progress_ratio = if stats.total > 0 {
                stats.processed as f64 / stats.total as f64
            } else {
                0.0
            };

            // Adaptive update logic:
            // - Update more frequently at the beginning and end
            // - Update less frequently in the middle
            let position_factor = 4.0 * progress_ratio * (1.0 - progress_ratio);
            let threshold = self.config.min_update_interval.as_secs_f64()
                + position_factor
                    * (self.config.max_update_interval.as_secs_f64()
                        - self.config.min_update_interval.as_secs_f64());

            elapsed.as_secs_f64() >= threshold
        } else {
            false
        }
    }

    /// Render the progress bar
    fn render(&mut self) {
        if self.hidden {
            return;
        }

        let stats = self.stats.lock().unwrap();
        let symbols = self.config.symbols.clone().unwrap_or_default();

        match self.config.style {
            ProgressStyle::Percentage => {
                self.renderer.render_percentage(&self.description, &stats);
            }
            ProgressStyle::Bar => {
                self.renderer.render_bar(
                    &self.description,
                    &stats,
                    self.config.width,
                    self.config.show_eta,
                    &symbols,
                );
            }
            ProgressStyle::BlockBar => {
                let block_symbols = ProgressSymbols::blocks();
                self.renderer.render_bar(
                    &self.description,
                    &stats,
                    self.config.width,
                    self.config.show_eta,
                    &block_symbols,
                );
            }
            ProgressStyle::Spinner => {
                self.renderer.render_spinner(
                    &self.description,
                    &stats,
                    &symbols,
                    self.config.show_eta,
                );
            }
            ProgressStyle::DetailedBar => {
                self.renderer.render_detailed_bar(
                    &self.description,
                    &stats,
                    self.config.width,
                    self.config.show_eta,
                    self.config.show_statistics,
                    self.config.show_speed,
                    &symbols,
                );
            }
        }
    }
}

/// Builder for creating progress visualizations
pub struct ProgressBuilder {
    description: String,
    total: u64,
    style: ProgressStyle,
    width: usize,
    show_eta: bool,
    show_statistics: bool,
    show_speed: bool,
    adaptive_rate: bool,
    min_update_interval: Duration,
    max_update_interval: Duration,
    template: Option<String>,
    symbols: Option<ProgressSymbols>,
    hidden: bool,
}

impl ProgressBuilder {
    /// Create a new progress builder
    pub fn new(description: &str, total: u64) -> Self {
        Self {
            description: description.to_string(),
            total,
            style: ProgressStyle::BlockBar,
            width: 40,
            show_eta: true,
            show_statistics: true,
            show_speed: true,
            adaptive_rate: true,
            min_update_interval: Duration::from_millis(100),
            max_update_interval: Duration::from_secs(1),
            template: None,
            symbols: None,
            hidden: false,
        }
    }

    /// Set the progress style
    pub const fn style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the progress bar width
    pub const fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Show or hide ETA
    pub const fn show_eta(mut self, show: bool) -> Self {
        self.show_eta = show;
        self
    }

    /// Show or hide statistics
    pub const fn show_statistics(mut self, show: bool) -> Self {
        self.show_statistics = show;
        self
    }

    /// Show or hide speed
    pub const fn show_speed(mut self, show: bool) -> Self {
        self.show_speed = show;
        self
    }

    /// Enable or disable adaptive update rate
    pub const fn adaptive_rate(mut self, enable: bool) -> Self {
        self.adaptive_rate = enable;
        self
    }

    /// Set the minimum update interval
    pub const fn min_update_interval(mut self, interval: Duration) -> Self {
        self.min_update_interval = interval;
        self
    }

    /// Set the maximum update interval
    pub const fn max_update_interval(mut self, interval: Duration) -> Self {
        self.max_update_interval = interval;
        self
    }

    /// Set a custom template
    pub fn template(mut self, template: &str) -> Self {
        self.template = Some(template.to_string());
        self
    }

    /// Set custom symbols
    pub fn symbols(mut self, symbols: ProgressSymbols) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Hide the progress bar
    pub const fn hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Build the progress tracker
    pub fn build(self) -> EnhancedProgressTracker {
        let config = ProgressConfig {
            style: self.style,
            width: self.width,
            show_eta: self.show_eta,
            show_statistics: self.show_statistics,
            show_speed: self.show_speed,
            adaptive_rate: self.adaptive_rate,
            min_update_interval: self.min_update_interval,
            max_update_interval: self.max_update_interval,
            template: self.template,
            symbols: self.symbols,
        };

        let mut tracker =
            EnhancedProgressTracker::new(&self.description, self.total).with_config(config);

        if self.hidden {
            tracker.hide();
        }

        tracker
    }
}
