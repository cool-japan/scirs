//! Progress formatting utilities
//!
//! This module provides various formatting options for progress displays,
//! including templates, themes, and specialized output formats.

use super::statistics::{format_duration, format_rate, ProgressStats};
use super::tracker::ProgressSymbols;

/// Progress display template
pub struct ProgressTemplate {
    /// Template string with placeholders
    template: String,
}

impl ProgressTemplate {
    /// Create a new progress template
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    /// Default template for detailed progress
    pub fn detailed() -> Self {
        Self::new("{description}: {bar} {percentage:>6.1}% | {processed}/{total} | {rate} | ETA: {eta} | Elapsed: {elapsed}")
    }

    /// Compact template for minimal display
    pub fn compact() -> Self {
        Self::new("{description}: {percentage:.1}% ({processed}/{total}) ETA: {eta}")
    }

    /// Template suitable for log files
    pub fn log_format() -> Self {
        Self::new("[{timestamp}] {description}: {percentage:.1}% complete ({processed}/{total}) - {rate} - ETA: {eta}")
    }

    /// Template for scientific computation logging
    pub fn scientific() -> Self {
        Self::new("{description}: Progress={percentage:>6.2}% Rate={rate:>10} Remaining={remaining:>8} ETA={eta}")
    }

    /// Render the template with progress statistics
    pub fn render(&self, description: &str, stats: &ProgressStats, bar: Option<&str>) -> String {
        let mut result = self.template.clone();

        // Basic replacements
        result = result.replace("{description}", description);
        result = result.replace("{percentage}", &format!("{:.1}", stats.percentage));
        result = result.replace("{processed}", &stats.processed.to_string());
        result = result.replace("{total}", &stats.total.to_string());
        result = result.replace("{remaining}", &stats.remaining().to_string());
        result = result.replace("{rate}", &format_rate(stats.items_per_second));
        result = result.replace("{eta}", &format_duration(&stats.eta));
        result = result.replace("{elapsed}", &format_duration(&stats.elapsed));

        // Formatted percentage with custom precision
        if let Some(captures) = extract_format_spec(&result, "percentage") {
            let formatted = format!(
                "{:width$.precision$}",
                stats.percentage,
                width = captures.width.unwrap_or(0),
                precision = captures.precision.unwrap_or(1)
            );
            result = result.replace(&captures.original, &formatted);
        }

        // Progress bar
        if let Some(bar_str) = bar {
            result = result.replace("{bar}", bar_str);
        }

        // Timestamp
        if result.contains("{timestamp}") {
            let now = chrono::Utc::now();
            result = result.replace("{timestamp}", &now.format("%Y-%m-%d %H:%M:%S").to_string());
        }

        // Additional custom processing could be added here

        result
    }
}

/// Format specification extracted from template
#[derive(Debug)]
struct FormatSpec {
    original: String,
    width: Option<usize>,
    precision: Option<usize>,
    #[allow(dead_code)]
    alignment: Option<char>,
}

/// Extract format specification from a placeholder
fn extract_format_spec(text: &str, field: &str) -> Option<FormatSpec> {
    let pattern = format!("{{{}", field);
    if let Some(start) = text.find(&pattern) {
        if let Some(end) = text[start..].find('}') {
            let spec_str = &text[start..start + end + 1];

            // Parse format specification like {percentage:>6.1}
            if let Some(colon_pos) = spec_str.find(':') {
                let format_part = &spec_str[colon_pos + 1..spec_str.len() - 1];

                let mut width = None;
                let mut precision = None;
                let mut alignment = None;

                // Parse alignment
                if format_part.starts_with('<')
                    || format_part.starts_with('>')
                    || format_part.starts_with('^')
                {
                    alignment = format_part.chars().next();
                }

                // Parse width and precision
                let numeric_part = format_part.trim_start_matches(['<', '>', '^']);
                if let Some(dot_pos) = numeric_part.find('.') {
                    if let Ok(w) = numeric_part[..dot_pos].parse::<usize>() {
                        width = Some(w);
                    }
                    if let Ok(p) = numeric_part[dot_pos + 1..].parse::<usize>() {
                        precision = Some(p);
                    }
                } else if let Ok(w) = numeric_part.parse::<usize>() {
                    width = Some(w);
                }

                return Some(FormatSpec {
                    original: spec_str.to_string(),
                    width,
                    precision,
                    alignment,
                });
            }
        }
    }
    None
}

/// Progress display theme
#[derive(Debug, Clone, Default)]
pub struct ProgressTheme {
    /// Symbols for progress visualization
    pub symbols: ProgressSymbols,
    /// Color scheme
    pub colors: ColorScheme,
    /// Animation settings
    pub animation: AnimationSettings,
}

/// Color scheme for progress display
#[derive(Debug, Clone, Default)]
pub struct ColorScheme {
    /// Color for progress bar fill
    pub fill_color: Option<String>,
    /// Color for progress bar empty
    pub empty_color: Option<String>,
    /// Color for text
    pub text_color: Option<String>,
    /// Color for percentage
    pub percentage_color: Option<String>,
    /// Color for ETA
    pub eta_color: Option<String>,
}

/// Animation settings
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    /// Animation speed (frames per second)
    pub fps: f64,
    /// Whether to animate spinner
    pub animate_spinner: bool,
    /// Whether to animate progress bar
    pub animate_bar: bool,
}

impl ProgressTheme {
    /// Modern theme with Unicode blocks
    pub fn modern() -> Self {
        Self {
            symbols: ProgressSymbols::blocks(),
            colors: ColorScheme::colorful(),
            animation: AnimationSettings::smooth(),
        }
    }

    /// Minimal theme for simple terminals
    pub fn minimal() -> Self {
        Self {
            symbols: ProgressSymbols {
                start: "[".to_string(),
                end: "]".to_string(),
                fill: "#".to_string(),
                empty: "-".to_string(),
                spinner: vec![
                    "|".to_string(),
                    "/".to_string(),
                    "-".to_string(),
                    "\\".to_string(),
                ],
            },
            colors: ColorScheme::monochrome(),
            animation: AnimationSettings::slow(),
        }
    }

    /// Scientific theme with precise formatting
    pub fn scientific() -> Self {
        Self {
            symbols: ProgressSymbols {
                start: "│".to_string(),
                end: "│".to_string(),
                fill: "█".to_string(),
                empty: "░".to_string(),
                spinner: vec![
                    "◐".to_string(),
                    "◓".to_string(),
                    "◑".to_string(),
                    "◒".to_string(),
                ],
            },
            colors: ColorScheme::scientific(),
            animation: AnimationSettings::precise(),
        }
    }
}

impl ColorScheme {
    /// Colorful scheme with ANSI colors
    pub fn colorful() -> Self {
        Self {
            fill_color: Some("\x1b[32m".to_string()),       // Green
            empty_color: Some("\x1b[90m".to_string()),      // Dark gray
            text_color: Some("\x1b[37m".to_string()),       // White
            percentage_color: Some("\x1b[36m".to_string()), // Cyan
            eta_color: Some("\x1b[33m".to_string()),        // Yellow
        }
    }

    /// Monochrome scheme
    pub fn monochrome() -> Self {
        Self::default()
    }

    /// Scientific color scheme with subtle colors
    pub fn scientific() -> Self {
        Self {
            fill_color: Some("\x1b[34m".to_string()),  // Blue
            empty_color: Some("\x1b[90m".to_string()), // Dark gray
            text_color: None,
            percentage_color: Some("\x1b[1m".to_string()), // Bold
            eta_color: Some("\x1b[2m".to_string()),        // Dim
        }
    }

    /// Apply color to text
    pub fn apply_color(&self, text: &str, color_type: ColorType) -> String {
        let color = match color_type {
            ColorType::Fill => &self.fill_color,
            ColorType::Empty => &self.empty_color,
            ColorType::Text => &self.text_color,
            ColorType::Percentage => &self.percentage_color,
            ColorType::ETA => &self.eta_color,
        };

        if let Some(color_code) = color {
            format!("{}{}\x1b[0m", color_code, text)
        } else {
            text.to_string()
        }
    }
}

/// Color type for applying colors
#[derive(Debug, Clone, Copy)]
pub enum ColorType {
    Fill,
    Empty,
    Text,
    Percentage,
    ETA,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            fps: 2.0,
            animate_spinner: true,
            animate_bar: false,
        }
    }
}

impl AnimationSettings {
    /// Smooth animation settings
    pub fn smooth() -> Self {
        Self {
            fps: 5.0,
            animate_spinner: true,
            animate_bar: true,
        }
    }

    /// Slow animation settings
    pub fn slow() -> Self {
        Self {
            fps: 1.0,
            animate_spinner: true,
            animate_bar: false,
        }
    }

    /// Precise animation for scientific use
    pub fn precise() -> Self {
        Self {
            fps: 1.0,
            animate_spinner: false,
            animate_bar: false,
        }
    }

    /// Get update interval based on FPS
    pub fn update_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(1.0 / self.fps)
    }
}

/// Specialized formatter for different output formats
pub struct ProgressFormatter;

impl ProgressFormatter {
    /// Format for JSON output
    pub fn json(description: &str, stats: &ProgressStats) -> String {
        serde_json::json!({
            "description": description,
            "processed": stats.processed,
            "total": stats.total,
            "percentage": stats.percentage,
            "rate": stats.items_per_second,
            "eta_seconds": stats.eta.as_secs(),
            "elapsed_seconds": stats.elapsed.as_secs()
        })
        .to_string()
    }

    /// Format for CSV output
    pub fn csv(description: &str, stats: &ProgressStats) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{}",
            description,
            stats.processed,
            stats.total,
            stats.percentage,
            stats.items_per_second,
            stats.eta.as_secs(),
            stats.elapsed.as_secs()
        )
    }

    /// Format for machine-readable output
    pub fn machine(description: &str, stats: &ProgressStats) -> String {
        format!(
            "PROGRESS|{}|{}|{}|{:.2}|{:.2}|{}|{}",
            description,
            stats.processed,
            stats.total,
            stats.percentage,
            stats.items_per_second,
            stats.eta.as_secs(),
            stats.elapsed.as_secs()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_template_render() {
        let template =
            ProgressTemplate::new("{description}: {percentage:.1}% ({processed}/{total})");
        let stats = ProgressStats::new(100);

        let result = template.render("Test", &stats, None);
        assert!(result.contains("Test"));
        assert!(result.contains("0.0%"));
        assert!(result.contains("0/100"));
    }

    #[test]
    fn test_format_spec_extraction() {
        let spec = extract_format_spec("{percentage:>6.1}", "percentage");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.width, Some(6));
        assert_eq!(spec.precision, Some(1));
    }

    #[test]
    fn test_color_scheme_apply() {
        let colors = ColorScheme::colorful();
        let colored = colors.apply_color("test", ColorType::Fill);
        assert!(colored.contains("\x1b[32m")); // Green
        assert!(colored.contains("\x1b[0m")); // Reset
    }

    #[test]
    fn test_progress_formatter_json() {
        let stats = ProgressStats::new(100);
        let json_output = ProgressFormatter::json("Test", &stats);
        assert!(json_output.contains("\"description\":\"Test\""));
        assert!(json_output.contains("\"total\":100"));
    }

    #[test]
    fn test_progress_formatter_csv() {
        let stats = ProgressStats::new(100);
        let csv_output = ProgressFormatter::csv("Test", &stats);
        assert!(csv_output.starts_with("Test,"));
        assert!(csv_output.contains(",100,"));
    }
}
