//! Progress bar renderer
//!
//! This module handles the terminal rendering of progress bars with different
//! styles and formatting options.

use std::io::{self, Write};

use super::statistics::{format_duration, format_rate, ProgressStats};
use super::tracker::ProgressSymbols;

/// Progress bar renderer
pub struct ProgressRenderer {
    /// Terminal handling state
    last_length: usize,
    /// Spinner index for animated spinners
    spinner_index: usize,
}

impl ProgressRenderer {
    /// Create a new renderer
    pub fn new() -> Self {
        Self {
            last_length: 0,
            spinner_index: 0,
        }
    }

    /// Initialize terminal for progress rendering
    pub fn init(&mut self) {
        // Hide cursor for cleaner progress display
        print!("\x1b[?25l");
        let _ = io::stdout().flush();
    }

    /// Finalize terminal after progress rendering
    pub fn finalize(&mut self) {
        // Show cursor and print a newline to ensure next output starts on a fresh line
        print!("\x1b[?25h");
        println!();
        let _ = io::stdout().flush();
    }

    /// Render percentage-only progress
    pub fn render_percentage(&self, description: &str, stats: &ProgressStats) {
        let output = format!("{}: {:.1}%", description, stats.percentage);
        self.print_progress(&output);
    }

    /// Render basic progress bar
    pub fn render_bar(
        &self,
        description: &str,
        stats: &ProgressStats,
        width: usize,
        show_eta: bool,
        symbols: &ProgressSymbols,
    ) {
        let percentage = stats.percentage;
        let filled_width = ((percentage / 100.0) * width as f64) as usize;
        let empty_width = width.saturating_sub(filled_width);

        let progress_bar = format!(
            "{}{}{}{}",
            symbols.start,
            symbols.fill.repeat(filled_width),
            symbols.empty.repeat(empty_width),
            symbols.end,
        );

        let mut output = format!("{}: {} {:.1}%", description, progress_bar, percentage);

        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }

        self.print_progress(&output);
    }

    /// Render spinner progress
    pub fn render_spinner(
        &mut self,
        description: &str,
        stats: &ProgressStats,
        symbols: &ProgressSymbols,
        show_eta: bool,
    ) {
        self.spinner_index = (self.spinner_index + 1) % symbols.spinner.len();
        let spinner = &symbols.spinner[self.spinner_index];

        let mut output = format!(
            "{} {} {}/{} ({:.1}%)",
            spinner, description, stats.processed, stats.total, stats.percentage
        );

        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }

        self.print_progress(&output);
    }

    /// Render detailed progress bar with statistics
    pub fn render_detailed_bar(
        &self,
        description: &str,
        stats: &ProgressStats,
        width: usize,
        show_eta: bool,
        show_statistics: bool,
        show_speed: bool,
        symbols: &ProgressSymbols,
    ) {
        let percentage = stats.percentage;
        let filled_width = ((percentage / 100.0) * width as f64) as usize;
        let empty_width = width.saturating_sub(filled_width);

        let progress_bar = format!(
            "{}{}{}{}",
            symbols.start,
            symbols.fill.repeat(filled_width),
            symbols.empty.repeat(empty_width),
            symbols.end,
        );

        let mut output = format!(
            "{}: {} {:.1}% ({}/{})",
            description, progress_bar, percentage, stats.processed, stats.total
        );

        if show_speed {
            output.push_str(&format!(" [{}]", format_rate(stats.items_per_second)));
        }

        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }

        if show_statistics {
            output.push_str(&format!(" | Elapsed: {}", format_duration(&stats.elapsed)));

            if stats.max_speed > 0.0 {
                output.push_str(&format!(" | Peak: {}", format_rate(stats.max_speed)));
            }
        }

        self.print_progress(&output);
    }

    /// Render a compact progress display suitable for log files
    pub fn render_compact(&self, description: &str, stats: &ProgressStats) {
        let output = format!(
            "{}: {}/{} ({:.1}%) - {} - ETA: {}",
            description,
            stats.processed,
            stats.total,
            stats.percentage,
            format_rate(stats.items_per_second),
            format_duration(&stats.eta)
        );
        self.print_progress(&output);
    }

    /// Print progress with carriage return for in-place updates
    fn print_progress(&self, output: &str) {
        // Calculate the display width (handling Unicode characters)
        let output_width = console_width(output);

        // Clear previous output if it was longer
        if self.last_length > output_width {
            let clear_length = self.last_length - output_width;
            print!("\r{}{}", output, " ".repeat(clear_length));
        } else {
            print!("\r{}", output);
        }

        let _ = io::stdout().flush();
    }

    /// Update the stored length for proper clearing
    pub fn update_length(&mut self, length: usize) {
        self.last_length = length;
    }
}

impl Default for ProgressRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the display width of a string, accounting for Unicode
fn console_width(s: &str) -> usize {
    // This is a simplified implementation
    // For full Unicode support, you might want to use the `unicode-width` crate
    s.chars().count()
}

/// Generate a color-coded progress bar based on completion percentage
pub fn colored_progress_bar(percentage: f64, width: usize) -> String {
    let filled_width = ((percentage / 100.0) * width as f64) as usize;
    let empty_width = width.saturating_sub(filled_width);

    // Choose color based on progress
    let color = if percentage >= 90.0 {
        "\x1b[32m" // Green for near completion
    } else if percentage >= 50.0 {
        "\x1b[33m" // Yellow for halfway
    } else {
        "\x1b[31m" // Red for beginning
    };

    let reset = "\x1b[0m";

    format!(
        "│{}{}{}{}│",
        color,
        "█".repeat(filled_width),
        reset,
        " ".repeat(empty_width)
    )
}

/// Create an ASCII art progress visualization
pub fn ascii_art_progress(percentage: f64) -> String {
    let blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
    let width = 20;
    let progress = percentage / 100.0 * width as f64;
    let full_blocks = progress.floor() as usize;
    let partial_block = ((progress - progress.floor()) * (blocks.len() - 1) as f64) as usize;

    let mut result = String::new();
    result.push_str(&"█".repeat(full_blocks));

    if full_blocks < width && partial_block > 0 {
        result.push_str(blocks[partial_block]);
    }

    let remaining = width - full_blocks - if partial_block > 0 { 1 } else { 0 };
    result.push_str(&" ".repeat(remaining));

    format!("│{}│", result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = ProgressRenderer::new();
        assert_eq!(renderer.last_length, 0);
        assert_eq!(renderer.spinner_index, 0);
    }

    #[test]
    fn test_colored_progress_bar() {
        let bar_low = colored_progress_bar(25.0, 10);
        assert!(bar_low.contains("\x1b[31m")); // Red for low progress

        let bar_high = colored_progress_bar(95.0, 10);
        assert!(bar_high.contains("\x1b[32m")); // Green for high progress
    }

    #[test]
    fn test_ascii_art_progress() {
        let art = ascii_art_progress(50.0);
        assert!(art.contains("│"));
        assert!(art.contains("█"));
    }

    #[test]
    fn test_console_width() {
        assert_eq!(console_width("hello"), 5);
        assert_eq!(console_width("test 123"), 8);
    }
}
