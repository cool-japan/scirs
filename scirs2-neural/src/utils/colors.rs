//! Terminal color utilities for visualization output
//!
//! This module provides utilities for adding colors to terminal output.

use std::fmt::Display;

/// ANSI color codes
#[derive(Debug, Clone, Copy)]
pub enum Color {
    /// Black color
    Black,
    /// Red color
    Red,
    /// Green color
    Green,
    /// Yellow color
    Yellow,
    /// Blue color
    Blue,
    /// Magenta color
    Magenta,
    /// Cyan color
    Cyan,
    /// White color
    White,
    /// Bright black (gray) color
    BrightBlack,
    /// Bright red color
    BrightRed,
    /// Bright green color
    BrightGreen,
    /// Bright yellow color
    BrightYellow,
    /// Bright blue color
    BrightBlue,
    /// Bright magenta color
    BrightMagenta,
    /// Bright cyan color
    BrightCyan,
    /// Bright white color
    BrightWhite,
}

impl Color {
    /// Get the ANSI foreground color code
    pub fn fg_code(&self) -> &'static str {
        match self {
            Color::Black => "\x1b[30m",
            Color::Red => "\x1b[31m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Blue => "\x1b[34m",
            Color::Magenta => "\x1b[35m",
            Color::Cyan => "\x1b[36m",
            Color::White => "\x1b[37m",
            Color::BrightBlack => "\x1b[90m",
            Color::BrightRed => "\x1b[91m",
            Color::BrightGreen => "\x1b[92m",
            Color::BrightYellow => "\x1b[93m",
            Color::BrightBlue => "\x1b[94m",
            Color::BrightMagenta => "\x1b[95m",
            Color::BrightCyan => "\x1b[96m",
            Color::BrightWhite => "\x1b[97m",
        }
    }

    /// Get the ANSI background color code
    pub fn bg_code(&self) -> &'static str {
        match self {
            Color::Black => "\x1b[40m",
            Color::Red => "\x1b[41m",
            Color::Green => "\x1b[42m",
            Color::Yellow => "\x1b[43m",
            Color::Blue => "\x1b[44m",
            Color::Magenta => "\x1b[45m",
            Color::Cyan => "\x1b[46m",
            Color::White => "\x1b[47m",
            Color::BrightBlack => "\x1b[100m",
            Color::BrightRed => "\x1b[101m",
            Color::BrightGreen => "\x1b[102m",
            Color::BrightYellow => "\x1b[103m",
            Color::BrightBlue => "\x1b[104m",
            Color::BrightMagenta => "\x1b[105m",
            Color::BrightCyan => "\x1b[106m",
            Color::BrightWhite => "\x1b[107m",
        }
    }
}

/// ANSI text style
#[derive(Debug, Clone, Copy)]
pub enum Style {
    /// Bold text
    Bold,
    /// Italic text
    Italic,
    /// Underlined text
    Underline,
    /// Blinking text
    Blink,
    /// Inverted colors
    Reverse,
    /// Dim text
    Dim,
}

impl Style {
    /// Get the ANSI style code
    pub fn code(&self) -> &'static str {
        match self {
            Style::Bold => "\x1b[1m",
            Style::Italic => "\x1b[3m",
            Style::Underline => "\x1b[4m",
            Style::Blink => "\x1b[5m",
            Style::Reverse => "\x1b[7m",
            Style::Dim => "\x1b[2m",
        }
    }
}

/// ANSI reset code
pub const RESET: &str = "\x1b[0m";

/// Colorize text with a foreground color
pub fn colorize<T: Display>(text: T, color: Color) -> String {
    format!("{}{}{}", color.fg_code(), text, RESET)
}

/// Colorize text with a background color
pub fn colorize_bg<T: Display>(text: T, color: Color) -> String {
    format!("{}{}{}", color.bg_code(), text, RESET)
}

/// Style text with a text style
pub fn stylize<T: Display>(text: T, style: Style) -> String {
    format!("{}{}{}", style.code(), text, RESET)
}

/// Colorize and style text
pub fn colorize_and_style<T: Display>(
    text: T,
    fg_color: Option<Color>,
    bg_color: Option<Color>,
    style: Option<Style>,
) -> String {
    let mut result = String::new();

    if let Some(fg) = fg_color {
        result.push_str(fg.fg_code());
    }

    if let Some(bg) = bg_color {
        result.push_str(bg.bg_code());
    }

    if let Some(s) = style {
        result.push_str(s.code());
    }

    result.push_str(&format!("{}", text));
    result.push_str(RESET);

    result
}

/// Detect if the terminal supports colors
///
/// This is a simple heuristic based on environment variables
pub fn supports_color() -> bool {
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return false;
        }
    }

    if let Ok(no_color) = std::env::var("NO_COLOR") {
        if !no_color.is_empty() {
            return false;
        }
    }

    if let Ok(color) = std::env::var("FORCE_COLOR") {
        if !color.is_empty() {
            return true;
        }
    }

    // Check if running on GitHub Actions
    if std::env::var("GITHUB_ACTIONS").is_ok() {
        return true;
    }

    // Use cfg to branch platform-specific code
    #[cfg(not(target_os = "windows"))]
    {
        // Simple heuristic: assume color is supported on non-Windows platforms
        true
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows, check for common terminals that support colors
        if let Ok(term) = std::env::var("TERM_PROGRAM") {
            if term == "vscode" || term == "mintty" || term == "alacritty" {
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Colorization options for terminal output
#[derive(Debug, Clone, Copy)]
pub struct ColorOptions {
    /// Whether to use colors
    pub enabled: bool,
    /// Whether to use background colors
    pub use_background: bool,
    /// Whether to use bright colors
    pub use_bright: bool,
}

impl Default for ColorOptions {
    fn default() -> Self {
        Self {
            enabled: supports_color(),
            use_background: true,
            use_bright: true,
        }
    }
}

/// Get the appropriate color based on a value's position in a range
///
/// Returns red for values close to 0.0, yellow for values around 0.5,
/// and green for values close to 1.0
pub fn gradient_color(value: f64, options: &ColorOptions) -> Option<Color> {
    if !options.enabled {
        return None;
    }

    if !(0.0..=1.0).contains(&value) {
        return None;
    }

    // Red -> Yellow -> Green gradient
    if value < 0.5 {
        // Red to Yellow (0.0 -> 0.5)
        if options.use_bright {
            Some(Color::BrightRed)
        } else {
            Some(Color::Red)
        }
    } else if value < 0.7 {
        // Yellow (0.5 -> 0.7)
        if options.use_bright {
            Some(Color::BrightYellow)
        } else {
            Some(Color::Yellow)
        }
    } else {
        // Green (0.7 -> 1.0)
        if options.use_bright {
            Some(Color::BrightGreen)
        } else {
            Some(Color::Green)
        }
    }
}

/// Generate a more fine-grained gradient color for heatmap visualizations
///
/// This provides a more detailed color spectrum for visualizing data with subtle differences
/// Returns a spectrum from cool (blues/purples) for low values to warm (reds/yellows) for high values
pub fn heatmap_gradient_color(value: f64, options: &ColorOptions) -> Option<Color> {
    if !options.enabled {
        return None;
    }

    if !(0.0..=1.0).contains(&value) {
        return None;
    }

    // More detailed gradient with 5 color stops
    if value < 0.2 {
        // Very low values (0.0 -> 0.2)
        if options.use_bright {
            Some(Color::BrightBlue)
        } else {
            Some(Color::Blue)
        }
    } else if value < 0.4 {
        // Low values (0.2 -> 0.4)
        if options.use_bright {
            Some(Color::BrightCyan)
        } else {
            Some(Color::Cyan)
        }
    } else if value < 0.6 {
        // Medium values (0.4 -> 0.6)
        if options.use_bright {
            Some(Color::BrightYellow)
        } else {
            Some(Color::Yellow)
        }
    } else if value < 0.8 {
        // High values (0.6 -> 0.8)
        if options.use_bright {
            Some(Color::BrightRed)
        } else {
            Some(Color::Red)
        }
    } else {
        // Very high values (0.8 -> 1.0)
        if options.use_bright {
            Some(Color::BrightMagenta)
        } else {
            Some(Color::Magenta)
        }
    }
}

/// Generate table cell content with appropriate color based on value
pub fn colored_metric_cell<T: Display>(
    value: T,
    normalized_value: f64,
    options: &ColorOptions,
) -> String {
    if !options.enabled {
        return format!("{}", value);
    }

    if let Some(color) = gradient_color(normalized_value, options) {
        colorize(value, color)
    } else {
        format!("{}", value)
    }
}

/// Generate a heatmap cell with color gradient for confusion matrix
pub fn heatmap_cell<T: Display>(value: T, normalized_value: f64, options: &ColorOptions) -> String {
    if !options.enabled {
        return format!("{}", value);
    }

    if let Some(color) = heatmap_gradient_color(normalized_value, options) {
        // For higher values, use bold to emphasize importance
        if normalized_value > 0.7 {
            colorize(stylize(value, Style::Bold), color)
        } else {
            colorize(value, color)
        }
    } else {
        format!("{}", value)
    }
}

/// Build a color legend for confusion matrix or other visualizations
pub fn color_legend(options: &ColorOptions) -> Option<String> {
    if !options.enabled {
        return None;
    }

    let mut legend = String::from("Color Legend: ");

    let low_color = if options.use_bright {
        Color::BrightRed
    } else {
        Color::Red
    };
    let mid_color = if options.use_bright {
        Color::BrightYellow
    } else {
        Color::Yellow
    };
    let high_color = if options.use_bright {
        Color::BrightGreen
    } else {
        Color::Green
    };

    legend.push_str(&format!("{} Low (0.0-0.5) ", colorize("■", low_color)));
    legend.push_str(&format!("{} Medium (0.5-0.7) ", colorize("■", mid_color)));
    legend.push_str(&format!("{} High (0.7-1.0)", colorize("■", high_color)));

    Some(legend)
}

/// Build a detailed heatmap color legend
pub fn heatmap_color_legend(options: &ColorOptions) -> Option<String> {
    if !options.enabled {
        return None;
    }

    let mut legend = String::from("Heatmap Legend: ");

    let colors = [
        (
            if options.use_bright {
                Color::BrightBlue
            } else {
                Color::Blue
            },
            "Very Low (0.0-0.2)",
        ),
        (
            if options.use_bright {
                Color::BrightCyan
            } else {
                Color::Cyan
            },
            "Low (0.2-0.4)",
        ),
        (
            if options.use_bright {
                Color::BrightYellow
            } else {
                Color::Yellow
            },
            "Medium (0.4-0.6)",
        ),
        (
            if options.use_bright {
                Color::BrightRed
            } else {
                Color::Red
            },
            "High (0.6-0.8)",
        ),
        (
            if options.use_bright {
                Color::BrightMagenta
            } else {
                Color::Magenta
            },
            "Very High (0.8-1.0)",
        ),
    ];

    for (i, (color, label)) in colors.iter().enumerate() {
        if i > 0 {
            legend.push(' ');
        }
        legend.push_str(&format!("{} {}", colorize("■", *color), label));
    }

    Some(legend)
}

/// Helper function to conditionally create a string with ANSI coloring for terminal output
pub fn colored_string<T: Display>(
    content: T,
    color: Option<Color>,
    style: Option<Style>,
) -> String {
    match (color, style) {
        (Some(c), Some(s)) => colorize_and_style(content, Some(c), None, Some(s)),
        (Some(c), None) => colorize(content, c),
        (None, Some(s)) => stylize(content, s),
        (None, None) => content.to_string(),
    }
}
