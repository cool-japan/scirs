//! Core Types and Configurations for Visualization
//!
//! This module contains all the fundamental data structures, enums, and configuration
//! types used throughout the visualization system.

/// Color map types for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Grayscale color map
    Gray,
    /// Jet color map (blue to red)
    Jet,
    /// Viridis perceptually uniform color map
    Viridis,
    /// Plasma color map
    Plasma,
    /// Inferno color map
    Inferno,
    /// Hot color map (black to white through red/yellow)
    Hot,
    /// Cool color map (cyan to magenta)
    Cool,
    /// Spring color map (magenta to yellow)
    Spring,
    /// Summer color map (green to yellow)
    Summer,
    /// Autumn color map (red to yellow)
    Autumn,
    /// Winter color map (blue to green)
    Winter,
}

/// Report output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// HTML format with styling
    Html,
    /// Markdown format
    Markdown,
    /// Plain text format
    Text,
}

/// RGB color representation
#[derive(Debug, Clone, Copy)]
pub struct RgbColor {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl RgbColor {
    /// Create a new RGB color
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to hexadecimal color string
    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Convert to RGB tuple
    pub fn to_tuple(&self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    /// Create from hex string (e.g., "#FF0000" for red)
    pub fn from_hex(hex: &str) -> Result<Self, &'static str> {
        if !hex.starts_with('#') || hex.len() != 7 {
            return Err("Invalid hex format, expected #RRGGBB");
        }

        let hex = &hex[1..];
        let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "Invalid red component")?;
        let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "Invalid green component")?;
        let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "Invalid blue component")?;

        Ok(RgbColor::new(r, g, b))
    }

    /// Create from HSV values (hue: 0-360, saturation: 0-1, value: 0-1)
    pub fn from_hsv(h: f64, s: f64, v: f64) -> Self {
        let h = h.rem_euclid(360.0);
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        RgbColor::new(
            ((r_prime + m) * 255.0) as u8,
            ((g_prime + m) * 255.0) as u8,
            ((b_prime + m) * 255.0) as u8,
        )
    }
}

/// Configuration for plotting operations
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the plot in pixels
    pub width: usize,
    /// Height of the plot in pixels
    pub height: usize,
    /// Title of the plot
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Color map to use
    pub colormap: ColorMap,
    /// Whether to show grid
    pub show_grid: bool,
    /// Number of bins for histograms
    pub num_bins: usize,
    /// Output format
    pub format: ReportFormat,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Image Analysis Plot".to_string(),
            xlabel: "X".to_string(),
            ylabel: "Y".to_string(),
            colormap: ColorMap::Viridis,
            show_grid: true,
            num_bins: 256,
            format: ReportFormat::Text,
        }
    }
}

impl PlotConfig {
    /// Create a new plot configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the plot dimensions
    pub fn with_dimensions(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the plot title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the axis labels
    pub fn with_labels(mut self, xlabel: impl Into<String>, ylabel: impl Into<String>) -> Self {
        self.xlabel = xlabel.into();
        self.ylabel = ylabel.into();
        self
    }

    /// Set the color map
    pub fn with_colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self
    }

    /// Set the output format
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the number of bins for histograms
    pub fn with_bins(mut self, num_bins: usize) -> Self {
        self.num_bins = num_bins;
        self
    }

    /// Enable or disable grid display
    pub fn with_grid(mut self, show_grid: bool) -> Self {
        self.show_grid = show_grid;
        self
    }
}

/// Configuration for report generation
#[derive(Debug, Clone)]
pub struct ReportConfig {
    /// Title of the report
    pub title: String,
    /// Author information
    pub author: String,
    /// Include detailed statistics
    pub include_statistics: bool,
    /// Include quality metrics
    pub include_qualitymetrics: bool,
    /// Include texture analysis
    pub includetexture_analysis: bool,
    /// Include histograms
    pub include_histograms: bool,
    /// Include profile plots
    pub include_profiles: bool,
    /// Output format (html, markdown, text)
    pub format: ReportFormat,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "Image Analysis Report".to_string(),
            author: "SciRS2 Image Analysis".to_string(),
            include_statistics: true,
            include_qualitymetrics: true,
            includetexture_analysis: true,
            include_histograms: true,
            include_profiles: true,
            format: ReportFormat::Markdown,
        }
    }
}

impl ReportConfig {
    /// Create a new report configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the report title and author
    pub fn with_header(mut self, title: impl Into<String>, author: impl Into<String>) -> Self {
        self.title = title.into();
        self.author = author.into();
        self
    }

    /// Set the output format
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.format = format;
        self
    }

    /// Configure which sections to include
    pub fn with_sections(
        mut self,
        statistics: bool,
        quality_metrics: bool,
        texture_analysis: bool,
        histograms: bool,
        profiles: bool,
    ) -> Self {
        self.include_statistics = statistics;
        self.include_qualitymetrics = quality_metrics;
        self.includetexture_analysis = texture_analysis;
        self.include_histograms = histograms;
        self.include_profiles = profiles;
        self
    }

    /// Enable all sections
    pub fn with_all_sections(mut self) -> Self {
        self.include_statistics = true;
        self.include_qualitymetrics = true;
        self.includetexture_analysis = true;
        self.include_histograms = true;
        self.include_profiles = true;
        self
    }

    /// Disable all optional sections (only basic info)
    pub fn minimal(mut self) -> Self {
        self.include_statistics = false;
        self.include_qualitymetrics = false;
        self.includetexture_analysis = false;
        self.include_histograms = false;
        self.include_profiles = false;
        self
    }
}

/// Predefined color schemes for different visualization purposes
pub struct ColorSchemes;

impl ColorSchemes {
    /// Scientific publication-friendly colors
    pub fn scientific() -> [RgbColor; 6] {
        [
            RgbColor::new(31, 120, 180), // Blue
            RgbColor::new(51, 160, 44),  // Green
            RgbColor::new(227, 26, 28),  // Red
            RgbColor::new(255, 127, 0),  // Orange
            RgbColor::new(106, 61, 154), // Purple
            RgbColor::new(177, 89, 40),  // Brown
        ]
    }

    /// Colorblind-friendly palette
    pub fn colorblind_friendly() -> [RgbColor; 8] {
        [
            RgbColor::new(0, 0, 0),       // Black
            RgbColor::new(230, 159, 0),   // Orange
            RgbColor::new(86, 180, 233),  // Sky blue
            RgbColor::new(0, 158, 115),   // Bluish green
            RgbColor::new(240, 228, 66),  // Yellow
            RgbColor::new(0, 114, 178),   // Blue
            RgbColor::new(213, 94, 0),    // Vermillion
            RgbColor::new(204, 121, 167), // Reddish purple
        ]
    }

    /// High contrast colors for presentations
    pub fn high_contrast() -> [RgbColor; 4] {
        [
            RgbColor::new(0, 0, 0),       // Black
            RgbColor::new(255, 255, 255), // White
            RgbColor::new(255, 0, 0),     // Red
            RgbColor::new(0, 0, 255),     // Blue
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_color_creation() {
        let color = RgbColor::new(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
    }

    #[test]
    fn test_rgb_to_hex() {
        let color = RgbColor::new(255, 0, 0);
        assert_eq!(color.to_hex(), "#ff0000");

        let color = RgbColor::new(0, 255, 0);
        assert_eq!(color.to_hex(), "#00ff00");

        let color = RgbColor::new(0, 0, 255);
        assert_eq!(color.to_hex(), "#0000ff");
    }

    #[test]
    fn test_rgb_from_hex() {
        let color = RgbColor::from_hex("#FF0000").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);

        let color = RgbColor::from_hex("#00FF00").unwrap();
        assert_eq!(color.r, 0);
        assert_eq!(color.g, 255);
        assert_eq!(color.b, 0);

        // Test invalid formats
        assert!(RgbColor::from_hex("FF0000").is_err());
        assert!(RgbColor::from_hex("#FF00").is_err());
        assert!(RgbColor::from_hex("#GGGGGG").is_err());
    }

    #[test]
    fn test_rgb_from_hsv() {
        // Red (0 degrees)
        let red = RgbColor::from_hsv(0.0, 1.0, 1.0);
        assert_eq!(red.r, 255);
        assert!(red.g < 5); // Should be 0, but allow for rounding
        assert!(red.b < 5);

        // Green (120 degrees)
        let green = RgbColor::from_hsv(120.0, 1.0, 1.0);
        assert!(green.r < 5);
        assert_eq!(green.g, 255);
        assert!(green.b < 5);

        // Blue (240 degrees)
        let blue = RgbColor::from_hsv(240.0, 1.0, 1.0);
        assert!(blue.r < 5);
        assert!(blue.g < 5);
        assert_eq!(blue.b, 255);
    }

    #[test]
    fn test_plot_config_builder() {
        let config = PlotConfig::new()
            .with_dimensions(1024, 768)
            .with_title("Test Plot")
            .with_labels("X Axis", "Y Axis")
            .with_colormap(ColorMap::Jet)
            .with_bins(128)
            .with_grid(false);

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.xlabel, "X Axis");
        assert_eq!(config.ylabel, "Y Axis");
        assert_eq!(config.colormap, ColorMap::Jet);
        assert_eq!(config.num_bins, 128);
        assert!(!config.show_grid);
    }

    #[test]
    fn test_report_config_builder() {
        let config = ReportConfig::new()
            .with_header("Custom Report", "Test Author")
            .with_format(ReportFormat::Html)
            .with_sections(true, true, false, true, false);

        assert_eq!(config.title, "Custom Report");
        assert_eq!(config.author, "Test Author");
        assert_eq!(config.format, ReportFormat::Html);
        assert!(config.include_statistics);
        assert!(config.include_qualitymetrics);
        assert!(!config.includetexture_analysis);
        assert!(config.include_histograms);
        assert!(!config.include_profiles);
    }

    #[test]
    fn test_color_schemes() {
        let scientific = ColorSchemes::scientific();
        assert_eq!(scientific.len(), 6);

        let colorblind = ColorSchemes::colorblind_friendly();
        assert_eq!(colorblind.len(), 8);

        let contrast = ColorSchemes::high_contrast();
        assert_eq!(contrast.len(), 4);

        // Test that first color in high contrast is black
        assert_eq!(contrast[0].r, 0);
        assert_eq!(contrast[0].g, 0);
        assert_eq!(contrast[0].b, 0);
    }

    #[test]
    fn test_colormap_enum() {
        // Test enum equality
        assert_eq!(ColorMap::Viridis, ColorMap::Viridis);
        assert_ne!(ColorMap::Viridis, ColorMap::Jet);

        // Test debug formatting
        let colormap = ColorMap::Plasma;
        let debug_str = format!("{:?}", colormap);
        assert!(debug_str.contains("Plasma"));
    }

    #[test]
    fn test_report_format_enum() {
        let format = ReportFormat::Markdown;
        assert_eq!(format, ReportFormat::Markdown);
        assert_ne!(format, ReportFormat::Html);

        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("Markdown"));
    }
}
