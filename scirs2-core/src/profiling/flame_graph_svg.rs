//! # SVG Flame Graph Export
//!
//! This module provides functionality to export flame graphs as interactive SVG files
//! that can be viewed in web browsers with zoom, search, and tooltip functionality.

use super::advanced::FlameGraphNode;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Duration;

/// SVG Flame Graph Configuration
#[derive(Debug, Clone)]
pub struct SvgFlameGraphConfig {
    /// Width of the SVG output
    pub width: u32,
    /// Height of the SVG output  
    pub height: u32,
    /// Font family for text
    pub font_family: String,
    /// Font size
    pub font_size: u32,
    /// Minimum width for a flame to be visible
    pub min_width: f64,
    /// Color scheme for the flames
    pub color_scheme: ColorScheme,
    /// Whether to include JavaScript interactivity
    pub interactive: bool,
    /// Title of the flame graph
    pub title: String,
    /// Subtitle with additional information
    pub subtitle: String,
}

impl Default for SvgFlameGraphConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 600,
            font_family: "Verdana, sans-serif".to_string(),
            font_size: 12,
            min_width: 0.1,
            color_scheme: ColorScheme::Spectral,
            interactive: true,
            title: "Flame Graph".to_string(),
            subtitle: "".to_string(),
        }
    }
}

/// Color schemes for flame graph visualization
#[derive(Debug, Clone, PartialEq)]
pub enum ColorScheme {
    /// Spectral color scheme (default)
    Spectral,
    /// Hot color scheme (reds and yellows)
    Hot,
    /// Cool color scheme (blues and greens)
    Cool,
    /// Grayscale
    Grayscale,
    /// Java color scheme (optimized for Java profiling)
    Java,
    /// Memory color scheme (for memory profiling)
    Memory,
}

impl ColorScheme {
    /// Get color for a given heat value (0.0 to 1.0)
    fn get_color(&self, heat: f64, function_name: &str) -> String {
        let heat = heat.clamp(0.0, 1.0);

        match self {
            ColorScheme::Spectral => {
                let r = (255.0 * (1.0 - heat)) as u8;
                let g = (255.0
                    * (if heat < 0.5 {
                        2.0 * heat
                    } else {
                        2.0 * (1.0 - heat)
                    })) as u8;
                let b = (255.0 * heat) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorScheme::Hot => {
                let r = (255.0 * heat.sqrt()) as u8;
                let g = (255.0 * heat.powi(2)) as u8;
                let b = (128.0 * heat.powi(3)) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorScheme::Cool => {
                let r = (128.0 * (1.0 - heat)) as u8;
                let g = (200.0 * (1.0 - heat * 0.5)) as u8;
                let b = (255.0 * (0.7 + 0.3 * heat)) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorScheme::Grayscale => {
                let intensity = (255.0 * (0.2 + 0.8 * (1.0 - heat))) as u8;
                format!("rgb({},{},{})", intensity, intensity, intensity)
            }
            ColorScheme::Java => {
                // Use function name hash for consistent coloring
                let hash = Self::hash_string(function_name);
                let hue = (hash % 360) as f64;
                let saturation = 70.0 + 30.0 * heat;
                let lightness = 60.0 - 20.0 * heat;
                Self::hsl_to_rgb(hue, saturation, lightness)
            }
            ColorScheme::Memory => {
                let r = (255.0 * heat) as u8;
                let g = (200.0 * (1.0 - heat * 0.7)) as u8;
                let b = (100.0 * (1.0 - heat)) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
        }
    }

    /// Simple hash function for string
    fn hash_string(s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    /// Convert HSL to RGB
    fn hsl_to_rgb(h: f64, s: f64, l: f64) -> String {
        let h = h / 360.0;
        let s = s / 100.0;
        let l = l / 100.0;

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let (r, g, b) = if h < 1.0 / 6.0 {
            (c, x, 0.0)
        } else if h < 2.0 / 6.0 {
            (x, c, 0.0)
        } else if h < 3.0 / 6.0 {
            (0.0, c, x)
        } else if h < 4.0 / 6.0 {
            (0.0, x, c)
        } else if h < 5.0 / 6.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        let r = ((r + m) * 255.0) as u8;
        let g = ((g + m) * 255.0) as u8;
        let b = ((b + m) * 255.0) as u8;

        format!("rgb({},{},{})", r, g, b)
    }
}

/// SVG Flame Graph Generator
#[derive(Debug)]
pub struct SvgFlameGraphGenerator {
    config: SvgFlameGraphConfig,
}

impl SvgFlameGraphGenerator {
    /// Create a new SVG flame graph generator
    pub fn new(config: SvgFlameGraphConfig) -> Self {
        Self { config }
    }

    /// Generate SVG flame graph from flame graph data
    pub fn generate_svg(&self, root: &FlameGraphNode) -> String {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<defs>
  <style><![CDATA[
    .func_g:hover {{ stroke:black; stroke-width:0.5; cursor:pointer; }}
    .func_text {{ font-family:{}; font-size:{}px; fill:rgb(0,0,0); }}
    .search_text {{ font-family:{}; font-size:{}px; fill:rgb(255,255,255); }}
  ]]></style>
</defs>
"#,
            self.config.width,
            self.config.height,
            self.config.font_family,
            self.config.font_size,
            self.config.font_family,
            self.config.font_size
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect x="0" y="0" width="{}" height="{}" fill="white"/>
"#,
            self.config.width, self.config.height
        ));

        // Title
        if !self.config.title.is_empty() {
            svg.push_str(&format!(
                r#"<text x="{}" y="24" class="func_text" style="font-size:20px; font-weight:bold; text-anchor:middle;">{}</text>
"#,
                self.config.width / 2,
                self.config.title
            ));
        }

        // Subtitle
        if !self.config.subtitle.is_empty() {
            svg.push_str(&format!(
                r#"<text x="{}" y="48" class="func_text" style="font-size:14px; text-anchor:middle;">{}</text>
"#,
                self.config.width / 2,
                self.config.subtitle
            ));
        }

        // Generate flame graph rectangles
        let total_time = root.total_time.as_nanos() as f64;
        let flame_height = 17.0;
        let start_y = if !self.config.title.is_empty() || !self.config.subtitle.is_empty() {
            60.0
        } else {
            10.0
        };

        self.generate_flames(
            root,
            &mut svg,
            0.0,
            self.config.width as f64,
            start_y,
            flame_height,
            total_time,
            0,
        );

        // Add JavaScript interactivity if enabled
        if self.config.interactive {
            svg.push_str(self.generate_javascript());
        }

        svg.push_str("</svg>");
        svg
    }

    /// Generate flame rectangles recursively
    fn generate_flames(
        &self,
        node: &FlameGraphNode,
        svg: &mut String,
        x: f64,
        width: f64,
        y: f64,
        height: f64,
        total_time: f64,
        _depth: usize,
    ) {
        if width < self.config.min_width {
            return;
        }

        let node_time = node.total_time.as_nanos() as f64;
        let heat = if total_time > 0.0 {
            node_time / total_time
        } else {
            0.0
        };
        let color = self.config.color_scheme.get_color(heat, &node.name);

        // Generate rectangle
        let escaped_name = self.escape_xml(&node.name);
        let percentage = if total_time > 0.0 {
            format!("{:.2}%", (node_time / total_time) * 100.0)
        } else {
            "0.00%".to_string()
        };

        svg.push_str(&format!(
            r#"<g class="func_g">
  <rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" rx="2" ry="2"/>
  <text x="{:.1}" y="{:.1}" class="func_text">{}</text>
  <title>{} ({})</title>
</g>
"#,
            x,
            y,
            width,
            height,
            color,
            x + 4.0,
            y + height * 0.7,
            self.truncate_text(&escaped_name, width),
            escaped_name,
            percentage
        ));

        // Generate children
        let mut child_x = x;
        for child in node.children.values() {
            let child_time = child.total_time.as_nanos() as f64;
            let child_width = if node_time > 0.0 {
                (child_time / node_time) * width
            } else {
                0.0
            };

            if child_width >= self.config.min_width {
                self.generate_flames(
                    child,
                    svg,
                    child_x,
                    child_width,
                    y + height + 1.0,
                    height,
                    total_time,
                    _depth + 1,
                );
            }

            child_x += child_width;
        }
    }

    /// Escape XML special characters
    fn escape_xml(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Truncate text to fit within width
    fn truncate_text(&self, text: &str, width: f64) -> String {
        let char_width = (self.config.font_size as f64) * 0.6; // Approximate character width
        let max_chars = ((width - 8.0) / char_width) as usize;

        if text.len() <= max_chars {
            text.to_string()
        } else if max_chars > 3 {
            format!("{}...", &text[..max_chars - 3])
        } else {
            "".to_string()
        }
    }

    /// Generate JavaScript for interactivity
    fn generate_javascript(&self) -> &'static str {
        r#"<script><![CDATA[
        // Search functionality
        function search(term) {
            const elements = document.querySelectorAll('.func_g');
            elements.forEach(el => {
                const text = el.querySelector('text').textContent;
                if (text.toLowerCase().includes(term.toLowerCase())) {
                    el.style.opacity = '1.0';
                } else {
                    el.style.opacity = '0.3';
                }
            });
        }
        
        // Reset search
        function resetSearch() {
            const elements = document.querySelectorAll('.func_g');
            elements.forEach(el => {
                el.style.opacity = '1.0';
            });
        }
        
        // Zoom functionality
        let currentZoom = 1.0;
        const svg = document.querySelector('svg');
        
        function zoom(factor) {
            currentZoom *= factor;
            svg.style.transform = `scale(${currentZoom})`;
        }
        
        // Mouse wheel zoom
        svg.addEventListener('wheel', function(e) {
            e.preventDefault();
            if (e.deltaY < 0) {
                zoom(1.1);
            } else {
                zoom(0.9);
            }
        });
        
        // Search on key press
        document.addEventListener('keydown', function(e) {
            if (e.key === 'r' || e.key === 'R') {
                resetSearch();
            } else if (e.key === 's' || e.key === 'S') {
                const term = prompt('Search for:');
                if (term) search(term);
            }
        });
        ]]></script>"#
    }

    /// Export SVG to file
    pub fn export_to_file(&self, root: &FlameGraphNode, path: &str) -> Result<(), std::io::Error> {
        let svg_content = self.generate_svg(root);

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(svg_content.as_bytes())?;
        writer.flush()?;

        Ok(())
    }
}

impl Default for SvgFlameGraphGenerator {
    fn default() -> Self {
        Self::new(SvgFlameGraphConfig::default())
    }
}

/// Enhanced flame graph with system resource correlation
#[derive(Debug)]
pub struct EnhancedFlameGraph {
    /// Performance flame graph
    pub performance: FlameGraphNode,
    /// Memory usage flame graph
    pub memory: Option<FlameGraphNode>,
    /// CPU usage data
    pub cpu_usage: Vec<(Duration, f64)>,
    /// Memory usage data
    pub memory_usage: Vec<(Duration, usize)>,
    /// Total profiling duration
    pub total_duration: Duration,
}

impl EnhancedFlameGraph {
    /// Create a new enhanced flame graph
    pub fn new(performance: FlameGraphNode) -> Self {
        Self {
            performance,
            memory: None,
            cpu_usage: Vec::new(),
            memory_usage: Vec::new(),
            total_duration: Duration::from_secs(0),
        }
    }

    /// Export as multi-panel SVG with system metrics
    pub fn export_enhanced_svg(&self, path: &str) -> Result<(), std::io::Error> {
        let config = SvgFlameGraphConfig {
            height: 800,
            title: "Enhanced Performance Profile".to_string(),
            ..Default::default()
        };

        let generator = SvgFlameGraphGenerator::new(config);
        let mut svg = String::new();

        // SVG header with increased height
        svg.push_str(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style><![CDATA[
    .func_g:hover { stroke:black; stroke-width:0.5; cursor:pointer; }
    .func_text { font-family:Verdana, sans-serif; font-size:12px; fill:rgb(0,0,0); }
    .chart_line { stroke:blue; stroke-width:2; fill:none; }
    .chart_area { fill:lightblue; opacity:0.3; }
  ]]></style>
</defs>
<rect x="0" y="0" width="1200" height="800" fill="white"/>
"#,
        );

        // Title
        svg.push_str(r#"<text x="600" y="24" class="func_text" style="font-size:18px; font-weight:bold; text-anchor:middle;">Enhanced Performance Profile</text>
"#);

        // Performance flame graph (top half)
        svg.push_str(r#"<text x="10" y="55" class="func_text" style="font-weight:bold;">CPU Performance Flame Graph</text>
"#);

        // Generate main flame graph
        let performance_svg = generator.generate_svg(&self.performance);
        let performance_content = self.extract_svg_content(&performance_svg);
        svg.push_str(&format!(
            r#"<g transform="translate(0, 70)">{}</g>
"#,
            performance_content
        ));

        // System metrics charts (bottom half)
        svg.push_str(r#"<text x="10" y="425" class="func_text" style="font-weight:bold;">System Resource Usage</text>
"#);

        // CPU usage chart
        if !self.cpu_usage.is_empty() {
            self.add_cpu_chart(&mut svg, 50.0, 450.0, 500.0, 100.0);
        }

        // Memory usage chart
        if !self.memory_usage.is_empty() {
            self.add_memory_chart(&mut svg, 600.0, 450.0, 500.0, 100.0);
        }

        svg.push_str("</svg>");

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(svg.as_bytes())?;
        writer.flush()?;

        Ok(())
    }

    /// Extract SVG content without header/footer
    fn extract_svg_content(&self, svg: &str) -> String {
        // Simple extraction - in a real implementation would use proper XML parsing
        if let Some(start) = svg.find("<rect x=\"0\" y=\"0\"") {
            if let Some(end) = svg.rfind("</svg>") {
                return svg[start..end].to_string();
            }
        }
        svg.to_string()
    }

    /// Add CPU usage chart to SVG
    fn add_cpu_chart(&self, svg: &mut String, x: f64, y: f64, width: f64, height: f64) {
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black"/>
<text x="{}" y="{}" class="func_text" style="font-size:10px;">CPU Usage (%)</text>
"#,
            x,
            y,
            width,
            height,
            x + 5.0,
            y - 5.0
        ));

        if self.cpu_usage.len() < 2 {
            return;
        }

        let max_cpu = self
            .cpu_usage
            .iter()
            .map(|(_, cpu)| *cpu)
            .fold(0.0, f64::max);
        let max_time = self.total_duration.as_secs_f64();

        let mut points = String::new();
        for (i, (time, cpu)) in self.cpu_usage.iter().enumerate() {
            let chart_x = x + (time.as_secs_f64() / max_time) * width;
            let chart_y = y + height - (cpu / max_cpu) * height;

            if i == 0 {
                points.push_str(&format!("M {:.1} {:.1}", chart_x, chart_y));
            } else {
                points.push_str(&format!(" L {:.1} {:.1}", chart_x, chart_y));
            }
        }

        svg.push_str(&format!(
            r#"<path d="{}" class="chart_line"/>
"#,
            points
        ));
    }

    /// Add memory usage chart to SVG
    fn add_memory_chart(&self, svg: &mut String, x: f64, y: f64, width: f64, height: f64) {
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black"/>
<text x="{}" y="{}" class="func_text" style="font-size:10px;">Memory Usage (MB)</text>
"#,
            x,
            y,
            width,
            height,
            x + 5.0,
            y - 5.0
        ));

        if self.memory_usage.len() < 2 {
            return;
        }

        let max_memory = self
            .memory_usage
            .iter()
            .map(|(_, mem)| *mem)
            .max()
            .unwrap_or(0);
        let max_time = self.total_duration.as_secs_f64();

        let mut points = String::new();
        for (i, (time, memory)) in self.memory_usage.iter().enumerate() {
            let chart_x = x + (time.as_secs_f64() / max_time) * width;
            let chart_y = y + height - ((*memory as f64) / (max_memory as f64)) * height;

            if i == 0 {
                points.push_str(&format!("M {:.1} {:.1}", chart_x, chart_y));
            } else {
                points.push_str(&format!(" L {:.1} {:.1}", chart_x, chart_y));
            }
        }

        svg.push_str(&format!(
            r#"<path d="{}" style="stroke:green; stroke-width:2; fill:none;"/>
"#,
            points
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_svg_generation() {
        let mut root = FlameGraphNode::new("main".to_string(), 0);
        root.add_sample(Duration::from_millis(100));

        let generator = SvgFlameGraphGenerator::default();
        let svg = generator.generate_svg(&root);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("main"));
    }

    #[test]
    fn test_color_schemes() {
        let schemes = [
            ColorScheme::Spectral,
            ColorScheme::Hot,
            ColorScheme::Cool,
            ColorScheme::Grayscale,
            ColorScheme::Java,
            ColorScheme::Memory,
        ];

        for scheme in &schemes {
            let color = scheme.get_color(0.5, "test_function");
            assert!(color.starts_with("rgb("));
            assert!(color.ends_with(")"));
        }
    }

    #[test]
    fn test_xml_escaping() {
        let generator = SvgFlameGraphGenerator::default();
        assert_eq!(
            generator.escape_xml("test<>&\"'"),
            "test&lt;&gt;&amp;&quot;&apos;"
        );
    }

    #[test]
    fn test_text_truncation() {
        let generator = SvgFlameGraphGenerator::default();
        let result = generator.truncate_text("very_long_function_name", 50.0);
        assert!(result.len() <= 50);
    }
}
