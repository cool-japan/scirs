//! Memory metrics reporting and visualization
//!
//! This module provides formatters and visualizers for memory metrics.

#[cfg(feature = "memory_metrics")]
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

use crate::memory::metrics::collector::MemoryReport;

/// Format bytes in human-readable format
#[allow(dead_code)]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Format duration in human-readable format
#[allow(dead_code)]
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();

    if total_secs < 60 {
        return format!("{total_secs}s");
    }

    let mins = total_secs / 60;
    let secs = total_secs % 60;

    if mins < 60 {
        return format!("{mins}m {secs}s");
    }

    let hours = mins / 60;
    let mins = mins % 60;

    format!("{hours}h {mins}m {secs}s")
}

impl MemoryReport {
    /// Format the report as a string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Memory Report (duration: {})\n",
            format_duration(Duration::from_secs(1))
        ));
        output.push_str(&format!(
            "Total Current Usage: {}\n",
            format_bytes(self.total_current_usage)
        ));
        output.push_str(&format!(
            "Total Peak Usage: {}\n",
            format_bytes(self.total_peak_usage)
        ));
        output.push_str(&format!(
            "Total Allocations: {}\n",
            self.total_allocation_count
        ));
        output.push_str(&format!(
            "Total Allocated Bytes: {}\n",
            format_bytes(self.total_allocated_bytes)
        ));

        output.push_str("\nComponent Statistics:\n");

        // Sort components by peak usage (descending)
        let mut components: Vec<_> = self.component_stats.iter().collect();
        components.sort_by(|a, b| b.1.peak_usage.cmp(&a.1.peak_usage));

        for (component, stats) in components {
            output.push_str(&format!("\n  {component}\n"));
            output.push_str(&format!(
                "    Current Usage: {}\n",
                format_bytes(stats.current_usage)
            ));
            output.push_str(&format!(
                "    Peak Usage: {}\n",
                format_bytes(stats.peak_usage)
            ));
            output.push_str(&format!(
                "    Allocation Count: {}\n",
                stats.allocation_count
            ));
            output.push_str(&format!(
                "    Total Allocated: {}\n",
                format_bytes(stats.total_allocated)
            ));
            output.push_str(&format!(
                "    Avg Allocation Size: {}\n",
                format_bytes(stats.avg_allocation_size as usize)
            ));

            // Calculate reuse ratio (if peak is non-zero)
            if stats.peak_usage > 0 {
                let reuse_ratio = stats.total_allocated as f64 / stats.peak_usage as f64;
                output.push_str(&format!("    Memory Reuse Ratio: {reuse_ratio:.2}\n"));
            }

            // Calculate allocation frequency
            if self.duration.as_secs_f64() > 0.0 {
                let alloc_per_sec = stats.allocation_count as f64 / self.duration.as_secs_f64();
                output.push_str(&format!("    Allocations/sec: {alloc_per_sec:.2}\n"));
            }
        }

        output
    }

    /// Export the report as JSON
    #[cfg(feature = "memory_metrics")]
    pub fn to_json(&self) -> JsonValue {
        let mut component_stats = serde_json::Map::new();

        for (component, stats) in &self.component_stats {
            let reuse_ratio = if stats.peak_usage > 0 {
                stats.total_allocated as f64 / stats.peak_usage as f64
            } else {
                0.0
            };

            let alloc_per_sec = if self.duration.as_secs_f64() > 0.0 {
                stats.allocation_count as f64 / self.duration.as_secs_f64()
            } else {
                0.0
            };

            let stats_obj = json!({
                "current_usage": stats.current_usage,
                "current_usage_formatted": format_bytes(stats.current_usage),
                "peak_usage": stats.peak_usage,
                "peak_usage_formatted": format_bytes(stats.peak_usage),
                "allocation_count": stats.allocation_count,
                "total_allocated": stats.total_allocated,
                "total_allocated_formatted": format_bytes(stats.total_allocated),
                "avg_allocation_size": stats.avg_allocation_size,
                "avg_allocation_size_formatted": format_bytes(stats.avg_allocation_size as usize),
                "reuse_ratio": reuse_ratio,
                "alloc_per_sec": alloc_per_sec,
            });

            component_stats.insert(component.clone(), stats_obj);
        }

        json!({
            "total_current_usage": self.total_current_usage,
            "total_current_usage_formatted": format_bytes(self.total_current_usage),
            "total_peak_usage": self.total_peak_usage,
            "total_peak_usage_formatted": format_bytes(self.total_peak_usage),
            "total_allocation_count": self.total_allocation_count,
            "total_allocated_bytes": self.total_allocated_bytes,
            "total_allocated_bytes_formatted": format_bytes(self.total_allocated_bytes),
            "duration_seconds": self.duration.as_secs_f64(),
            "duration_formatted": format_duration(self.duration),
            "components": component_stats,
        })
    }

    /// Export the report as JSON - stub when memory_metrics is disabled
    #[cfg(not(feature = "memory_metrics"))]
    pub fn to_json_2(&self) -> String {
        "{}".to_string()
    }

    /// Generate a visual chart (ASCII or SVG)
    #[cfg(feature = "memory_visualization")]
    pub fn generate_chart(&self, format: ChartFormat) -> String {
        match format {
            ChartFormat::Ascii => self.generate_ascii_chart(),
            ChartFormat::Svg => self.generate_svg_chart(),
        }
    }

    /// Generate an ASCII chart of memory usage by component
    #[cfg(feature = "memory_visualization")]
    fn generate_ascii_chart(&self) -> String {
        let mut output = String::new();

        output.push_str("Memory Usage by Component (ASCII Chart)\n\n");

        // Sort components by peak usage (descending)
        let mut components: Vec<_> = self.component_stats.iter().collect();
        components.sort_by(|a, b| b.1.peak_usage.cmp(&a.1.peak_usage));

        // Find maximum usage for scaling
        let max_usage = components
            .first()
            .map(|(_, stats)| stats.peak_usage)
            .unwrap_or(0);

        if max_usage == 0 {
            return "No memory usage data available.".to_string();
        }

        // Chart width in characters
        const CHART_WIDTH: usize = 50;

        for (component, stats) in components {
            let peak_width =
                (stats.peak_usage as f64 / max_usage as f64 * CHART_WIDTH as f64) as usize;
            let current_width =
                (stats.current_usage as f64 / max_usage as f64 * CHART_WIDTH as f64) as usize;

            let peak_bar = "#".repeat(peak_width);
            let current_bar = if current_width > 0 {
                "|".repeat(current_width)
            } else {
                String::new()
            };

            output.push_str(&format!(
                "{:<20} [{:<50}] {}\n",
                component,
                format!("{current_bar}{peak_bar}"),
                format_bytes(stats.peak_usage)
            ));
        }

        output.push_str("\nLegend: | = Current Usage, # = Peak Usage\n");

        output
    }

    /// Generate an SVG chart
    #[cfg(feature = "memory_visualization")]
    fn generate_svg_chart(&self) -> String {
        // In a real implementation, this would generate an SVG chart
        // For now, just return a placeholder
        "SVG chart generation not implemented".to_string()
    }
}

/// Chart format for visualization
#[cfg(feature = "memory_visualization")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartFormat {
    /// ASCII chart
    Ascii,
    /// SVG chart
    Svg,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::metrics::collector::ComponentMemoryStats;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3600 + 90)), "1h 1m 30s");
    }

    #[test]
    fn test_memory_report_format() {
        let mut component_stats = HashMap::new();

        component_stats.insert(
            "Component1".to_string(),
            ComponentMemoryStats {
                current_usage: 1024,
                peak_usage: 2048,
                allocation_count: 10,
                total_allocated: 4096,
                avg_allocation_size: 409.6,
            },
        );

        component_stats.insert(
            "Component2".to_string(),
            ComponentMemoryStats {
                current_usage: 512,
                peak_usage: 1024,
                allocation_count: 5,
                total_allocated: 2048,
                avg_allocation_size: 409.6,
            },
        );

        let report = MemoryReport {
            total_current_usage: 1536,
            total_peak_usage: 3072,
            total_allocation_count: 15,
            total_allocated_bytes: 6144,
            component_stats,
            duration: Duration::from_secs(60),
        };

        let formatted = report.format();

        // Basic checks on the formatted output
        assert!(formatted.contains("Total Current Usage: 1.50 KB"));
        assert!(formatted.contains("Total Peak Usage: 3.00 KB"));
        assert!(formatted.contains("Total Allocations: 15"));
        assert!(formatted.contains("Component1"));
        assert!(formatted.contains("Component2"));
    }

    #[test]
    fn test_memory_report_to_json() {
        let mut component_stats = HashMap::new();

        component_stats.insert(
            "Component1".to_string(),
            ComponentMemoryStats {
                current_usage: 1024,
                peak_usage: 2048,
                allocation_count: 10,
                total_allocated: 4096,
                avg_allocation_size: 409.6,
            },
        );

        let report = MemoryReport {
            total_current_usage: 1024,
            total_peak_usage: 2048,
            total_allocation_count: 10,
            total_allocated_bytes: 4096,
            component_stats,
            duration: Duration::from_secs(30),
        };

        #[cfg(feature = "memory_metrics")]
        {
            let json = report.to_json();
            assert_eq!(json["total_current_usage"], 1024);
            assert_eq!(json["total_peak_usage"], 2048);
            assert_eq!(json["total_allocation_count"], 10);
            assert_eq!(json["components"]["Component1"]["current_usage"], 1024);
        }

        #[cfg(not(feature = "memory_metrics"))]
        {
            // Just to avoid the unused variable warning
            let _ = report;
        }
    }
}
