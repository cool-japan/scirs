//! Enhanced visualization utilities for dendrograms
//!
//! This module provides comprehensive tools for visualizing hierarchical clustering results,
//! organized into focused submodules for better maintainability and clarity.
//!
//! # Modules
//!
//! - [`types`] - Core data structures and configuration types
//! - [`plotting`] - Dendrogram creation and positioning logic
//! - [`colors`] - Color palette generation and management
//! - [`export`] - Export functionality for various formats (SVG, HTML, JSON)
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_cluster::hierarchy::visualization::{
//!     create_dendrogramplot, DendrogramConfig, ColorScheme
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a linkage matrix (from hierarchical clustering)
//! let linkage = Array2::from_shape_vec((3, 4), vec![
//!     0.0, 1.0, 0.1, 2.0,
//!     2.0, 3.0, 0.2, 2.0,
//!     4.0, 5.0, 0.3, 4.0,
//! ]).unwrap();
//!
//! // Configure the visualization
//! let mut config = DendrogramConfig::default();
//! config.color_scheme = ColorScheme::Viridis;
//! config.show_labels = true;
//!
//! // Create the plot
//! let plot = create_dendrogramplot(
//!     linkage.view(),
//!     Some(&["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]),
//!     config
//! ).unwrap();
//!
//! // Export to various formats
//! let svg = plot.to_svg().unwrap();
//! let html = plot.to_html().unwrap();
//! let json = plot.to_json().unwrap();
//! ```

// Re-export modules
pub mod colors;
pub mod export;
pub mod plotting;
pub mod types;

// Re-export core types and functions for convenience
pub use colors::{get_color_palette, interpolate_colors, rgb_to_hex};
pub use export::{ExportConfig, ExportFormat};
pub use plotting::create_dendrogramplot;
pub use types::*;

// Additional convenience re-exports
pub use types::{
    BranchStyle, ColorScheme, ColorThreshold, DendrogramConfig, DendrogramOrientation,
    DendrogramPlot, DendrogramStyling, FontWeight, MarkerShape, TruncateMode,
};
