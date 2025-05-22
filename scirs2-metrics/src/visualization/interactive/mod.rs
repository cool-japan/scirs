//! Interactive visualization utilities
//!
//! This module provides tools for creating interactive visualizations that can
//! respond to user input and update dynamically.

pub mod roc_curve;

pub use roc_curve::{
    interactive_roc_curve_from_labels, interactive_roc_curve_visualization, InteractiveOptions,
    InteractiveROCVisualizer,
};
