//! Neural Forecasting Models for Time Series
//!
//! This module provides cutting-edge implementations for neural network-based
//! time series forecasting, including LSTM, GRU, Transformer, Mamba/State Space Models,
//! Temporal Fusion Transformers, and Mixture of Experts architectures.
//! These implementations focus on core algorithmic components and can be
//! extended with actual neural network frameworks.
//!
//! ## Advanced Architectures
//! - **LSTM Networks**: Long Short-Term Memory networks for sequence modeling
//! - **Transformer Models**: Self-attention based architectures
//! - **N-BEATS**: Neural basis expansion analysis for time series forecasting
//! - **Mamba/State Space Models**: Linear complexity for long sequences with selective state spaces
//! - **Flash Attention**: Memory-efficient attention computation for transformers
//! - **Temporal Fusion Transformers**: Specialized architecture for time series forecasting
//! - **Mixture of Experts**: Conditional computation for model scaling

// Re-export all module components for backward compatibility and ease of use
pub use self::{
    config::*,
    lstm::*,
    transformer::*,
    nbeats::*,
    mamba::*,
    attention::*,
    temporal_fusion::*,
    mixture_of_experts::*,
};

// Module declarations
pub mod config;
pub mod lstm;
pub mod transformer;
pub mod nbeats;
pub mod mamba;
pub mod attention;
pub mod temporal_fusion;
pub mod mixture_of_experts;