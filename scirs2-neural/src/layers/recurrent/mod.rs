//! Recurrent neural network layers implementations
//!
//! This module provides implementations of various recurrent neural network layers,
//! including basic RNN, LSTM, and GRU layers.

// Re-export all modules
pub mod bidirectional;
pub mod gru;
pub mod lstm;
pub mod rnn;

// Re-export all types for backward compatibility
pub use bidirectional::Bidirectional;
pub use gru::{GRUConfig, GRU};
pub use lstm::{LSTMConfig, LSTM};
pub use rnn::{RNNConfig, RecurrentActivation, RNN};

// Common type definitions used across recurrent layers
use ndarray::{Array, IxDyn};
use std::cell::RefCell;

/// Type alias for LSTM gate cache (input, forget, output, cell gates)
pub type LstmGateCache<F> = RefCell<
    Option<(
        Array<F, IxDyn>,
        Array<F, IxDyn>,
        Array<F, IxDyn>,
        Array<F, IxDyn>,
    )>,
>;

/// Type alias for LSTM forward step output (new hidden, new cell, gates)
pub type LstmStepOutput<F> = (
    Array<F, IxDyn>,
    Array<F, IxDyn>,
    (
        Array<F, IxDyn>,
        Array<F, IxDyn>,
        Array<F, IxDyn>,
        Array<F, IxDyn>,
    ),
);

/// Type alias for GRU gate cache (reset, update, new gates)
pub type GruGateCache<F> = RefCell<Option<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)>>;

/// Type alias for GRU forward output (output, gates)
pub type GruForwardOutput<F> = (
    Array<F, IxDyn>,
    (Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>),
);
