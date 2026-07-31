//! Recurrent layer implementations

pub mod bidirectional;
pub mod gru;
pub mod lstm;
pub mod rnn;

// Re-export main types
pub use bidirectional::Bidirectional;
pub use gru::GRU;
pub use lstm::LSTM;
pub use rnn::RNN;

// Type aliases for compatibility
use scirs2_core::ndarray::{Array, IxDyn};
use std::sync::{Arc, RwLock};

/// Type alias for LSTM state cache (hidden, cell)
pub type LstmStateCache<F> = Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>)>>>;

/// Gate activations produced by a single LSTM time step
///
/// Tuple order is `(input_gate, forget_gate, cell_gate, output_gate)`, each of
/// shape `[batch_size, hidden_size]`.
pub type LstmGates<F> = (
    Array<F, IxDyn>,
    Array<F, IxDyn>,
    Array<F, IxDyn>,
    Array<F, IxDyn>,
);

/// Type alias for LSTM gate cache (input, forget, output, cell gates)
pub type LstmGateCache<F> = Arc<RwLock<Option<LstmGates<F>>>>;

/// Cache of the per-time-step LSTM gate activations recorded by `forward`
///
/// Backpropagation through time needs every gate activation of every step, so
/// the forward pass stores one [`LstmGates`] entry per time step.
pub type LstmGateSeqCache<F> = Arc<RwLock<Option<Vec<LstmGates<F>>>>>;

/// Type alias for GRU state cache
pub type GruStateCache<F> = Arc<RwLock<Option<Array<F, IxDyn>>>>;

/// Gate activations produced by a single GRU time step
///
/// Tuple order is `(reset_gate, update_gate, new_gate)`, each of shape
/// `[batch_size, hidden_size]`.
pub type GruGates<F> = (Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>);

/// Type alias for GRU gate cache (reset, update, new gates)
pub type GruGateCache<F> = Arc<RwLock<Option<GruGates<F>>>>;

/// Cache of the per-time-step GRU gate activations recorded by `forward`
pub type GruGateSeqCache<F> = Arc<RwLock<Option<Vec<GruGates<F>>>>>;

/// Type alias for RNN state cache
pub type RnnStateCache<F> = Arc<RwLock<Option<Array<F, IxDyn>>>>;

/// Type alias for LSTM step output (new_h, new_c, (input_gate, forget_gate, cell_gate, output_gate))
pub type LstmStepOutput<F> = (Array<F, IxDyn>, Array<F, IxDyn>, LstmGates<F>);

/// Type alias for GRU forward output (new_h, (reset_gate, update_gate, new_gate))
pub type GruForwardOutput<F> = (Array<F, IxDyn>, GruGates<F>);
