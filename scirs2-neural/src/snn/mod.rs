//! Spiking Neural Networks (SNN)
//!
//! This module provides biologically plausible spiking neuron models,
//! synaptic dynamics, learning rules, and full SNN simulation infrastructure.
//!
//! ## Modules
//!
//! | Module              | Contents |
//! |---------------------|----------|
//! | [`neuron_models`]   | LIF, AdEx, Izhikevich, Hodgkin-Huxley neurons |
//! | [`synapse`]         | Exponential, Alpha, STDP synapses; delay buffers |
//! | [`stdp`]            | STDP, Triplet STDP, BCM, Oja learning rules |
//! | [`snn_layer`]       | `SpikingLayer`, `SpikingNetwork` |
//! | [`population_coding`] | Spike encoding / decoding schemes |
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_neural::snn::{
//!     neuron_models::{LIFConfig, LIFNeuron},
//!     snn_layer::SpikingNetwork,
//! };
//!
//! let config = LIFConfig::default();
//! let mut net = SpikingNetwork::new(&[4, 8, 2], &config, 10.0, 0.1).expect("operation should succeed");
//!
//! // Build input: 50 time steps, 4 input channels alternating
//! let input: Vec<Vec<bool>> = (0..50)
//!     .map(|t| (0..4).map(|i| (t + i) % 3 == 0).collect())
//!     .collect();
//!
//! let record = net.simulate(&input, 50).expect("operation should succeed");
//! println!("Total spikes: {}", scirs2_neural::snn::snn_layer::SpikingNetwork::count_spikes(&record));
//! ```

pub mod neuron_models;
pub mod population_coding;
pub mod snn_layer;
pub mod stdp;
pub mod synapse;

// Re-exports for convenience
pub use neuron_models::{
    AdExNeuron, HodgkinHuxleyNeuron, IzhikevichNeuron, IzhikevichPattern, LIFConfig, LIFNeuron,
};
pub use population_coding::{
    inter_spike_intervals, isi_cv, SpikeDecoder, SpikeEncoder, SpikeEncoding,
};
pub use snn_layer::{SpikingLayer, SpikingNetwork};
pub use stdp::{BCMRule, OjaRule, TripletSTDP, TripletState, STDP};
pub use synapse::{AlphaSynapse, ExponentialSynapse, STDPSynapse, SpikeBoolDelay, SynapticDelay};
