//! Extended centrality measures for graph analysis.
//!
//! This module extends the basic centrality measures in `measures.rs` with
//! advanced algorithms suitable for complex network analysis.
//!
//! ## Submodules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`extended`] | VoteRank, HITS, trust centrality, k-core decomposition, effective resistance |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::centrality::extended::{
//!     VoteRankCentrality, HITSCentrality, CoreDecomposition, EffectiveResistance, TrustCentrality,
//! };
//!
//! let mut adj = Array2::<f64>::zeros((5, 5));
//! for i in 0..4_usize {
//!     adj[[i, i+1]] = 1.0;
//!     adj[[i+1, i]] = 1.0;
//! }
//! adj[[0, 4]] = 1.0; adj[[4, 0]] = 1.0; // close the cycle
//!
//! // VoteRank: find top-2 spreaders
//! let spreaders = VoteRankCentrality::new(2).compute(&adj).unwrap();
//!
//! // HITS: hubs and authorities
//! let (hubs, auths) = HITSCentrality::new(50, 1e-8).compute(&adj).unwrap();
//!
//! // k-core decomposition
//! let core = CoreDecomposition::compute(&adj);
//! println!("Max core = {}", core.max_core);
//!
//! // Effective resistance
//! let er = EffectiveResistance::compute(&adj).unwrap();
//! println!("Kirchhoff index = {:.3}", er.kirchhoff_index());
//! ```

pub mod extended;

pub use extended::{
    CoreDecomposition, EffectiveResistance, HITSCentrality, TrustCentrality, TrustCentralityResult,
    VoteRankCentrality,
};

pub mod advanced_centrality;

pub use advanced_centrality::{
    communicability, harmonic_centrality, katz_centrality, percolation_centrality,
    subgraph_centrality,
};
