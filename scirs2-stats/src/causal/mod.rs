//! Causal Inference Methods
//!
//! This module provides a comprehensive suite of causal inference estimators:
//!
//! ## Sub-modules
//!
//! | Module | Methods |
//! |--------|---------|
//! | [`instrumental_variables`] | 2SLS, LIML, Hausman test, weak-instrument diagnostics |
//! | [`difference_in_differences`] | DiD (TWFE), synthetic control, event study, staggered DiD |
//! | [`regression_discontinuity`] | Sharp RDD, fuzzy RDD, bandwidth selection, RD plots |
//! | [`propensity_score`] | Logistic PS model, IPW, nearest-neighbour / kernel matching |
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_stats::causal::instrumental_variables::{IVEstimator, WeakInstrumentTest};
//! use scirs2_stats::causal::propensity_score::{
//!     PropensityScoreModel, IPW, PSMatching, MatchingMethod,
//! };
//! ```
//!
//! ## References
//!
//! - Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics.
//! - Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods.
//! - Imbens, G.W. & Kalyanaraman, K. (2012). Optimal Bandwidth Choice for
//!   the Regression Discontinuity Estimator.
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity
//!   Score in Observational Studies for Causal Effects.

pub mod difference_in_differences;
pub mod instrumental_variables;
pub mod propensity_score;
pub mod regression_discontinuity;

// ---------------------------------------------------------------------------
// Re-exports — instrumental variables
// ---------------------------------------------------------------------------

pub use instrumental_variables::{
    HausmanResult, HausmanTest, IVEstimator, IVResult, WeakInstrumentResult, WeakInstrumentTest,
    LIML,
};

// ---------------------------------------------------------------------------
// Re-exports — difference-in-differences
// ---------------------------------------------------------------------------

pub use difference_in_differences::{
    AttGt, DiD, DiDResult, EventCoefficient, EventStudy, EventStudyResult, StaggeredDiD,
    StaggeredDiDResult, SyntheticControl,
};

// ---------------------------------------------------------------------------
// Re-exports — regression discontinuity
// ---------------------------------------------------------------------------

pub use regression_discontinuity::{
    BandwidthMethod, BandwidthSelector, FuzzyRDD, RDDPlot, RDDResult, RDD,
};

// ---------------------------------------------------------------------------
// Re-exports — propensity score
// ---------------------------------------------------------------------------

pub use propensity_score::{
    MatchingMethod, MatchingResult, OverlapCheck, OverlapResult, PSMatching, PSResult,
    PropensityScoreModel, TrimMethod, IPW,
};

/// Convenience function: fit a propensity score model and estimate ATE/ATT/ATC via IPW.
pub use propensity_score::ps_estimate;

// ---------------------------------------------------------------------------
// Structural Equation Models
// ---------------------------------------------------------------------------

pub mod sem;

pub use sem::{satisfies_backdoor, IdentificationResult, LinearEquation, SEMWithIntercepts, SEM};

// ---------------------------------------------------------------------------
// Linear SEM with ndarray interface
// ---------------------------------------------------------------------------

pub mod conditional_independence;
pub mod fci_algorithm;
pub mod hedge;
pub mod id_algorithm;
pub mod linear_sem;
pub mod pc_algorithm;
pub mod semi_markov_graph;
pub mod symbolic_prob;

pub use linear_sem::{LinearSEM, LinearSEMWithIntercepts};

// ---------------------------------------------------------------------------
// Causal graph types for constraint-based algorithms (PC, FCI, etc.)
// ---------------------------------------------------------------------------

/// Marks on the endpoint of an edge in a mixed graph.
///
/// Used by constraint-based causal discovery algorithms (PC, FCI) to represent
/// different types of edges in CPDAGs and PAGs:
///
/// - `Tail` — definite tail (non-ancestral mark, as in `→` tails, `—` tails).
/// - `Arrow` — definite arrowhead.
/// - `Circle` — unknown endpoint (used by FCI for partial ancestral graphs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeMark {
    /// Definite tail: non-arrowhead endpoint.
    Tail,
    /// Definite arrowhead.
    Arrow,
    /// Unknown endpoint mark (FCI/PAG only).
    Circle,
}

/// Mixed graph for causal discovery algorithms.
///
/// Each edge `(i, j)` is stored as a pair `(mark_at_i, mark_at_j)`, where
/// `mark_at_i` is the mark at node `i` (endpoint facing node `i`) and
/// `mark_at_j` is the mark at node `j`.
///
/// - Directed edge `i → j`: `(Tail, Arrow)` stored at entry `(i, j)`.
/// - Undirected edge `i — j`: `(Tail, Tail)`.
/// - Bidirected edge `i ↔ j`: `(Arrow, Arrow)`.
/// - Circle endpoint `i o→ j`: `(Circle, Arrow)`.
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Variable names.
    pub var_names: Vec<String>,
    /// Adjacency: `edges[i][j] = Some((mark_at_i_from_j, mark_at_j_from_i))`.
    /// If `edges[i][j].is_some()` then `edges[j][i].is_some()` as well.
    edges: Vec<Vec<Option<(EdgeMark, EdgeMark)>>>,
    /// Separation sets: `sep[i][j]` is the set that d-separates i and j.
    pub sep_sets: Vec<Vec<Option<Vec<usize>>>>,
}

impl CausalGraph {
    /// Create a new graph with the given variable names, initially fully connected.
    pub fn new(var_names: &[&str]) -> Self {
        let p = var_names.len();
        Self {
            var_names: var_names.iter().map(|s| s.to_string()).collect(),
            edges: vec![vec![None; p]; p],
            sep_sets: vec![vec![None; p]; p],
        }
    }

    /// Number of variables (nodes).
    pub fn num_vars(&self) -> usize {
        self.var_names.len()
    }

    /// Set or update an edge between `i` and `j`.
    ///
    /// `mark_at_i` is the endpoint mark at node `i`; `mark_at_j` is the mark at `j`.
    /// Setting an edge is symmetric: `edges[i][j]` and `edges[j][i]` are both updated.
    pub fn set_edge(&mut self, i: usize, j: usize, mark_at_i: EdgeMark, mark_at_j: EdgeMark) {
        self.edges[i][j] = Some((mark_at_i, mark_at_j));
        self.edges[j][i] = Some((mark_at_j, mark_at_i));
    }

    /// Remove an edge between `i` and `j`.
    pub fn remove_edge(&mut self, i: usize, j: usize) {
        self.edges[i][j] = None;
        self.edges[j][i] = None;
    }

    /// Whether there is any edge between `i` and `j`.
    pub fn is_adjacent(&self, i: usize, j: usize) -> bool {
        self.edges[i][j].is_some()
    }

    /// Get the mark at node `to` on the edge from `from` to `to`.
    ///
    /// Returns `None` if the edge doesn't exist.
    /// The returned mark is the one facing node `to` (i.e., the arrowhead/tail at `to`).
    pub fn get_mark_at(&self, from: usize, to: usize) -> Option<EdgeMark> {
        self.edges[from][to].map(|(_, mark_at_to)| mark_at_to)
    }

    /// Get the mark at node `from` on the edge between `from` and `to`.
    ///
    /// Returns `None` if the edge doesn't exist.
    pub fn get_mark_from(&self, from: usize, to: usize) -> Option<EdgeMark> {
        self.edges[from][to].map(|(mark_at_from, _)| mark_at_from)
    }

    /// Whether there is a directed edge `i → j` (tail at `i`, arrow at `j`).
    pub fn is_directed(&self, i: usize, j: usize) -> bool {
        matches!(self.edges[i][j], Some((EdgeMark::Tail, EdgeMark::Arrow)))
    }

    /// Whether there is an undirected edge `i — j` (tail at both ends).
    pub fn is_undirected(&self, i: usize, j: usize) -> bool {
        matches!(self.edges[i][j], Some((EdgeMark::Tail, EdgeMark::Tail)))
    }

    /// Whether there is a bidirected edge `i ↔ j` (arrow at both ends).
    pub fn is_bidirected(&self, i: usize, j: usize) -> bool {
        matches!(self.edges[i][j], Some((EdgeMark::Arrow, EdgeMark::Arrow)))
    }

    /// Return an iterator over the neighbours of node `i`.
    pub fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_vars()).filter(move |&j| j != i && self.is_adjacent(i, j))
    }

    /// Return the separation set for nodes `i` and `j`, if any.
    pub fn get_sep_set(&self, i: usize, j: usize) -> Option<&Vec<usize>> {
        self.sep_sets[i][j].as_ref()
    }

    /// Set the separation set for nodes `i` and `j`.
    pub fn set_sep_set(&mut self, i: usize, j: usize, sep: Vec<usize>) {
        self.sep_sets[i][j] = Some(sep.clone());
        self.sep_sets[j][i] = Some(sep);
    }

    /// Initialize the graph as a complete undirected graph (all edges `i — j`).
    pub fn make_complete(&mut self) {
        let p = self.num_vars();
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    self.edges[i][j] = Some((EdgeMark::Tail, EdgeMark::Tail));
                }
            }
        }
    }
}
