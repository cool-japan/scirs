//! Combinatorial optimization algorithms.
//!
//! This module provides solvers for classic combinatorial optimization problems:
//!
//! - **TSP** ([`tsp`]): Traveling Salesman Problem — nearest-neighbor heuristic,
//!   2-opt, 3-opt, Or-opt local search, MST lower bound, and a full solver.
//! - **Knapsack** ([`knapsack`]): 0/1 knapsack DP (exact), branch-and-bound (exact),
//!   greedy heuristic, fractional knapsack, and multi-dimensional knapsack.
//! - **Assignment** ([`assignment`]): Hungarian algorithm (Kuhn-Munkres) for
//!   minimum-cost assignment, and sparse minimum-cost bipartite matching.
//! - **Graph Coloring** ([`graph_coloring`]): Welsh-Powell greedy, DSATUR,
//!   exact backtracking, and chromatic number computation.
//! - **Covering** ([`covering`]): Set cover, weighted set cover, vertex cover
//!   (2-approximation), König's exact minimum vertex cover for bipartite graphs,
//!   and hitting set.

pub mod assignment;
pub mod covering;
pub mod graph_coloring;
pub mod knapsack;
pub mod tsp;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use assignment::{hungarian_algorithm, min_cost_matching, AssignmentResult};
pub use covering::{
    greedy_set_cover, hitting_set, min_vertex_cover_bip, vertex_cover_2approx, weighted_set_cover,
    CoveringResult,
};
pub use graph_coloring::{ColoringResult, GraphColoring};
pub use knapsack::{
    fractional_knapsack, knapsack_branch_bound, knapsack_dp, knapsack_greedy,
    multi_knapsack_greedy, KnapsackItem, KnapsackResult, MultiKnapsackItem,
};
pub use tsp::{
    mst_lower_bound, nearest_neighbor_heuristic, or_opt, three_opt_move, tour_length, two_opt,
    TspResult, TspSolver,
};
