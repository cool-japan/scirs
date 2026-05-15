//! Clean `Vec<f64>`-based multi-objective optimisation algorithms.
//!
//! This module provides self-contained, easy-to-use implementations of
//! multi-objective evolutionary algorithms that work directly with Rust
//! `Vec<f64>` slices — no ndarray wrappers required.
//!
//! # Modules
//!
//! | Module | Algorithm |
//! |--------|-----------|
//! | [`mod@nsga2`]  | NSGA-II — Non-dominated Sorting Genetic Algorithm II (Deb 2002) |
//! | [`mod@nsga3`]  | NSGA-III — Many-objective NSGA with reference-point niching (Deb 2014) |
//! | [`mod@moead`]  | MOEA/D — Decomposition-based MOEA (Zhang 2007) |
//! | [`pareto`]     | Pareto dominance, ranking, hypervolume, and distance metrics |
//! | [`indicators`] | Quality indicators: HV, IGD, IGD+, GD, ε-indicator, R2, spacing |
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_optimize::multiobjective::nsga2::{nsga2, Nsga2Config};
//!
//! // 2-variable, 2-objective ZDT1 benchmark
//! let bounds = vec![(0.0_f64, 1.0_f64); 5];
//! let mut cfg = Nsga2Config::default();
//! cfg.population_size = 20;
//! cfg.n_generations   = 5;
//!
//! let result = nsga2(2, &bounds, |x| {
//!     let f1 = x[0];
//!     let g  = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len()-1) as f64;
//!     vec![f1, g * (1.0 - (f1 / g).sqrt())]
//! }, cfg).expect("valid input");
//!
//! println!("Pareto front size: {}", result.pareto_front.len());
//! ```
//!
//! # Many-objective (≥ 4 objectives) example
//!
//! ```rust
//! use scirs2_optimize::multiobjective::nsga3::{nsga3, Nsga3Config};
//!
//! let n_obj = 4;
//! let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_obj + 3];
//! let mut cfg = Nsga3Config::default();
//! cfg.population_size = 30;
//! cfg.n_generations   = 10;
//! cfg.n_divisions     = 3;
//!
//! let result = nsga3(n_obj, &bounds, |x| {
//!     let n = x.len();
//!     let k = n - n_obj + 1;
//!     let g: f64 = x[n-k..].iter().map(|&xi| (xi - 0.5).powi(2)).sum();
//!     let mut f = vec![0.0f64; n_obj];
//!     for i in 0..n_obj {
//!         let mut val = 1.0 + g;
//!         for j in 0..n_obj - 1 - i {
//!             val *= (x[j] * std::f64::consts::FRAC_PI_2).cos();
//!         }
//!         if i > 0 { val *= (x[n_obj - 1 - i] * std::f64::consts::FRAC_PI_2).sin(); }
//!         f[i] = val;
//!     }
//!     f
//! }, cfg).expect("valid input");
//!
//! println!("Reference points: {}", result.reference_points.len());
//! ```

pub mod hypervolume;
pub mod indicators;
pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod pareto;

// ── hypervolume module re-exports ────────────────────────────────────────────
pub use hypervolume::{
    exclusive_hypervolume, hypervolume_2d as hv_2d, hypervolume_3d, hypervolume_contribution_wfg,
    hypervolume_wfg,
};

// ── indicators module re-exports ──────────────────────────────────────────────
pub use indicators::{
    additive_epsilon_indicator, delta_metric, dominates, generational_distance, hypervolume_2d,
    hypervolume_contribution, hypervolume_mc, igd, igd_plus, non_dominated_sort, r2_indicator,
    spacing_metric, spread, R2Utility,
};

// ── MOEA/D re-exports ─────────────────────────────────────────────────────────
pub use moead::{
    build_neighborhood, generate_weight_vectors, moead, tchebycheff_scalarization, MoeadConfig,
    MoeadResult,
};

// ── NSGA-II re-exports ────────────────────────────────────────────────────────
pub use nsga2::{nsga2, Individual, Nsga2Config, Nsga2Result};

// ── NSGA-III re-exports ───────────────────────────────────────────────────────
pub use nsga3::{
    adapt_reference_points, associate_to_reference_points, generate_reference_points,
    generate_reference_points_inner, nsga3, reference_line_distance, Nsga3Config, Nsga3Result,
};

// ── pareto module re-exports ──────────────────────────────────────────────────
pub use pareto::{
    crowding_distance, dominates as pareto_dominates, epsilon_indicator,
    generational_distance as pareto_gd, hypervolume_2d as pareto_hv2d,
    hypervolume_indicator as pareto_hv, igd as pareto_igd, non_dominated_sort as pareto_nds,
    pareto_front, pareto_front_2d, pareto_rank, spread_metric,
};
