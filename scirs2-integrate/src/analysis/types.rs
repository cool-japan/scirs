//! Core data structures for dynamical systems analysis
//!
//! This module contains all the fundamental types used for bifurcation analysis,
//! stability assessment, and related dynamical systems analysis.

use crate::error::{IntegrateError, IntegrateResult, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::collections::HashMap;

/// Bifurcation point information
#[derive(Debug, Clone)]
pub struct BifurcationPoint {
    /// Parameter value at bifurcation
    pub parameter_value: f64,
    /// State at bifurcation
    pub state: Array1<f64>,
    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,
    /// Eigenvalues at bifurcation point
    pub eigenvalues: Vec<Complex64>,
}

/// Types of bifurcations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BifurcationType {
    /// Fold/saddle-node bifurcation
    Fold,
    /// Transcritical bifurcation
    Transcritical,
    /// Pitchfork bifurcation
    Pitchfork,
    /// Hopf bifurcation (birth of limit cycle)
    Hopf,
    /// Period-doubling bifurcation
    PeriodDoubling,
    /// Homoclinic bifurcation
    Homoclinic,
    /// Unknown/unclassified bifurcation
    Unknown,
    /// Cusp bifurcation (codimension-2)
    Cusp,
    /// Takens-Bogdanov bifurcation
    TakensBogdanov,
    /// Bautin bifurcation (generalized Hopf)
    Bautin,
    /// Zero-Hopf bifurcation
    ZeroHopf,
    /// Double-Hopf bifurcation
    DoubleHopf,
}

/// Stability assessment result
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Fixed points found
    pub fixed_points: Vec<FixedPoint>,
    /// Periodic orbits found
    pub periodic_orbits: Vec<PeriodicOrbit>,
    /// Lyapunov exponents
    pub lyapunov_exponents: Option<Array1<f64>>,
    /// Basin of attraction estimates
    pub basin_analysis: Option<BasinAnalysis>,
}

/// Fixed point information
#[derive(Debug, Clone)]
pub struct FixedPoint {
    /// Location of fixed point
    pub location: Array1<f64>,
    /// Stability type
    pub stability: StabilityType,
    /// Eigenvalues of linearization
    pub eigenvalues: Vec<Complex64>,
    /// Eigenvectors of linearization
    pub eigenvectors: Array2<Complex64>,
}

/// Periodic orbit information
#[derive(Debug, Clone)]
pub struct PeriodicOrbit {
    /// Representative point on orbit
    pub representative_point: Array1<f64>,
    /// Period of orbit
    pub period: f64,
    /// Stability type
    pub stability: StabilityType,
    /// Floquet multipliers
    pub floquet_multipliers: Vec<Complex64>,
}

/// Stability classification
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityType {
    /// Stable (attracting)
    Stable,
    /// Unstable (repelling)
    Unstable,
    /// Saddle (mixed stability)
    Saddle,
    /// Center (neutrally stable)
    Center,
    /// Degenerate (requires higher-order analysis)
    Degenerate,
    /// Spiral stable
    SpiralStable,
    /// Spiral unstable
    SpiralUnstable,
    /// Node stable
    NodeStable,
    /// Node unstable
    NodeUnstable,
    /// Marginally stable
    Marginally,
}

/// Basin of attraction analysis
#[derive(Debug, Clone)]
pub struct BasinAnalysis {
    /// Grid points analyzed
    pub grid_points: Array2<f64>,
    /// Attractor index for each grid point (-1 for divergent)
    pub attractor_indices: Array2<i32>,
    /// List of attractors found
    pub attractors: Vec<Array1<f64>>,
}

/// Two-parameter bifurcation analysis result
#[derive(Debug, Clone)]
pub struct TwoParameterBifurcationResult {
    /// Parameter grid
    pub parameter_grid: Array2<f64>,
    /// Stability classification at each grid point
    pub stability_map: Array2<f64>,
    /// Detected bifurcation curves
    pub bifurcation_curves: Vec<BifurcationCurve>,
    /// Parameter 1 range
    pub parameter_range_1: (f64, f64),
    /// Parameter 2 range
    pub parameter_range_2: (f64, f64),
}

/// Bifurcation curve in parameter space
#[derive(Debug, Clone)]
pub struct BifurcationCurve {
    /// Points on the curve
    pub points: Vec<(f64, f64)>,
    /// Type of bifurcation
    pub curve_type: BifurcationType,
}

/// Continuation result
#[derive(Debug, Clone)]
pub struct ContinuationResult {
    /// Solution branch
    pub solution_branch: Vec<Array1<f64>>,
    /// Parameter values along branch
    pub parameter_values: Vec<f64>,
    /// Whether continuation converged
    pub converged: bool,
    /// Final residual
    pub final_residual: f64,
}

/// Sensitivity analysis result
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResult {
    /// First-order sensitivities with respect to each parameter
    pub first_order_sensitivities: HashMap<String, Array1<f64>>,
    /// Parameter interaction effects (second-order)
    pub parameter_interactions: HashMap<(String, String), Array1<f64>>,
    /// Nominal parameter values
    pub nominal_parameters: HashMap<String, f64>,
    /// Nominal state
    pub nominal_state: Array1<f64>,
}

/// Normal form analysis result
#[derive(Debug, Clone)]
pub struct NormalFormResult {
    /// Coefficients of the normal form
    pub normal_form_coefficients: Array1<f64>,
    /// Transformation matrix to normal form coordinates
    pub transformation_matrix: Array2<f64>,
    /// Type of normal form
    pub normal_form_type: BifurcationType,
    /// Stability analysis description
    pub stability_analysis: String,
}
