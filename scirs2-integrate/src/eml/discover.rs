//! ODE discovery facade — wraps `scirs2_symbolic::regression::discover_ode` with
//! an ndarray-native `(t, y)` interface and integrate-crate error types.
//!
//! # Variable convention
//!
//! The symbolic features are `Var(0)`, `Var(1)`, ..., `Var(n_states - 1)` —
//! one per state dimension.  The time axis is consumed only to estimate `dt`
//! via uniform-sampling validation; it is **not** passed as a feature to the SR
//! engine (consistent with the underlying `discover_ode` SINDy convention).
//!
//! # Limitations (Phase 3 first cut)
//!
//! * Uniform `dt` only — rejects trajectories where the interval between any two
//!   consecutive samples deviates from the mean by more than 1e-9 * |dt|.
//! * Boundary samples are dropped (central-difference derivative estimation).
//! * No noise pre-filtering; callers should smooth noisy data before calling.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_symbolic::regression::{DiscoveredFormula, OdeConfig, SrConfig};
use std::fmt;

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for ODE discovery from trajectory data.
///
/// Maps to [`scirs2_symbolic::regression::OdeConfig`] + [`SrConfig`] internally.
///
/// | Field               | Maps to                    | Default |
/// |---------------------|----------------------------|---------|
/// | `max_complexity`    | `SrConfig::max_nodes`      | 12      |
/// | `population_size`   | `SrConfig::beam_width`     | 200     |
/// | `n_generations`     | `SrConfig::max_iter`       | 50      |
#[derive(Debug, Clone)]
pub struct OdeDiscoveryConfig {
    /// Maximum formula size (node count).  Larger values allow more expressive
    /// formulas at the cost of search time.
    pub max_complexity: usize,
    /// Beam width — number of best candidates retained per generation.
    pub population_size: usize,
    /// Number of search iterations.
    pub n_generations: usize,
    /// Number of top formulas to return per state dimension.
    pub top_n: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for OdeDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_complexity: 12,
            population_size: 200,
            n_generations: 50,
            top_n: 3,
            seed: 42,
        }
    }
}

impl OdeDiscoveryConfig {
    /// Convert to the symbolic crate's [`OdeConfig`] given a uniform `dt`.
    fn to_ode_config(&self, dt: f64) -> OdeConfig {
        let sr = SrConfig::default()
            .with_max_nodes(self.max_complexity)
            .with_beam_width(self.population_size)
            .with_max_iter(self.n_generations)
            .with_top_n(self.top_n)
            .with_seed(self.seed);
        OdeConfig::new(dt).with_sr_config(sr)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`discover_ode_from_trajectory`].
#[derive(Debug)]
pub enum OdeDiscoveryError {
    /// The trajectory has zero samples.
    EmptyTrajectory,
    /// The number of rows in `y` does not equal the length of `t`.
    DimensionMismatch {
        /// Length of the time vector.
        t_len: usize,
        /// Number of rows in the state matrix.
        y_rows: usize,
    },
    /// The time points are not uniformly spaced (required for central-
    /// difference derivative estimation).
    NonUniformSampling {
        /// Maximum deviation from uniform spacing found in the trajectory.
        max_dev: f64,
    },
    /// An error propagated from the underlying symbolic regression engine.
    SymbolicError(String),
}

impl fmt::Display for OdeDiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeDiscoveryError::EmptyTrajectory => {
                write!(f, "trajectory is empty — no samples to process")
            }
            OdeDiscoveryError::DimensionMismatch { t_len, y_rows } => write!(
                f,
                "dimension mismatch: t has {t_len} points but y has {y_rows} rows"
            ),
            OdeDiscoveryError::NonUniformSampling { max_dev } => write!(
                f,
                "non-uniform sampling detected (max interval deviation = {max_dev:.3e}); \
                 only uniform time grids are supported"
            ),
            OdeDiscoveryError::SymbolicError(msg) => {
                write!(f, "symbolic regression error: {msg}")
            }
        }
    }
}

impl std::error::Error for OdeDiscoveryError {}

// ---------------------------------------------------------------------------
// Main facade function
// ---------------------------------------------------------------------------

/// Discover ODE right-hand sides from trajectory data using symbolic regression.
///
/// Given a time series of state vectors `y` sampled at (approximately) uniform
/// time points `t`, this function:
///
/// 1. Validates shape consistency and uniform sampling.
/// 2. Estimates derivatives via central finite differences.
/// 3. Runs symbolic regression independently per state dimension to recover
///    candidate RHS expressions `dxⱼ/dt = fⱼ(x)`.
///
/// # Arguments
///
/// * `t` — shape `(n_samples,)`.  Must be approximately uniformly spaced.
/// * `y` — shape `(n_samples, n_states)`.  Row `i` is the state vector at `t[i]`.
/// * `config` — search configuration; see [`OdeDiscoveryConfig`].
///
/// # Returns
///
/// `Vec<Vec<DiscoveredFormula>>` — outer index is the state dimension `j`;
/// inner `Vec` holds the top-N candidate formulas for `dx_j/dt`, ranked by
/// combined fitness (best first).  Returns an empty outer `Vec` when
/// `n_samples < 3` (not enough points for central differences) or `n_states == 0`
/// (both handled by the underlying engine returning empty).
///
/// # Note on return type
///
/// The spec draft suggested returning `Vec<DiscoveredFormula>` (flat), but the
/// underlying `discover_ode` returns one ranked list *per state dimension*.
/// Flattening would destroy the per-dimension attribution, so we preserve the
/// `Vec<Vec<_>>` shape.  Callers that only care about the best formula per
/// dimension can call `result.iter().filter_map(|v| v.first()).collect()`.
pub fn discover_ode_from_trajectory(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    config: &OdeDiscoveryConfig,
) -> Result<Vec<Vec<DiscoveredFormula>>, OdeDiscoveryError> {
    let n_samples = t.len();
    let y_rows = y.shape()[0];

    // 1. Validate non-empty.
    if n_samples == 0 || y_rows == 0 {
        return Err(OdeDiscoveryError::EmptyTrajectory);
    }

    // 2. Validate shape consistency.
    if n_samples != y_rows {
        return Err(OdeDiscoveryError::DimensionMismatch {
            t_len: n_samples,
            y_rows,
        });
    }

    // 3. Compute mean dt and validate uniformity (requires >= 2 samples).
    let dt = if n_samples >= 2 {
        let total = t[n_samples - 1] - t[0];
        total / (n_samples - 1) as f64
    } else {
        // Single sample — central differences will return empty; pass dt = 1.0.
        1.0
    };

    if n_samples >= 3 && dt.abs() > 0.0 {
        let tol = 1e-9 * dt.abs();
        let mut max_dev = 0.0_f64;
        for i in 1..n_samples {
            let interval = t[i] - t[i - 1];
            let dev = (interval - dt).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
        if max_dev > tol {
            return Err(OdeDiscoveryError::NonUniformSampling { max_dev });
        }
    }

    // 4. Build OdeConfig and delegate to symbolic engine.
    let ode_cfg = config.to_ode_config(dt);
    let results = scirs2_symbolic::regression::discover_ode(y, &ode_cfg);

    // `discover_ode` returns empty on too-few samples or zero dimensions;
    // both are valid empty-success cases from the facade's perspective.
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    /// Helper: build a uniform time vector from 0 to (n-1)*dt.
    fn uniform_t(n: usize, dt: f64) -> Array1<f64> {
        Array1::from_vec((0..n).map(|i| i as f64 * dt).collect())
    }

    #[test]
    fn test_empty_t_returns_err() {
        let t = Array1::<f64>::zeros(0);
        let y = Array2::<f64>::zeros((0, 1));
        let cfg = OdeDiscoveryConfig::default();
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            matches!(result, Err(OdeDiscoveryError::EmptyTrajectory)),
            "expected EmptyTrajectory, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn test_dim_mismatch_returns_err() {
        let t = uniform_t(3, 0.1);
        // y has 4 rows but t has 3 entries
        let y = Array2::<f64>::zeros((4, 1));
        let cfg = OdeDiscoveryConfig::default();
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            matches!(
                result,
                Err(OdeDiscoveryError::DimensionMismatch {
                    t_len: 3,
                    y_rows: 4
                })
            ),
            "expected DimensionMismatch{{t_len:3,y_rows:4}}, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn test_discover_runs_without_panic() {
        // 5-point linear trajectory; use very small n_generations to keep test fast.
        let dt = 0.1;
        let t = uniform_t(5, dt);
        let traj_data: Vec<f64> = (0..5).map(|i| i as f64 * dt).collect();
        let y = Array2::from_shape_vec((5, 1), traj_data).expect("shape");
        let cfg = OdeDiscoveryConfig {
            n_generations: 3,
            population_size: 10,
            ..Default::default()
        };
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        // Either Ok (may be empty due to too-few interior samples) or a
        // SymbolicError — both are acceptable; panics are not.
        assert!(
            result.is_ok() || matches!(result, Err(OdeDiscoveryError::SymbolicError(_))),
            "unexpected error variant: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn test_single_state_trajectory() {
        // 10-point trajectory following x(t) = exp(-t).
        let dt = 0.1;
        let n = 10;
        let t = uniform_t(n, dt);
        let traj_data: Vec<f64> = (0..n).map(|i| (-(i as f64) * dt).exp()).collect();
        let y = Array2::from_shape_vec((n, 1), traj_data).expect("shape");
        let cfg = OdeDiscoveryConfig {
            n_generations: 5,
            population_size: 20,
            top_n: 2,
            ..Default::default()
        };
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            result.is_ok(),
            "expected Ok, got {:?}",
            result.err().map(|e| e.to_string())
        );
        if let Ok(formulas) = result {
            // When n >= 3, discover_ode returns one Vec per state dimension.
            assert!(
                formulas.len() <= 1,
                "single-state trajectory should produce at most 1 dimension"
            );
        }
    }
}
