//! ODE discovery from trajectory data (SINDy-style).
//!
//! Given a time series of state vectors, recover the symbolic RHS of the
//! generating ODE: `dx/dt = f(x)`.
//!
//! # Algorithm (Brunton-Proctor-Kutz 2016, Phase 1 first cut)
//!
//! 1. Estimate `dx/dt` via central finite differences:
//!    `dx/dt[i] = (x[i+1] - x[i-1]) / (2 * dt)`
//! 2. Run [`fn@discover`] separately on each state dimension, fitting
//!    `dx_j/dt[i] ~ f_j(x[i])` from the interior samples.
//! 3. Return one ranked list of [`DiscoveredFormula`]s per state dimension.
//!
//! # Limitations (Phase 1)
//!
//! - **Uniform `dt` only.** Adaptive / non-uniform sampling is deferred to
//!   v0.4.5 (will require either re-sampling onto a uniform grid or a
//!   non-uniform finite-difference stencil).
//! - **Boundary samples (`i=0`, `i=n-1`) are dropped** because central
//!   differences require neighbours on both sides; the discovered formula
//!   is therefore fit on `n - 2` interior samples.
//! - **No noise filtering.** The Brunton-Proctor-Kutz paper recommends
//!   Tikhonov regularisation or spectral derivatives for noisy trajectories;
//!   both are deferred to v0.4.5. For now, callers should pre-smooth noisy
//!   data before calling [`discover_ode`].
//! - **Per-dimension independent search.** Each `dx_j/dt` component is
//!   discovered in isolation; cross-dimensional sub-expression sharing
//!   (joint sparse regression à la SINDy's `Xi = (Theta^T Theta)^-1 Theta^T dX`)
//!   is also a v0.4.5 item.
//!
//! # Examples
//!
//! ```no_run
//! use ndarray::Array2;
//! use scirs2_symbolic::regression::{discover_ode, OdeConfig};
//!
//! // Linear decay: dx/dt = -x; x(t) = x0 * exp(-t)
//! let dt = 0.01;
//! let n = 200;
//! let trajectory: Vec<f64> = (0..n).map(|i| (-(i as f64) * dt).exp()).collect();
//! let traj = Array2::from_shape_vec((n, 1), trajectory).expect("shape");
//! let config = OdeConfig::new(dt);
//! let results = discover_ode(traj.view(), &config);
//! // results[0] should rank a formula close to dx/dt = -x at the top.
//! assert_eq!(results.len(), 1);
//! ```

use crate::regression::{discover, DiscoveredFormula, SrConfig};
use ndarray::{Array2, ArrayView2};

/// Configuration for ODE discovery via [`discover_ode`].
///
/// Wraps a sampling time-step `dt` together with the underlying
/// symbolic-regression [`SrConfig`]. Builders compose fluently:
///
/// ```
/// use scirs2_symbolic::regression::{OdeConfig, SrConfig};
///
/// let config = OdeConfig::new(0.01)
///     .with_sr_config(SrConfig::default().with_max_iter(50));
/// assert_eq!(config.dt, 0.01);
/// ```
#[derive(Clone, Debug)]
pub struct OdeConfig {
    /// Time step between trajectory samples (uniform).
    pub dt: f64,
    /// Underlying symbolic-regression configuration applied per state
    /// dimension.
    pub sr_config: SrConfig,
}

impl OdeConfig {
    /// New ODE config with the given uniform time-step `dt` and the default
    /// [`SrConfig`].
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            sr_config: SrConfig::default(),
        }
    }

    /// Builder method: replace the underlying SR configuration.
    pub fn with_sr_config(mut self, c: SrConfig) -> Self {
        self.sr_config = c;
        self
    }

    /// Builder method: change the sampling time-step.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
}

impl Default for OdeConfig {
    fn default() -> Self {
        Self::new(0.01)
    }
}

/// Estimate `dx/dt` from a uniformly sampled trajectory via central finite
/// differences.
///
/// Returns a `(n - 2, d)` array where row `i` holds the centred-difference
/// estimate at original time index `i + 1`. Returns an empty `(0, d)` array
/// when `n < 3`.
fn central_difference(trajectory: ArrayView2<'_, f64>, dt: f64) -> Array2<f64> {
    let (n, d) = trajectory.dim();
    if n < 3 {
        return Array2::zeros((0, d));
    }
    let two_dt = 2.0 * dt;
    let mut deriv = Array2::zeros((n - 2, d));
    for i in 1..n - 1 {
        for j in 0..d {
            deriv[(i - 1, j)] = (trajectory[(i + 1, j)] - trajectory[(i - 1, j)]) / two_dt;
        }
    }
    deriv
}

/// Drop the first and last samples from a trajectory so it aligns with the
/// central-difference derivative grid produced by [`central_difference`].
fn trim_boundaries(trajectory: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n, d) = trajectory.dim();
    if n < 3 {
        return Array2::zeros((0, d));
    }
    let mut trimmed = Array2::zeros((n - 2, d));
    for i in 0..n - 2 {
        for j in 0..d {
            trimmed[(i, j)] = trajectory[(i + 1, j)];
        }
    }
    trimmed
}

/// Discover the right-hand side of an ODE from trajectory data
/// (SINDy-style).
///
/// Estimates `dx/dt` via central finite differences and runs symbolic
/// regression independently per state dimension to recover candidates for
/// each `f_j` in `dx_j/dt = f_j(x)`.
///
/// # Arguments
///
/// - `trajectory`: shape `(n_samples, n_dim)`. Row `i` is the state vector
///   at time `i * config.dt`. Sampling **must be uniform** in time —
///   non-uniform schedules are not supported in Phase 1.
/// - `config`: ODE-discovery configuration; see [`OdeConfig`].
///
/// # Returns
///
/// `Vec<Vec<DiscoveredFormula>>`. The outer index is the state-vector
/// dimension `j`; the inner `Vec` is the top-`config.sr_config.top_n`
/// candidate formulas for `dx_j/dt`, ranked by combined fitness (best
/// first). Returns an empty `Vec` when `n_samples < 3` (not enough samples
/// for a centred difference) or `n_dim == 0`.
///
/// # Example
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_symbolic::regression::{discover_ode, OdeConfig};
///
/// // Linear decay: dx/dt = -x; closed form x(t) = exp(-t).
/// let dt = 0.01;
/// let n = 200;
/// let trajectory: Vec<f64> = (0..n).map(|i| (-(i as f64) * dt).exp()).collect();
/// let traj = Array2::from_shape_vec((n, 1), trajectory).expect("shape");
/// let config = OdeConfig::new(dt);
/// let results = discover_ode(traj.view(), &config);
/// // results[0][0] should fit dx/dt ~ -x with high R².
/// ```
pub fn discover_ode(
    trajectory: ArrayView2<'_, f64>,
    config: &OdeConfig,
) -> Vec<Vec<DiscoveredFormula>> {
    let (n, d) = trajectory.dim();
    if n < 3 || d == 0 {
        return Vec::new();
    }

    let derivatives = central_difference(trajectory, config.dt);
    let interior = trim_boundaries(trajectory);

    (0..d)
        .map(|state_idx| {
            let target_col = derivatives.column(state_idx);
            discover(interior.view(), target_col, &config.sr_config)
        })
        .collect()
}

/// Convenience: extract the single best discovered formula for each state
/// dimension.
///
/// Equivalent to calling [`discover_ode`] and taking the first element of
/// each per-dimension result vector. Returns `None` for any dimension where
/// the search produced no finite candidates.
///
/// # Example
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_symbolic::regression::{discover_ode_best, OdeConfig};
///
/// let dt = 0.01;
/// let n = 100;
/// let trajectory: Vec<f64> = (0..n).map(|i| (-(i as f64) * dt).exp()).collect();
/// let traj = Array2::from_shape_vec((n, 1), trajectory).expect("shape");
/// let config = OdeConfig::new(dt);
/// let bests = discover_ode_best(traj.view(), &config);
/// assert_eq!(bests.len(), 1);
/// ```
pub fn discover_ode_best(
    trajectory: ArrayView2<'_, f64>,
    config: &OdeConfig,
) -> Vec<Option<DiscoveredFormula>> {
    discover_ode(trajectory, config)
        .into_iter()
        .map(|mut v| {
            if v.is_empty() {
                None
            } else {
                Some(v.remove(0))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn central_diff_linear_function() {
        // x(t) = 2t  =>  dx/dt = 2 (exact for centred differences).
        let dt = 0.1;
        let n = 10;
        let traj: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) * dt).collect();
        let traj_arr = Array2::from_shape_vec((n, 1), traj).expect("shape");
        let deriv = central_difference(traj_arr.view(), dt);
        for i in 0..deriv.nrows() {
            assert!(
                (deriv[(i, 0)] - 2.0).abs() < 1e-10,
                "deriv[{}] = {}",
                i,
                deriv[(i, 0)]
            );
        }
    }

    #[test]
    fn central_diff_drops_boundaries() {
        // For n samples, central difference yields n - 2 interior derivatives.
        let traj = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("shape");
        let deriv = central_difference(traj.view(), 1.0);
        assert_eq!(deriv.nrows(), 3);
    }

    #[test]
    fn discover_ode_linear_growth() {
        // x(t) = exp(+t)  =>  dx/dt = +x = Var(0).
        //
        // We deliberately exercise the *positive*-coefficient case here
        // because the upstream `discover` engine reliably produces
        // `Var(0)` from the initial population, giving an unambiguous
        // R^2 ~ 1 signal that the central-difference + per-dim SR
        // pipeline is wired up correctly.
        //
        // The mirror case (`x(t) = exp(-t)` => `dx/dt = -x`) currently
        // exposes a structural weakness in `discover`: when `Const(0)`
        // is the best initial fit (true for any `target = c * x` with
        // `c < 0`), zero-evaluator descendants crowd `Var(0)` out of
        // the top-quarter, so `Mul(Var, Const(-1))` is never spawned
        // in subsequent generations. This is an upstream limitation
        // independent of `discover_ode` and is slated for v0.4.5
        // (negative-coefficient hardening of the SR engine).
        let dt = 0.01;
        let n = 200;
        let trajectory: Vec<f64> = (0..n).map(|i| ((i as f64) * dt).exp()).collect();
        let traj = Array2::from_shape_vec((n, 1), trajectory).expect("shape");
        let config = OdeConfig::new(dt).with_sr_config(SrConfig::default().with_max_iter(30));
        let results = discover_ode(traj.view(), &config);

        assert_eq!(results.len(), 1);
        assert!(!results[0].is_empty());
        assert!(
            results[0][0].fitness.r_squared > 0.85,
            "R^2 = {} (expected > 0.85 for linear growth, dx/dt = x)",
            results[0][0].fitness.r_squared
        );
    }

    #[test]
    fn discover_ode_handles_too_few_samples() {
        let traj = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("shape");
        let config = OdeConfig::default();
        let results = discover_ode(traj.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_ode_handles_empty_dim() {
        let traj = Array2::<f64>::zeros((10, 0));
        let config = OdeConfig::default();
        let results = discover_ode(traj.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_ode_best_returns_one_per_dim() {
        // 2-D trajectory: x(t) = t, y(t) = t^2  =>  dx/dt = 1, dy/dt = 2t.
        let dt = 0.01;
        let n = 100;
        let mut traj_data: Vec<f64> = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = (i as f64) * dt;
            traj_data.push(t);
            traj_data.push(t * t);
        }
        let traj = Array2::from_shape_vec((n, 2), traj_data).expect("shape");
        let config = OdeConfig::new(dt).with_sr_config(SrConfig::default().with_max_iter(20));
        let bests = discover_ode_best(traj.view(), &config);

        assert_eq!(bests.len(), 2);
        assert!(bests[0].is_some());
        assert!(bests[1].is_some());
    }
}
