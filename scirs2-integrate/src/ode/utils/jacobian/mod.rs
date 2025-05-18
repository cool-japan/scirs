//! Jacobian approximation and handling utilities for ODE solvers
//!
//! This module provides tools for computing, updating, and reusing Jacobian matrices
//! in implicit ODE solvers. It implements various approximation strategies, reuse logic,
//! and specialized techniques for different problem types.

mod autodiff;
mod newton;
mod parallel;
mod specialized;

pub use autodiff::*;
pub use newton::*;
pub use parallel::*;
pub use specialized::*;

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use ndarray::{Array1, Array2, ArrayView1};

/// Strategy for Jacobian approximation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JacobianStrategy {
    /// Standard finite difference approximation
    FiniteDifference,
    /// Sparse finite difference (more efficient for sparse Jacobians)
    SparseFiniteDifference,
    /// Broyden's update (quasi-Newton)
    BroydenUpdate,
    /// Modified Newton (reuse Jacobian for multiple steps)
    ModifiedNewton,
    /// Colored finite difference (for systems with structure)
    ColoredFiniteDifference,
    /// Parallel finite difference (for large systems)
    ParallelFiniteDifference,
    /// Parallel sparse finite difference (for large sparse systems)
    ParallelSparseFiniteDifference,
    /// Automatic differentiation (exact derivatives)
    AutoDiff,
    /// Adaptive selection (uses autodiff if available, falls back to finite difference)
    #[default]
    Adaptive,
}

/// Compute Jacobian using finite differences (for compatibility)
pub fn compute_jacobian<F, Func>(f: &Func, t: F, y: &Array1<F>) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, &ArrayView1<F>) -> IntegrateResult<Array1<F>>,
{
    let f_current = f(t, &y.view())?;
    // Convert the function to the right signature for finite_difference_jacobian
    let f_unwrapped = |t: F, y: ArrayView1<F>| -> Array1<F> {
        match f(t, &y) {
            Ok(val) => val,
            Err(_) => Array1::zeros(y.len()), // Fallback to zeros on error
        }
    };
    Ok(crate::ode::utils::common::finite_difference_jacobian(
        &f_unwrapped,
        t,
        y,
        &f_current,
        F::from_f64(1e-8).unwrap(),
    ))
}

// Re-export finite_difference_jacobian for direct use
pub use crate::ode::utils::common::finite_difference_jacobian;

/// Structure of the Jacobian matrix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JacobianStructure {
    /// Dense matrix (default)
    #[default]
    Dense,
    /// Banded matrix (efficient for certain types of ODEs)
    Banded { lower: usize, upper: usize },
    /// Sparse matrix (large systems with few nonzeros)
    Sparse,
    /// Structured matrix (e.g., Toeplitz, circulant)
    Structured,
}

/// Manages Jacobian computation, updates, and reuse
#[derive(Debug, Clone)]
pub struct JacobianManager<F: IntegrateFloat> {
    /// The current Jacobian approximation
    jacobian: Option<Array2<F>>,
    /// The system state at which the Jacobian was computed
    state_point: Option<(F, Array1<F>)>, // (t, y)
    /// The function evaluation at the state point
    f_eval: Option<Array1<F>>,
    /// Number of steps since last full Jacobian computation
    age: usize,
    /// Maximum age before recomputation
    max_age: usize,
    /// Approximation strategy
    strategy: JacobianStrategy,
    /// Structure information
    #[allow(dead_code)]
    structure: JacobianStructure,
    /// Condition number estimate (if available)
    #[allow(dead_code)]
    condition_estimate: Option<F>,
    /// Factorized form (if available)
    factorized: bool,
}

impl<F: IntegrateFloat> Default for JacobianManager<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> JacobianManager<F> {
    /// Create a new Jacobian manager with default settings
    pub fn new() -> Self {
        JacobianManager {
            jacobian: None,
            state_point: None,
            f_eval: None,
            age: 0,
            max_age: 50, // Recompute after 50 steps by default
            strategy: JacobianStrategy::default(),
            structure: JacobianStructure::default(),
            condition_estimate: None,
            factorized: false,
        }
    }

    /// Create a new Jacobian manager with specific strategy and structure
    pub fn with_strategy(strategy: JacobianStrategy, structure: JacobianStructure) -> Self {
        JacobianManager {
            jacobian: None,
            state_point: None,
            f_eval: None,
            age: 0,
            max_age: match strategy {
                JacobianStrategy::ModifiedNewton => 50,
                JacobianStrategy::BroydenUpdate => 20,
                JacobianStrategy::ParallelFiniteDifference => 1,
                JacobianStrategy::ParallelSparseFiniteDifference => 1,
                _ => 1, // Other strategies don't reuse as much
            },
            strategy,
            structure,
            condition_estimate: None,
            factorized: false,
        }
    }

    /// Create a Jacobian manager with automatically selected strategy
    /// based on system size and structure
    pub fn with_auto_strategy(n_dim: usize, is_banded: bool) -> Self {
        let (strategy, structure) = if n_dim > 100 {
            // For very large systems, use parallel computation
            let parallel_available = cfg!(feature = "parallel");

            if parallel_available {
                if is_banded {
                    (
                        JacobianStrategy::ParallelSparseFiniteDifference,
                        JacobianStructure::Banded {
                            lower: n_dim / 10,
                            upper: n_dim / 10,
                        },
                    )
                } else {
                    (
                        JacobianStrategy::ParallelFiniteDifference,
                        JacobianStructure::Dense,
                    )
                }
            } else {
                // Fall back to modified Newton for large systems without parallel
                (JacobianStrategy::ModifiedNewton, JacobianStructure::Dense)
            }
        } else if n_dim > 20 {
            // For medium-sized systems, use Broyden updates
            (JacobianStrategy::BroydenUpdate, JacobianStructure::Dense)
        } else {
            // For small systems, try autodiff if available
            let autodiff_available = is_autodiff_available();

            if autodiff_available {
                (JacobianStrategy::AutoDiff, JacobianStructure::Dense)
            } else {
                (JacobianStrategy::FiniteDifference, JacobianStructure::Dense)
            }
        };

        Self::with_strategy(strategy, structure)
    }

    /// Check if the Jacobian needs to be recomputed
    pub fn needs_update(&self, t: F, y: &Array1<F>, force_age: Option<usize>) -> bool {
        // Always recompute if we don't have a Jacobian yet
        if self.jacobian.is_none() {
            return true;
        }

        // Check age against threshold (possibly overridden)
        let max_age = force_age.unwrap_or(self.max_age);
        if self.age >= max_age {
            return true;
        }

        // For certain strategies, we always recompute
        match self.strategy {
            JacobianStrategy::FiniteDifference => self.age > 0,
            _ => {
                // For other strategies, check if state has changed significantly
                if let Some((old_t, old_y)) = &self.state_point {
                    // Calculate relative distance between current and previous state
                    let t_diff = (t - *old_t).abs();
                    let mut y_diff = F::zero();
                    for i in 0..y.len() {
                        let rel_diff = if old_y[i].abs() > F::from_f64(1e-10).unwrap() {
                            (y[i] - old_y[i]).abs() / old_y[i].abs()
                        } else {
                            (y[i] - old_y[i]).abs()
                        };
                        y_diff = y_diff.max(rel_diff);
                    }

                    // Determine thresholds based on strategy
                    let (t_threshold, y_threshold) = match self.strategy {
                        JacobianStrategy::ModifiedNewton => (
                            F::from_f64(0.3).unwrap(), // 30% change in time
                            F::from_f64(0.3).unwrap(), // 30% change in y
                        ),
                        JacobianStrategy::BroydenUpdate => (
                            F::from_f64(0.1).unwrap(), // 10% change in time
                            F::from_f64(0.1).unwrap(), // 10% change in y
                        ),
                        _ => (
                            F::from_f64(0.01).unwrap(), // 1% change in time
                            F::from_f64(0.01).unwrap(), // 1% change in y
                        ),
                    };

                    // Check if state has changed enough to warrant recomputation
                    t_diff > t_threshold || y_diff > y_threshold
                } else {
                    // No previous state, so recompute
                    true
                }
            }
        }
    }

    /// Compute or update the Jacobian matrix
    pub fn update_jacobian<Func>(
        &mut self,
        t: F,
        y: &Array1<F>,
        f: &Func,
        scale: Option<F>,
    ) -> IntegrateResult<&Array2<F>>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
    {
        let scale_val = scale.unwrap_or_else(|| F::from_f64(1.0).unwrap());
        let n = y.len();

        match self.strategy {
            JacobianStrategy::FiniteDifference | JacobianStrategy::SparseFiniteDifference => {
                // Compute function at current point (if not available)
                let f_current = if let Some(f_val) = &self.f_eval {
                    f_val.clone()
                } else {
                    f(t, y.view())
                };

                // Create or resize Jacobian if needed
                let mut jac = if let Some(j) = &self.jacobian {
                    if j.shape() == [n, n] {
                        j.clone()
                    } else {
                        Array2::zeros((n, n))
                    }
                } else {
                    Array2::zeros((n, n))
                };

                // Perturbation size, scaled by variable magnitude
                let base_eps = F::from_f64(1e-8).unwrap();

                if self.strategy == JacobianStrategy::SparseFiniteDifference {
                    // For sparse problems, use specialized algorithm (future work)
                    // This would involve using graph coloring to reduce evaluations
                    // For now, fall back to standard finite difference
                    self.compute_dense_finite_difference(t, y, &f_current, f, &mut jac, base_eps);
                } else {
                    // Standard finite difference for each column
                    self.compute_dense_finite_difference(t, y, &f_current, f, &mut jac, base_eps);
                }

                // Apply scaling if needed (e.g., for implicit integration methods)
                if scale_val != F::one() {
                    for i in 0..n {
                        for j in 0..n {
                            if i == j {
                                jac[[i, j]] = F::one() - scale_val * jac[[i, j]];
                            } else {
                                jac[[i, j]] = -scale_val * jac[[i, j]];
                            }
                        }
                    }
                }

                // Update state information
                self.jacobian = Some(jac);
                self.state_point = Some((t, y.clone()));
                self.f_eval = Some(f_current);
                self.age = 0;
                self.factorized = false;

                // Return reference to the jacobian
                Ok(self.jacobian.as_ref().unwrap())
            }
            JacobianStrategy::ParallelFiniteDifference
            | JacobianStrategy::ParallelSparseFiniteDifference => {
                // Compute function at current point (if not available)
                let f_current = if let Some(f_val) = &self.f_eval {
                    f_val.clone()
                } else {
                    f(t, y.view())
                };

                // Use parallel implementation based on strategy
                let jac = if self.strategy == JacobianStrategy::ParallelSparseFiniteDifference {
                    // For parallel sparse computation, need sparsity pattern
                    // Default to dense pattern if none provided
                    let _sparsity_pattern = Array2::<bool>::from_elem((n, n), true);

                    // Check if F has Send + Sync bounds required for parallel computation
                    #[cfg(feature = "parallel")]
                    {
                        // Use parallel sparse Jacobian computation
                        parallel_sparse_jacobian(
                            &f,
                            t,
                            y,
                            &f_current,
                            Some(&sparsity_pattern),
                            F::one(),
                        )?
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        // Fallback to non-parallel version
                        finite_difference_jacobian(&f, t, y, &f_current, F::from_f64(1e-8).unwrap())
                    }
                } else {
                    // Use parallel dense Jacobian computation
                    #[cfg(feature = "parallel")]
                    {
                        parallel_finite_difference_jacobian(&f, t, y, &f_current, F::one())?
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        // Fallback to non-parallel version
                        finite_difference_jacobian(&f, t, y, &f_current, F::from_f64(1e-8).unwrap())
                    }
                };

                // Apply scaling if needed
                let scaled_jac = if scale_val != F::one() {
                    let mut scaled = Array2::<F>::zeros((n, n));
                    for i in 0..n {
                        for j in 0..n {
                            if i == j {
                                scaled[[i, j]] = F::one() - scale_val * jac[[i, j]];
                            } else {
                                scaled[[i, j]] = -scale_val * jac[[i, j]];
                            }
                        }
                    }
                    scaled
                } else {
                    jac
                };

                // Update state information
                self.jacobian = Some(scaled_jac);
                self.state_point = Some((t, y.clone()));
                self.f_eval = Some(f_current);
                self.age = 0;
                self.factorized = false;

                // Return reference to the jacobian
                Ok(self.jacobian.as_ref().unwrap())
            }
            JacobianStrategy::BroydenUpdate => {
                // Check if we need to do a full recomputation
                if self.jacobian.is_none() || self.age >= self.max_age || self.state_point.is_none()
                {
                    // Do a full computation using finite differences
                    return self.update_jacobian_with_strategy(
                        t,
                        y,
                        f,
                        scale,
                        JacobianStrategy::FiniteDifference,
                    );
                }

                // Perform a Broyden update to the existing Jacobian
                let (_old_t, old_y) = self.state_point.as_ref().unwrap();
                let old_f = self.f_eval.as_ref().unwrap();

                // Calculate new function value
                let new_f = f(t, y.view());

                // Calculate differences
                let delta_y = y - old_y;
                let delta_f = &new_f - old_f;

                // Get existing Jacobian
                let mut jac = self.jacobian.as_ref().unwrap().clone();

                // Broyden's update formula: J_new = J_old + (df - J_old * dy) * dy^T / (dy^T * dy)
                let mut jac_dy = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        jac_dy[i] += jac[[i, j]] * delta_y[j];
                    }
                }

                let dy_norm_squared: F = delta_y.iter().map(|&x| x * x).sum();
                if dy_norm_squared > F::from_f64(1e-14).unwrap() {
                    for i in 0..n {
                        for j in 0..n {
                            jac[[i, j]] += (delta_f[i] - jac_dy[i]) * delta_y[j] / dy_norm_squared;
                        }
                    }
                }

                // Apply scaling if needed
                if scale_val != F::one() {
                    for i in 0..n {
                        for j in 0..n {
                            if i == j {
                                jac[[i, j]] = F::one() - scale_val * jac[[i, j]];
                            } else {
                                jac[[i, j]] = -scale_val * jac[[i, j]];
                            }
                        }
                    }
                }

                // Update state information
                self.jacobian = Some(jac);
                self.state_point = Some((t, y.clone()));
                self.f_eval = Some(new_f);
                self.age += 1;
                self.factorized = false;

                Ok(self.jacobian.as_ref().unwrap())
            }
            JacobianStrategy::ModifiedNewton => {
                // Check if we need to do a full recomputation
                if self.jacobian.is_none() || self.age >= self.max_age {
                    // Compute a new Jacobian using finite differences
                    return self.update_jacobian_with_strategy(
                        t,
                        y,
                        f,
                        scale,
                        JacobianStrategy::FiniteDifference,
                    );
                }

                // Otherwise just reuse the existing Jacobian
                self.age += 1;

                // Update function evaluation at current point
                let f_current = f(t, y.view());
                self.f_eval = Some(f_current);

                Ok(self.jacobian.as_ref().unwrap())
            }
            JacobianStrategy::ColoredFiniteDifference => {
                // This is for future implementation - coloring would reduce the number of
                // function evaluations needed for systems with structure
                // For now, fall back to standard finite difference
                self.update_jacobian_with_strategy(
                    t,
                    y,
                    f,
                    scale,
                    JacobianStrategy::FiniteDifference,
                )
            }
            JacobianStrategy::AutoDiff => {
                // Compute function at current point (if not available)
                let f_current = if let Some(f_val) = &self.f_eval {
                    f_val.clone()
                } else {
                    f(t, y.view())
                };

                // Use autodiff to compute exact Jacobian
                let jac = autodiff_jacobian(f, t, y, &f_current, F::one())?;

                // Apply scaling if needed
                let scaled_jac = if scale_val != F::one() {
                    let mut scaled = Array2::<F>::zeros((n, n));
                    for i in 0..n {
                        for j in 0..n {
                            if i == j {
                                scaled[[i, j]] = F::one() - scale_val * jac[[i, j]];
                            } else {
                                scaled[[i, j]] = -scale_val * jac[[i, j]];
                            }
                        }
                    }
                    scaled
                } else {
                    jac
                };

                // Update state information
                self.jacobian = Some(scaled_jac);
                self.state_point = Some((t, y.clone()));
                self.f_eval = Some(f_current);
                self.age = 0;
                self.factorized = false;

                Ok(self.jacobian.as_ref().unwrap())
            }
            JacobianStrategy::Adaptive => {
                // For adaptive strategy, try autodiff first if available
                if is_autodiff_available() {
                    // Try using autodiff
                    let f_current = f(t, y.view());
                    match autodiff_jacobian(f, t, y, &f_current, scale.unwrap_or(F::one())) {
                        Ok(jac) => {
                            self.jacobian = Some(jac);
                        }
                        Err(_) => {
                            // Fall back to finite differences if autodiff fails
                            self.jacobian = Some(finite_difference_jacobian(
                                f,
                                t,
                                y,
                                &f_current,
                                F::from(1e-8).unwrap(),
                            ));
                        }
                    }
                } else {
                    // Use finite differences directly if autodiff is not available
                    let f_current = f(t, y.view());
                    self.jacobian = Some(finite_difference_jacobian(
                        f,
                        t,
                        y,
                        &f_current,
                        F::from(1e-8).unwrap(),
                    ));
                }

                self.age = 0;
                self.factorized = false;

                Ok(self.jacobian.as_ref().unwrap())
            }
        }
    }

    /// Helper function to switch strategy temporarily
    fn update_jacobian_with_strategy<Func>(
        &mut self,
        t: F,
        y: &Array1<F>,
        f: &Func,
        scale: Option<F>,
        strategy: JacobianStrategy,
    ) -> IntegrateResult<&Array2<F>>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
    {
        let original_strategy = self.strategy;
        self.strategy = strategy;
        self.update_jacobian(t, y, f, scale)?;
        self.strategy = original_strategy;
        Ok(self.jacobian.as_ref().unwrap())
    }

    /// Helper function to compute dense finite difference Jacobian
    fn compute_dense_finite_difference<Func>(
        &self,
        t: F,
        y: &Array1<F>,
        f_current: &Array1<F>,
        f: &Func,
        jac: &mut Array2<F>,
        base_eps: F,
    ) where
        Func: Fn(F, ArrayView1<F>) -> Array1<F>,
    {
        let n = y.len();

        for j in 0..n {
            // Compute perturbation size scaled by variable magnitude
            let eps = base_eps * (F::one() + y[j].abs()).max(F::one());

            // Perturb the j-th component
            let mut y_perturbed = y.clone();
            y_perturbed[j] += eps;

            // Evaluate function at perturbed point
            let f_perturbed = f(t, y_perturbed.view());

            // Compute j-th column of Jacobian
            for i in 0..n {
                jac[[i, j]] = (f_perturbed[i] - f_current[i]) / eps;
            }
        }
    }

    /// Get the current Jacobian (returns None if not computed)
    pub fn jacobian(&self) -> Option<&Array2<F>> {
        self.jacobian.as_ref()
    }

    /// Get the age of the current Jacobian
    pub fn age(&self) -> usize {
        self.age
    }

    /// Update the age threshold for recomputation
    pub fn set_max_age(&mut self, max_age: usize) {
        self.max_age = max_age;
    }

    /// Mark the Jacobian as factorized
    pub fn mark_factorized(&mut self, factorized: bool) {
        self.factorized = factorized;
    }

    /// Check if the Jacobian is factorized
    pub fn is_factorized(&self) -> bool {
        self.factorized
    }
}
