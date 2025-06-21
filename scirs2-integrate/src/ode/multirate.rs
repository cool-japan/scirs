//! Multirate Methods for ODEs with Multiple Timescales
//!
//! This module implements multirate integration methods for systems where
//! different components evolve on different time scales. This is common in
//! many applications like:
//! - Chemical kinetics (fast/slow reactions)
//! - Electrical circuits (fast/slow transients)
//! - Climate models (fast weather, slow climate)
//! - Biological systems (fast enzyme kinetics, slow gene expression)

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::{ODEMethod, ODEResult};
use ndarray::{s, Array1, ArrayView1};
use std::collections::VecDeque;

/// Multirate ODE system with fast and slow components
pub trait MultirateSystem<F: IntegrateFloat> {
    /// Evaluate slow component: dy_slow/dt = f_slow(t, y_slow, y_fast)
    fn slow_rhs(&self, t: F, y_slow: ArrayView1<F>, y_fast: ArrayView1<F>) -> Array1<F>;

    /// Evaluate fast component: dy_fast/dt = f_fast(t, y_slow, y_fast)
    fn fast_rhs(&self, t: F, y_slow: ArrayView1<F>, y_fast: ArrayView1<F>) -> Array1<F>;

    /// Get dimension of slow variables
    fn slow_dim(&self) -> usize;

    /// Get dimension of fast variables
    fn fast_dim(&self) -> usize;
}

/// Multirate integration method types
#[derive(Debug, Clone)]
pub enum MultirateMethod {
    /// Explicit multirate Runge-Kutta method
    ExplicitMRK {
        macro_steps: usize,
        micro_steps: usize,
    },
    /// Implicit-explicit (IMEX) multirate method
    IMEX {
        macro_steps: usize,
        micro_steps: usize,
    },
    /// Compound fast-slow method
    CompoundFastSlow {
        fast_method: ODEMethod,
        slow_method: ODEMethod,
    },
    /// Extrapolated multirate method
    Extrapolated { base_ratio: usize, levels: usize },
}

/// Multirate solver configuration
#[derive(Debug, Clone)]
pub struct MultirateOptions<F: IntegrateFloat> {
    /// Multirate method to use
    pub method: MultirateMethod,
    /// Macro step size (for slow components)
    pub macro_step: F,
    /// Relative tolerance
    pub rtol: F,
    /// Absolute tolerance
    pub atol: F,
    /// Maximum number of macro steps
    pub max_steps: usize,
    /// Time scale separation estimate
    pub timescale_ratio: Option<F>,
}

impl<F: IntegrateFloat> Default for MultirateOptions<F> {
    fn default() -> Self {
        Self {
            method: MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 10,
            },
            macro_step: F::from(0.01).unwrap(),
            rtol: F::from(1e-6).unwrap(),
            atol: F::from(1e-9).unwrap(),
            max_steps: 10000,
            timescale_ratio: None,
        }
    }
}

/// Multirate ODE solver
pub struct MultirateSolver<F: IntegrateFloat> {
    options: MultirateOptions<F>,
    /// History of solutions for extrapolation methods
    history: VecDeque<(F, Array1<F>)>,
    /// Current macro step size
    current_macro_step: F,
    /// Current micro step size
    #[allow(dead_code)]
    current_micro_step: F,
}

impl<F: IntegrateFloat> MultirateSolver<F> {
    /// Create new multirate solver
    pub fn new(options: MultirateOptions<F>) -> Self {
        let current_macro_step = options.macro_step;
        let current_micro_step = match &options.method {
            MultirateMethod::ExplicitMRK { micro_steps, .. } => {
                current_macro_step / F::from(*micro_steps).unwrap()
            }
            MultirateMethod::IMEX { micro_steps, .. } => {
                current_macro_step / F::from(*micro_steps).unwrap()
            }
            _ => current_macro_step / F::from(10).unwrap(),
        };

        Self {
            options,
            history: VecDeque::new(),
            current_macro_step,
            current_micro_step,
        }
    }

    /// Solve multirate ODE system
    pub fn solve<S>(
        &mut self,
        system: S,
        t_span: [F; 2],
        y0: Array1<F>,
    ) -> IntegrateResult<ODEResult<F>>
    where
        S: MultirateSystem<F>,
    {
        let [t0, tf] = t_span;
        let slow_dim = system.slow_dim();
        let fast_dim = system.fast_dim();

        if y0.len() != slow_dim + fast_dim {
            return Err(IntegrateError::ValueError(format!(
                "Initial condition dimension {} does not match system dimension {}",
                y0.len(),
                slow_dim + fast_dim
            )));
        }

        let mut t = t0;
        let mut y = y0.clone();
        let mut solution_t = vec![t];
        let mut solution_y = vec![y.clone()];
        let mut step_count = 0;

        while t < tf && step_count < self.options.max_steps {
            // Adjust step size near final time
            let dt = if t + self.current_macro_step > tf {
                tf - t
            } else {
                self.current_macro_step
            };

            // Split state into slow and fast components
            let y_slow = y.slice(s![..slow_dim]).to_owned();
            let y_fast = y.slice(s![slow_dim..]).to_owned();

            // Take multirate step
            let (new_y_slow, new_y_fast) = match &self.options.method {
                MultirateMethod::ExplicitMRK {
                    macro_steps,
                    micro_steps,
                } => self.explicit_mrk_step(
                    &system,
                    t,
                    dt,
                    y_slow.view(),
                    y_fast.view(),
                    *macro_steps,
                    *micro_steps,
                )?,
                MultirateMethod::IMEX {
                    macro_steps,
                    micro_steps,
                } => self.imex_step(
                    &system,
                    t,
                    dt,
                    y_slow.view(),
                    y_fast.view(),
                    *macro_steps,
                    *micro_steps,
                )?,
                MultirateMethod::CompoundFastSlow {
                    fast_method: _,
                    slow_method: _,
                } => self.compound_fast_slow_step(&system, t, dt, y_slow.view(), y_fast.view())?,
                MultirateMethod::Extrapolated { base_ratio, levels } => self.extrapolated_step(
                    &system,
                    t,
                    dt,
                    y_slow.view(),
                    y_fast.view(),
                    *base_ratio,
                    *levels,
                )?,
            };

            // Combine slow and fast components
            let mut new_y = Array1::zeros(slow_dim + fast_dim);
            new_y.slice_mut(s![..slow_dim]).assign(&new_y_slow);
            new_y.slice_mut(s![slow_dim..]).assign(&new_y_fast);

            t += dt;
            y = new_y;
            solution_t.push(t);
            solution_y.push(y.clone());
            step_count += 1;

            // Update history for extrapolation methods
            if matches!(self.options.method, MultirateMethod::Extrapolated { .. }) {
                self.history.push_back((t, y.clone()));
                if self.history.len() > 10 {
                    self.history.pop_front();
                }
            }
        }

        if step_count >= self.options.max_steps {
            return Err(IntegrateError::ConvergenceError(
                "Maximum number of steps exceeded in multirate solver".to_string(),
            ));
        }

        Ok(ODEResult {
            t: solution_t,
            y: solution_y,
            success: true,
            message: Some(format!("Multirate method: {:?}", self.options.method)),
            n_eval: step_count * 4, // Approximate
            n_steps: step_count,
            n_accepted: step_count,
            n_rejected: 0,
            n_lu: 0,
            n_jac: 0,
            method: ODEMethod::RK4, // Default representation
        })
    }

    /// Explicit multirate Runge-Kutta step
    fn explicit_mrk_step<S>(
        &self,
        system: &S,
        t: F,
        dt: F,
        y_slow: ArrayView1<F>,
        y_fast: ArrayView1<F>,
        _macro_steps: usize,
        micro_steps: usize,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>
    where
        S: MultirateSystem<F>,
    {
        let dt_micro = dt / F::from(micro_steps).unwrap();

        // RK4 step for slow component (large step)
        let k1_slow = system.slow_rhs(t, y_slow, y_fast);

        // Fast component evolution over macro step with micro steps
        let mut y_fast_current = y_fast.to_owned();
        let mut t_micro = t;

        for _ in 0..micro_steps {
            // RK4 micro step for fast component
            let k1_fast = system.fast_rhs(t_micro, y_slow, y_fast_current.view());
            let k2_fast = system.fast_rhs(
                t_micro + dt_micro / F::from(2).unwrap(),
                y_slow,
                (y_fast_current.clone() + k1_fast.clone() * dt_micro / F::from(2).unwrap()).view(),
            );
            let k3_fast = system.fast_rhs(
                t_micro + dt_micro / F::from(2).unwrap(),
                y_slow,
                (y_fast_current.clone() + k2_fast.clone() * dt_micro / F::from(2).unwrap()).view(),
            );
            let k4_fast = system.fast_rhs(
                t_micro + dt_micro,
                y_slow,
                (y_fast_current.clone() + k3_fast.clone() * dt_micro).view(),
            );

            let two = F::from(2).unwrap();
            let six = F::from(6).unwrap();
            let rk_sum = k1_fast.clone() + &k2_fast * two + &k3_fast * two + k4_fast.clone();
            y_fast_current = y_fast_current + &rk_sum * (dt_micro / six);
            t_micro += dt_micro;
        }

        // Complete slow step using final fast state
        let k2_slow = system.slow_rhs(t + dt / F::from(2).unwrap(), y_slow, y_fast_current.view());
        let k3_slow = system.slow_rhs(
            t + dt / F::from(2).unwrap(),
            (y_slow.to_owned() + k1_slow.clone() * dt / F::from(2).unwrap()).view(),
            y_fast_current.view(),
        );
        let k4_slow = system.slow_rhs(
            t + dt,
            (y_slow.to_owned() + k3_slow.clone() * dt).view(),
            y_fast_current.view(),
        );

        let two = F::from(2).unwrap();
        let six = F::from(6).unwrap();
        let rk_sum_slow = k1_slow.clone() + &k2_slow * two + &k3_slow * two + k4_slow.clone();
        let new_y_slow = y_slow.to_owned() + &rk_sum_slow * (dt / six);

        Ok((new_y_slow, y_fast_current))
    }

    /// Implicit-explicit (IMEX) multirate step
    fn imex_step<S>(
        &self,
        system: &S,
        t: F,
        dt: F,
        y_slow: ArrayView1<F>,
        y_fast: ArrayView1<F>,
        _macro_steps: usize,
        micro_steps: usize,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>
    where
        S: MultirateSystem<F>,
    {
        // For this implementation, use explicit treatment
        // In practice, IMEX would treat stiff fast components implicitly
        self.explicit_mrk_step(system, t, dt, y_slow, y_fast, _macro_steps, micro_steps)
    }

    /// Compound fast-slow method step
    fn compound_fast_slow_step<S>(
        &self,
        system: &S,
        t: F,
        dt: F,
        y_slow: ArrayView1<F>,
        y_fast: ArrayView1<F>,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>
    where
        S: MultirateSystem<F>,
    {
        // First solve fast subsystem to quasi-steady state
        let mut y_fast_current = y_fast.to_owned();
        let dt_fast = dt / F::from(100).unwrap(); // Very small steps for fast system

        // Fast relaxation phase
        for _ in 0..50 {
            // Allow fast system to equilibrate
            let k_fast = system.fast_rhs(t, y_slow, y_fast_current.view());
            y_fast_current = y_fast_current + k_fast * dt_fast;
        }

        // Then advance slow system with equilibrated fast variables
        let k_slow = system.slow_rhs(t, y_slow, y_fast_current.view());
        let new_y_slow = y_slow.to_owned() + k_slow * dt;

        // Final fast adjustment
        let k_fast_final = system.fast_rhs(t + dt, new_y_slow.view(), y_fast_current.view());
        let new_y_fast = y_fast_current + k_fast_final * dt;

        Ok((new_y_slow, new_y_fast))
    }

    /// Extrapolated multirate step
    fn extrapolated_step<S>(
        &self,
        system: &S,
        t: F,
        dt: F,
        y_slow: ArrayView1<F>,
        y_fast: ArrayView1<F>,
        base_ratio: usize,
        levels: usize,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>
    where
        S: MultirateSystem<F>,
    {
        // Richardson extrapolation with different micro step sizes
        let mut solutions = Vec::new();

        for level in 0..levels {
            let micro_steps = base_ratio * (2_usize.pow(level as u32));
            let (y_slow_approx, y_fast_approx) =
                self.explicit_mrk_step(system, t, dt, y_slow, y_fast, 4, micro_steps)?;
            solutions.push((y_slow_approx, y_fast_approx));
        }

        // Simple Richardson extrapolation (linear)
        if solutions.len() >= 2 {
            let (y_slow_coarse, y_fast_coarse) = &solutions[0];
            let (y_slow_fine, y_fast_fine) = &solutions[1];

            // Extrapolated solution: y_ext = y_fine + (y_fine - y_coarse)
            let y_slow_ext = y_slow_fine + (y_slow_fine - y_slow_coarse);
            let y_fast_ext = y_fast_fine + (y_fast_fine - y_fast_coarse);

            Ok((y_slow_ext, y_fast_ext))
        } else {
            Ok(solutions.into_iter().next().unwrap())
        }
    }
}

/// Example multirate system: fast oscillator coupled to slow drift
pub struct FastSlowOscillator<F: IntegrateFloat> {
    /// Fast frequency
    pub omega_fast: F,
    /// Slow time scale
    pub epsilon: F,
    /// Coupling strength
    pub coupling: F,
}

impl<F: IntegrateFloat> MultirateSystem<F> for FastSlowOscillator<F> {
    fn slow_rhs(&self, _t: F, y_slow: ArrayView1<F>, y_fast: ArrayView1<F>) -> Array1<F> {
        let x_slow = y_slow[0];
        let v_slow = y_slow[1];
        let x_fast = y_fast[0];

        // Slow dynamics: influenced by fast oscillations
        let dx_slow_dt = v_slow;
        let dv_slow_dt = -self.epsilon * x_slow + self.coupling * x_fast;

        Array1::from_vec(vec![dx_slow_dt, dv_slow_dt])
    }

    fn fast_rhs(&self, _t: F, y_slow: ArrayView1<F>, y_fast: ArrayView1<F>) -> Array1<F> {
        let x_slow = y_slow[0];
        let x_fast = y_fast[0];
        let v_fast = y_fast[1];

        // Fast oscillator dynamics
        let dx_fast_dt = v_fast;
        let dv_fast_dt = -self.omega_fast * self.omega_fast * x_fast + self.coupling * x_slow;

        Array1::from_vec(vec![dx_fast_dt, dv_fast_dt])
    }

    fn slow_dim(&self) -> usize {
        2
    }
    fn fast_dim(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_multirate_system_dimensions() {
        let system = FastSlowOscillator {
            omega_fast: 10.0,
            epsilon: 0.1,
            coupling: 0.05,
        };

        assert_eq!(system.slow_dim(), 2);
        assert_eq!(system.fast_dim(), 2);
        assert_eq!(system.slow_dim() + system.fast_dim(), 4);
    }

    #[test]
    fn test_multirate_solver_creation() {
        let options = MultirateOptions {
            method: MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 10,
            },
            macro_step: 0.01,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 1000,
            timescale_ratio: Some(100.0),
        };

        let solver = MultirateSolver::new(options);
        assert_abs_diff_eq!(solver.current_macro_step, 0.01);
        assert_abs_diff_eq!(solver.current_micro_step, 0.001);
    }

    #[test]
    fn test_fast_slow_oscillator_solve() {
        let system = FastSlowOscillator {
            omega_fast: 20.0, // Fast oscillations
            epsilon: 0.1,     // Slow dynamics
            coupling: 0.02,   // Weak coupling
        };

        let options = MultirateOptions {
            method: MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 20,
            },
            macro_step: 0.05,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 200,
            timescale_ratio: Some(200.0),
        };

        let mut solver = MultirateSolver::new(options);

        // Initial conditions: [x_slow, v_slow, x_fast, v_fast]
        let y0 = Array1::from_vec(vec![1.0, 0.0, 0.1, 0.0]);

        let result = solver.solve(system, [0.0, 1.0], y0.clone()).unwrap();

        // Check that solution was computed
        assert!(result.t.len() > 1);
        assert_eq!(result.y.len(), result.t.len());
        assert_eq!(result.y[0].len(), 4);

        // Check that fast and slow components behave appropriately
        let final_state = result.y.last().unwrap();

        // Fast oscillator should still be oscillating (non-zero velocity)
        let fast_velocity: f64 = final_state[3];
        assert!(fast_velocity.abs() > 1e-6); // Fast velocity

        // Slow component should have evolved
        let slow_pos_change: f64 = final_state[0] - y0[0];
        assert!(slow_pos_change.abs() > 1e-3); // Slow position changed
    }

    #[test]
    fn test_compound_fast_slow_method() {
        let system = FastSlowOscillator {
            omega_fast: 50.0, // Very fast oscillations
            epsilon: 0.05,    // Very slow dynamics
            coupling: 0.01,   // Weak coupling
        };

        let options = MultirateOptions {
            method: MultirateMethod::CompoundFastSlow {
                fast_method: ODEMethod::RK4,
                slow_method: ODEMethod::RK4,
            },
            macro_step: 0.1,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 100,
            timescale_ratio: Some(1000.0),
        };

        let mut solver = MultirateSolver::new(options);
        let y0 = Array1::from_vec(vec![1.0, 0.0, 0.1, 0.0]);

        let result = solver.solve(system, [0.0, 0.5], y0).unwrap();

        assert!(result.t.len() > 1);
        assert!(result.n_steps > 0);
    }

    #[test]
    fn test_extrapolated_multirate_method() {
        let system = FastSlowOscillator {
            omega_fast: 15.0,
            epsilon: 0.2,
            coupling: 0.03,
        };

        let options = MultirateOptions {
            method: MultirateMethod::Extrapolated {
                base_ratio: 5,
                levels: 2,
            },
            macro_step: 0.02,
            rtol: 1e-8,
            atol: 1e-11,
            max_steps: 500,
            timescale_ratio: Some(75.0),
        };

        let mut solver = MultirateSolver::new(options);
        let y0 = Array1::from_vec(vec![0.5, 0.0, 0.2, 0.1]);

        let result = solver.solve(system, [0.0, 0.2], y0).unwrap();

        assert!(result.t.len() > 1);
        assert!(result.n_steps > 0);
    }
}
