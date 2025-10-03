//! Integration of convergence diagnostics with callback system
//!
//! This module provides utilities for integrating convergence diagnostics
//! with optimization algorithms through a callback mechanism.

use crate::error::OptimizeError;
use crate::unconstrained::convergence_diagnostics::{
    ConvergenceDiagnostics, DiagnosticCollector, DiagnosticOptions, LineSearchDiagnostic,
};
use crate::unconstrained::OptimizeResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::cell::RefCell;
use std::rc::Rc;

/// Callback function type for optimization monitoring
pub type OptimizationCallback = Box<dyn FnMut(&CallbackInfo) -> CallbackResult>;

/// Information passed to callback functions
#[derive(Debug, Clone)]
pub struct CallbackInfo {
    /// Current iteration number
    pub iteration: usize,
    /// Current point
    pub x: Array1<f64>,
    /// Current function value
    pub f: f64,
    /// Current gradient
    pub grad: Array1<f64>,
    /// Step taken (if available)
    pub step: Option<Array1<f64>>,
    /// Search direction (if available)
    pub direction: Option<Array1<f64>>,
    /// Line search information (if available)
    pub line_search: Option<LineSearchDiagnostic>,
    /// Time elapsed since start
    pub elapsed_time: std::time::Duration,
}

/// Result from callback function
#[derive(Debug, Clone, Copy)]
pub enum CallbackResult {
    /// Continue optimization
    Continue,
    /// Stop optimization
    Stop,
    /// Stop with custom message
    StopWithMessage(&'static str),
}

/// Wrapper for optimization with diagnostic callbacks
pub struct DiagnosticOptimizer {
    /// Diagnostic collector
    collector: Rc<RefCell<DiagnosticCollector>>,
    /// User callbacks
    callbacks: Vec<OptimizationCallback>,
    /// Start time
    start_time: std::time::Instant,
}

impl DiagnosticOptimizer {
    /// Create new diagnostic optimizer
    pub fn new(_diagnostic_options: DiagnosticOptions) -> Self {
        Self {
            collector: Rc::new(RefCell::new(DiagnosticCollector::new(_diagnostic_options))),
            callbacks: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Add a callback
    pub fn add_callback(&mut self, callback: OptimizationCallback) {
        self.callbacks.push(callback);
    }

    /// Add a simple progress callback
    pub fn add_progress_callback(&mut self, every_n_nit: usize) {
        let mut last_printed = 0;
        self.add_callback(Box::new(move |info| {
            if info.iteration >= last_printed + every_n_nit {
                println!(
                    "Iteration {}: f = {:.6e}, |grad| = {:.6e}",
                    info.iteration,
                    info.f,
                    info.grad.mapv(|x| x.abs()).sum()
                );
                last_printed = info.iteration;
            }
            CallbackResult::Continue
        }));
    }

    /// Add a convergence monitoring callback
    pub fn add_convergence_monitor(&mut self, patience: usize, min_improvement: f64) {
        let mut best_f = f64::INFINITY;
        let mut no_improvement_count = 0;

        self.add_callback(Box::new(move |info| {
            if info.f < best_f - min_improvement {
                best_f = info.f;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            if no_improvement_count >= patience {
                CallbackResult::StopWithMessage("Early stopping: no _improvement")
            } else {
                CallbackResult::Continue
            }
        }));
    }

    /// Add a time limit callback
    pub fn add_time_limit(&mut self, max_duration: std::time::Duration) {
        self.add_callback(Box::new(move |info| {
            if info.elapsed_time > max_duration {
                CallbackResult::StopWithMessage("Time limit exceeded")
            } else {
                CallbackResult::Continue
            }
        }));
    }

    /// Process callbacks and update diagnostics
    pub fn process_iteration(&mut self, info: &CallbackInfo) -> CallbackResult {
        // Update diagnostic collector
        if let (Some(step), Some(direction), Some(line_search)) =
            (&info.step, &info.direction, &info.line_search)
        {
            self.collector.borrow_mut().record_iteration(
                info.f,
                &info.grad.view(),
                &step.view(),
                &direction.view(),
                line_search.clone(),
            );
        }

        // Process user callbacks
        for callback in &mut self.callbacks {
            match callback(info) {
                CallbackResult::Continue => continue,
                result => return result,
            }
        }

        CallbackResult::Continue
    }

    /// Get final diagnostics
    pub fn get_diagnostics(self) -> ConvergenceDiagnostics {
        let collector = Rc::try_unwrap(self.collector)
            .expect("Failed to unwrap Rc")
            .into_inner();
        collector.finalize()
    }
}

/// Optimization wrapper that integrates diagnostics
#[allow(dead_code)]
pub fn optimize_with_diagnostics<F, O>(
    optimizer_fn: O,
    fun: F,
    x0: Array1<f64>,
    diagnostic_options: DiagnosticOptions,
    callbacks: Vec<OptimizationCallback>,
) -> Result<(OptimizeResult<f64>, ConvergenceDiagnostics), OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
    O: FnOnce(
        F,
        Array1<f64>,
        &mut DiagnosticOptimizer,
    ) -> Result<OptimizeResult<f64>, OptimizeError>,
{
    let mut diagnostic_optimizer = DiagnosticOptimizer::new(diagnostic_options);

    // Add user callbacks
    for callback in callbacks {
        diagnostic_optimizer.add_callback(callback);
    }

    // Run optimization
    let result = optimizer_fn(fun, x0, &mut diagnostic_optimizer)?;

    // Get diagnostics
    let diagnostics = diagnostic_optimizer.get_diagnostics();

    Ok((result, diagnostics))
}

/// Example of integrating diagnostics into an optimization algorithm
#[allow(dead_code)]
pub fn minimize_with_diagnostics<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: &crate::unconstrained::Options,
    diagnostic_optimizer: &mut DiagnosticOptimizer,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let mut x = x0.clone();
    let mut f = fun(&x.view());
    let mut iteration = 0;

    // Simplified optimization loop for demonstration
    loop {
        // Compute gradient (simplified - would use finite differences or AD)
        let grad = finite_diff_gradient(&mut fun, &x.view(), 1e-8);

        // Compute search direction (simplified - just negative gradient)
        let direction = -&grad;

        // Line search (simplified)
        let alpha = 0.1;
        let step = alpha * &direction;
        let x_new = &x + &step;
        let f_new = fun(&x_new.view());

        // Create callback info
        let callback_info = CallbackInfo {
            iteration,
            x: x.clone(),
            f,
            grad: grad.clone(),
            step: Some(step.clone()),
            direction: Some(direction.clone()),
            line_search: Some(LineSearchDiagnostic {
                n_fev: 1,
                n_gev: 1,
                alpha,
                alpha_init: 1.0,
                success: f_new < f,
                wolfe_satisfied: (true, true),
            }),
            elapsed_time: diagnostic_optimizer.start_time.elapsed(),
        };

        // Process callbacks
        match diagnostic_optimizer.process_iteration(&callback_info) {
            CallbackResult::Continue => {}
            CallbackResult::Stop => break,
            CallbackResult::StopWithMessage(msg) => {
                return Ok(OptimizeResult {
                    x,
                    fun: f,
                    nit: iteration,
                    func_evals: iteration * 2,
                    nfev: iteration * 2,
                    success: false,
                    message: msg.to_string(),
                    jacobian: Some(grad),
                    hessian: None,
                });
            }
        }

        // Check convergence
        if grad.mapv(|x| x.abs()).sum() < options.gtol {
            break;
        }

        // Update state
        x = x_new;
        f = f_new;
        iteration += 1;

        if iteration >= options.max_iter {
            break;
        }
    }

    Ok(OptimizeResult {
        x,
        fun: f,
        nit: iteration,
        func_evals: iteration * 2,
        nfev: iteration * 2,
        success: iteration < options.max_iter,
        message: if iteration < options.max_iter {
            "Optimization converged".to_string()
        } else {
            "Maximum iterations reached".to_string()
        },
        jacobian: None,
        hessian: None,
    })
}

/// Simple finite difference gradient
#[allow(dead_code)]
fn finite_diff_gradient<F>(fun: &mut F, x: &ArrayView1<f64>, eps: f64) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let f0 = fun(x);
    let mut x_pert = x.to_owned();

    for i in 0..n {
        let h = eps * (1.0 + x[i].abs());
        x_pert[i] = x[i] + h;
        let f_plus = fun(&x_pert.view());
        grad[i] = (f_plus - f0) / h;
        x_pert[i] = x[i];
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_optimizer() {
        let options = DiagnosticOptions::default();
        let mut optimizer = DiagnosticOptimizer::new(options);

        // Add a simple callback
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();
        optimizer.add_callback(Box::new(move |_info| {
            callback_called_clone.store(true, Ordering::SeqCst);
            CallbackResult::Continue
        }));

        // Create test callback info
        let info = CallbackInfo {
            iteration: 0,
            x: Array1::zeros(2),
            f: 1.0,
            grad: Array1::ones(2),
            step: Some(Array1::from_vec(vec![0.1, 0.1])),
            direction: Some(Array1::from_vec(vec![-1.0, -1.0])),
            line_search: Some(LineSearchDiagnostic {
                n_fev: 1,
                n_gev: 1,
                alpha: 1.0,
                alpha_init: 1.0,
                success: true,
                wolfe_satisfied: (true, true),
            }),
            elapsed_time: std::time::Duration::from_secs(0),
        };

        let result = optimizer.process_iteration(&info);
        assert!(matches!(result, CallbackResult::Continue));
        assert!(callback_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_early_stopping_callback() {
        let options = DiagnosticOptions::default();
        let mut optimizer = DiagnosticOptimizer::new(options);

        optimizer.add_convergence_monitor(2, 0.1);

        // Create test info with no improvement
        let info = CallbackInfo {
            iteration: 0,
            x: Array1::zeros(2),
            f: 1.0,
            grad: Array1::ones(2),
            step: None,
            direction: None,
            line_search: None,
            elapsed_time: std::time::Duration::from_secs(0),
        };

        // First two iterations should continue
        assert!(matches!(
            optimizer.process_iteration(&info),
            CallbackResult::Continue
        ));
        assert!(matches!(
            optimizer.process_iteration(&info),
            CallbackResult::Continue
        ));

        // Third iteration with no improvement should stop
        let result = optimizer.process_iteration(&info);
        assert!(matches!(result, CallbackResult::StopWithMessage(_)));
    }
}
