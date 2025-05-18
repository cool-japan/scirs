//! Basin-hopping algorithm for global optimization
//!
//! Basin-hopping is a stochastic algorithm which attempts to find the
//! global minimum of a function by combining random perturbations with
//! local minimization.

use crate::error::OptimizeError;
use crate::unconstrained::{minimize, Bounds, Method, OptimizeResult, Options};
use ndarray::{Array1, ArrayView1};
use rand::distr::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;

/// Options for Basin-hopping algorithm
#[derive(Debug, Clone)]
pub struct BasinHoppingOptions {
    /// Number of basin hopping iterations
    pub niter: usize,
    /// Temperature parameter for accept/reject criterion (higher means more permissive)
    pub temperature: f64,
    /// Step size for random displacement
    pub stepsize: f64,
    /// Number of iterations with no improvement before terminating
    pub niter_success: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Local minimization method to use
    pub minimizer_method: Method,
    /// Bounds for variables
    pub bounds: Option<Vec<(f64, f64)>>,
}

impl Default for BasinHoppingOptions {
    fn default() -> Self {
        Self {
            niter: 100,
            temperature: 1.0,
            stepsize: 0.5,
            niter_success: None,
            seed: None,
            minimizer_method: Method::LBFGS,
            bounds: None,
        }
    }
}

/// Accept test function type
pub type AcceptTest = Box<dyn Fn(f64, f64, f64) -> bool>;

/// Take step function type  
pub type TakeStep = Box<dyn FnMut(&Array1<f64>) -> Array1<f64>>;

/// Basin-hopping solver
pub struct BasinHopping<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    func: F,
    x0: Array1<f64>,
    options: BasinHoppingOptions,
    ndim: usize,
    rng: StdRng,
    accept_test: AcceptTest,
    take_step: TakeStep,
    storage: Storage,
    nfev: usize,
}

/// Storage for the best result found
#[derive(Debug, Clone)]
struct Storage {
    x: Array1<f64>,
    fun: f64,
    success: bool,
}

impl Storage {
    fn new(x: Array1<f64>, fun: f64, success: bool) -> Self {
        Self { x, fun, success }
    }

    fn update(&mut self, x: Array1<f64>, fun: f64, success: bool) -> bool {
        if success && (fun < self.fun || !self.success) {
            self.x = x;
            self.fun = fun;
            self.success = success;
            true
        } else {
            false
        }
    }
}

impl<F> BasinHopping<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    /// Create new Basin-hopping solver
    pub fn new(
        func: F,
        x0: Array1<f64>,
        options: BasinHoppingOptions,
        accept_test: Option<AcceptTest>,
        take_step: Option<TakeStep>,
    ) -> Self {
        let ndim = x0.len();
        let seed = options.seed.unwrap_or_else(rand::random);
        let rng = StdRng::seed_from_u64(seed);

        // Default accept test is Metropolis criterion
        let accept_test = accept_test.unwrap_or_else(|| {
            let temp = options.temperature;
            Box::new(move |f_new: f64, f_old: f64, _: f64| {
                if f_new < f_old {
                    true
                } else {
                    let delta = (f_old - f_new) / temp;
                    delta > 0.0 && rand::random::<f64>() < delta.exp()
                }
            })
        });

        // Default take step is random displacement
        let take_step = take_step.unwrap_or_else(|| {
            let stepsize = options.stepsize;
            let bounds = options.bounds.clone();
            let mut rng = rng.clone();
            Box::new(move |x: &Array1<f64>| {
                let mut x_new = x.clone();
                for i in 0..x.len() {
                    let uniform = Uniform::new(-stepsize, stepsize).unwrap();
                    x_new[i] += rng.sample(uniform);

                    // Apply bounds if specified
                    if let Some(ref bounds) = bounds {
                        if i < bounds.len() {
                            let (lb, ub) = bounds[i];
                            x_new[i] = x_new[i].max(lb).min(ub);
                        }
                    }
                }
                x_new
            })
        });

        // Perform initial minimization to get starting point
        let initial_result = minimize(
            func.clone(),
            &x0.to_vec(),
            options.minimizer_method,
            Some(Options {
                bounds: options.bounds.clone().map(|b| {
                    Bounds::from_vecs(
                        b.iter().map(|&(lb, _)| Some(lb)).collect(),
                        b.iter().map(|&(_, ub)| Some(ub)).collect(),
                    )
                    .unwrap()
                }),
                ..Default::default()
            }),
        )
        .unwrap();

        let storage = Storage::new(
            initial_result.x.clone(),
            initial_result.fun,
            initial_result.success,
        );

        Self {
            func,
            x0: initial_result.x,
            options,
            ndim,
            rng,
            accept_test,
            take_step,
            storage,
            nfev: initial_result.nfev,
        }
    }

    /// Run one iteration of basin-hopping
    fn step(&mut self) -> (Array1<f64>, f64, bool) {
        // Take a random step
        let x_new = (self.take_step)(&self.x0);

        // Minimize from new point
        let result = minimize(
            |x| (self.func)(x),
            &x_new.to_vec(),
            self.options.minimizer_method,
            Some(Options {
                bounds: self.options.bounds.clone().map(|b| {
                    Bounds::from_vecs(
                        b.iter().map(|&(lb, _)| Some(lb)).collect(),
                        b.iter().map(|&(_, ub)| Some(ub)).collect(),
                    )
                    .unwrap()
                }),
                ..Default::default()
            }),
        )
        .unwrap();

        self.nfev += result.nfev;

        // Accept or reject the new minimum
        let accept = (self.accept_test)(result.fun, self.storage.fun, self.temperature());

        if accept {
            self.x0 = result.x.clone();
        }

        (result.x, result.fun, result.success)
    }

    /// Get current temperature (could be adaptive in future)
    fn temperature(&self) -> f64 {
        self.options.temperature
    }

    /// Run the basin-hopping algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut nit = 0;
        let mut success_counter = 0;
        let mut message = "Maximum number of iterations reached".to_string();

        for _ in 0..self.options.niter {
            let (x, fun, success) = self.step();
            nit += 1;

            // Update storage if better solution found
            if self.storage.update(x.clone(), fun, success) {
                success_counter = 0;
            } else {
                success_counter += 1;
            }

            // Check early termination based on success iterations
            if let Some(niter_success) = self.options.niter_success {
                if success_counter >= niter_success {
                    message = format!("No improvement in {} iterations", niter_success);
                    break;
                }
            }
        }

        OptimizeResult {
            x: self.storage.x.clone(),
            fun: self.storage.fun,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit,
            iterations: nit,
            success: self.storage.success,
            message,
            ..Default::default()
        }
    }
}

/// Perform global optimization using basin-hopping
pub fn basinhopping<F>(
    func: F,
    x0: Array1<f64>,
    options: Option<BasinHoppingOptions>,
    accept_test: Option<AcceptTest>,
    take_step: Option<TakeStep>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();

    let mut solver = BasinHopping::new(func, x0, options, accept_test, take_step);
    Ok(solver.run())
}
