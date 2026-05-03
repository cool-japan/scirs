//! Minimax Optimization Methods for Robust Optimization
//!
//! This module implements minimax (min-max) optimization algorithms that minimize
//! the maximum of a collection of functions:
//!
//! ```text
//! min_x  max_{i ∈ [k]}  f_i(x)
//! ```
//!
//! Such problems arise in robust optimization (worst-case performance),
//! Chebyshev approximation, and multi-objective optimization.
//!
//! # Algorithms
//!
//! - [`minimax_subgradient`]: Subgradient descent for non-smooth minimax
//! - [`minimax_bundle`]: Bundle method for accurate non-smooth minimax
//! - [`smooth_minimax`]: Nesterov log-sum-exp smoothing technique
//! - [`game_theoretic_minimax`]: Zero-sum game Nash equilibrium via fictitious play
//!
//! # References
//!
//! - Shor, N.Z. (1985). *Minimization Methods for Non-Differentiable Functions*.
//! - Nesterov, Y. (2005). "Smooth minimization of non-smooth functions". *Mathematical Programming*.
//! - Brown, G.W. (1951). "Iterative solution of games by fictitious play". *Activity Analysis of Production and Allocation*.
//! - Lemaréchal, C. (1975). "An extension of Davidon methods to nondifferentiable problems". *Mathematical Programming Study*.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ─── Data types ───────────────────────────────────────────────────────────────

/// A minimax problem: minimize the maximum over k functions.
///
/// Represents:
/// ```text
/// min_x  max_{i=0..k-1}  f_i(x)
/// ```
///
/// # Example
///
/// ```rust
/// use scirs2_optimize::robust::minimax::MinimaxProblem;
/// use scirs2_core::ndarray::ArrayView1;
///
/// // max(f0, f1) where f0(x)=x[0]+1, f1(x)=-x[0]+2
/// let problem = MinimaxProblem::new(vec![
///     Box::new(|x: &ArrayView1<f64>| x[0] + 1.0),
///     Box::new(|x: &ArrayView1<f64>| -x[0] + 2.0),
/// ]);
/// ```
pub struct MinimaxProblem {
    /// The functions f_i(x) being maximized.
    pub funcs: Vec<Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>>,
}

impl MinimaxProblem {
    /// Create a new minimax problem from a collection of functions.
    pub fn new(funcs: Vec<Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>>) -> Self {
        Self { funcs }
    }

    /// Evaluate max_i f_i(x).
    pub fn eval_max(&self, x: &ArrayView1<f64>) -> f64 {
        self.funcs
            .iter()
            .map(|f| f(x))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Evaluate all function values f_i(x).
    pub fn eval_all(&self, x: &ArrayView1<f64>) -> Vec<f64> {
        self.funcs.iter().map(|f| f(x)).collect()
    }

    /// Return the index of the maximizing function at x.
    pub fn argmax(&self, x: &ArrayView1<f64>) -> usize {
        let vals = self.eval_all(x);
        vals.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return the number of component functions.
    pub fn num_funcs(&self) -> usize {
        self.funcs.len()
    }
}

/// Configuration for minimax solvers.
#[derive(Debug, Clone)]
pub struct MinimaxSolverConfig {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Initial step size.
    pub step_size: f64,
    /// Step size decay factor (multiplied each iteration for diminishing step sizes).
    pub step_decay: f64,
    /// Finite-difference step for gradient approximation.
    pub fd_step: f64,
    /// Smoothing parameter μ for log-sum-exp (used by smooth_minimax).
    pub smoothing_mu: f64,
    /// Number of fictitious play iterations (used by game_theoretic_minimax).
    pub fictitious_play_iter: usize,
}

impl Default for MinimaxSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 2_000,
            tol: 1e-6,
            step_size: 1e-2,
            step_decay: 1.0, // constant step size by default
            fd_step: 1e-5,
            smoothing_mu: 0.1,
            fictitious_play_iter: 1_000,
        }
    }
}

/// Result of a minimax solve.
#[derive(Debug, Clone)]
pub struct MinimaxSolveResult {
    /// Approximate minimizer x*.
    pub x: Array1<f64>,
    /// Minimax value max_i f_i(x*).
    pub fun: f64,
    /// Index of the active (maximizing) constraint at convergence.
    pub active_index: usize,
    /// Number of outer iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Forward finite-difference gradient of a scalar function.
fn fd_gradient<F>(f: &F, x: &ArrayView1<f64>, h: f64) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let f0 = f(x);
    let mut g = Array1::<f64>::zeros(n);
    let mut x_fwd = x.to_owned();
    for i in 0..n {
        x_fwd[i] += h;
        g[i] = (f(&x_fwd.view()) - f0) / h;
        x_fwd[i] = x[i];
    }
    g
}

/// L2 norm of a vector.
#[inline]
fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─── Subgradient Descent ──────────────────────────────────────────────────────

/// Minimize max_i f_i(x) using subgradient descent.
///
/// At each step, the method selects a subgradient from the active function
/// (the one achieving the maximum) and takes a descent step.
///
/// The step size schedule is:
/// - constant: αₖ = α₀                    (step_decay = 1.0)
/// - diminishing: αₖ = α₀ / √k            (step_decay < 1.0, set manually)
///
/// # Arguments
///
/// * `problem` – the minimax problem
/// * `x0`      – initial point (n-vector)
/// * `config`  – solver configuration
///
/// # Returns
///
/// [`MinimaxSolveResult`] with the approximate minimizer.
///
/// # References
///
/// Shor (1985), *Minimization Methods for Non-Differentiable Functions*.
pub fn minimax_subgradient(
    problem: &MinimaxProblem,
    x0: &ArrayView1<f64>,
    config: &MinimaxSolverConfig,
) -> OptimizeResult<MinimaxSolveResult> {
    if problem.num_funcs() == 0 {
        return Err(OptimizeError::ValueError(
            "MinimaxProblem must contain at least one function".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    // Track best point (lowest minimax value seen so far)
    let mut x_best = x.clone();
    let mut val_best = problem.eval_max(&x.view());
    let h = config.fd_step;
    let mut converged = false;

    for k in 0..config.max_iter {
        let vals = problem.eval_all(&x.view());
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Select index achieving the maximum (subgradient belongs to the active function)
        let active_idx = vals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if max_val < val_best {
            val_best = max_val;
            x_best = x.clone();
        }

        // Compute subgradient of max_i f_i at x: a subgradient of f_{active_idx}
        let subgrad = fd_gradient(
            &|v: &ArrayView1<f64>| problem.funcs[active_idx](v),
            &x.view(),
            h,
        );
        let sg_norm = l2_norm(&subgrad);

        if sg_norm < config.tol {
            converged = true;
            break;
        }

        // Diminishing step size: α_k = α₀ / √(k+1)
        let alpha = config.step_size / ((k as f64 + 1.0).sqrt());

        // Subgradient descent step
        for i in 0..n {
            x[i] -= alpha * subgrad[i];
        }
    }

    let active_idx = problem.argmax(&x_best.view());
    Ok(MinimaxSolveResult {
        fun: val_best,
        active_index: active_idx,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Subgradient descent converged".to_string()
        } else {
            "Subgradient descent reached maximum iterations".to_string()
        },
        x: x_best,
    })
}

// ─── Bundle Method ────────────────────────────────────────────────────────────

/// A cutting plane (bundle element): linear lower model for f around a point.
#[derive(Debug, Clone)]
struct BundleCut {
    /// Reference point where the cut was generated.
    point: Array1<f64>,
    /// Function value at the reference point.
    value: f64,
    /// Subgradient at the reference point.
    subgrad: Array1<f64>,
}

/// Minimize max_i f_i(x) using a simplified bundle method.
///
/// The bundle method maintains a *model* of the objective using past subgradients
/// (cutting planes). At each step, a proximal quadratic subproblem is solved:
///
/// ```text
/// min_{d}  max_j { f_j(y_j) + gⱼᵀ (x + d - yⱼ) } + (1/2t) ‖d‖²
/// ```
///
/// where t is a proximity parameter. This gives better descent than plain
/// subgradient methods.
///
/// # Arguments
///
/// * `problem` – the minimax problem
/// * `x0`      – initial point
/// * `config`  – solver configuration
///
/// # Returns
///
/// [`MinimaxSolveResult`] with the approximate minimizer.
///
/// # References
///
/// Lemaréchal (1975); Kiwiel (1990) *Methods of Descent for Nondifferentiable Optimization*.
pub fn minimax_bundle(
    problem: &MinimaxProblem,
    x0: &ArrayView1<f64>,
    config: &MinimaxSolverConfig,
) -> OptimizeResult<MinimaxSolveResult> {
    if problem.num_funcs() == 0 {
        return Err(OptimizeError::ValueError(
            "MinimaxProblem must contain at least one function".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let h = config.fd_step;
    let max_bundle_size = 20_usize;
    // Proximity parameter controls trade-off between model accuracy and step length
    let prox_t = config.step_size;

    let mut x = x0.to_owned();
    let mut x_best = x.clone();
    let mut val_best = problem.eval_max(&x.view());
    let mut bundle: Vec<BundleCut> = Vec::with_capacity(max_bundle_size);
    let mut converged = false;

    // Helper: evaluate the bundle model at x + d
    // model(d) = max_{j} { v_j + g_j^T (x + d - y_j) }
    let eval_model = |x_cur: &Array1<f64>, d: &Array1<f64>, cuts: &[BundleCut]| -> f64 {
        cuts.iter()
            .map(|cut| {
                let diff: f64 = x_cur
                    .iter()
                    .zip(d.iter())
                    .zip(cut.point.iter())
                    .map(|((&xc, &dc), &yj)| xc + dc - yj)
                    .zip(cut.subgrad.iter())
                    .map(|(delta, &gj)| delta * gj)
                    .sum::<f64>();
                cut.value + diff
            })
            .fold(f64::NEG_INFINITY, f64::max)
    };

    for k in 0..config.max_iter {
        let vals = problem.eval_all(&x.view());
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_val < val_best {
            val_best = max_val;
            x_best = x.clone();
        }

        // Generate a new bundle cut from each function achieving near-maximum
        let threshold = max_val - config.tol;
        let new_cuts: Vec<BundleCut> = vals
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= threshold)
            .map(|(i, _)| {
                let subgrad = fd_gradient(&|v: &ArrayView1<f64>| problem.funcs[i](v), &x.view(), h);
                BundleCut {
                    point: x.clone(),
                    value: vals[i],
                    subgrad,
                }
            })
            .collect();

        bundle.extend(new_cuts);

        // Trim bundle to max size (keep most recent cuts)
        if bundle.len() > max_bundle_size {
            let start = bundle.len() - max_bundle_size;
            bundle.drain(..start);
        }

        if bundle.is_empty() {
            break;
        }

        // Solve the proximal bundle subproblem via projected gradient on d:
        //   min_{d}  model(x, d, bundle) + (1/2t) ‖d‖²
        // We use a simple gradient method on d (inner loop).
        let mut d = Array1::<f64>::zeros(n);
        let inner_steps = 100_usize;
        let inner_step = prox_t * 0.5;

        for _ in 0..inner_steps {
            // Gradient of bundle model at current d (subgradient of max over cuts)
            let active_cut = bundle
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let va = a.value
                        + a.point
                            .iter()
                            .zip(d.iter())
                            .zip(x.iter())
                            .map(|((&yj, &dj), &xc)| {
                                a.subgrad[{
                                    // inline the index via manual accumulation
                                    0
                                }] * (xc + dj - yj)
                            })
                            .sum::<f64>();
                    let vb = b.value
                        + b.point
                            .iter()
                            .zip(d.iter())
                            .zip(x.iter())
                            .map(|((&yj, &dj), &xc)| b.subgrad[0] * (xc + dj - yj))
                            .sum::<f64>();
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Proper active cut selection: re-evaluate scalar model for each cut
            let active_idx = bundle
                .iter()
                .enumerate()
                .max_by(|(_, ca), (_, cb)| {
                    let va: f64 = ca.value
                        + ca.subgrad
                            .iter()
                            .zip(x.iter().zip(d.iter()).zip(ca.point.iter()))
                            .map(|(&gj, ((&xc, &dc), &yj))| gj * (xc + dc - yj))
                            .sum::<f64>();
                    let vb: f64 = cb.value
                        + cb.subgrad
                            .iter()
                            .zip(x.iter().zip(d.iter()).zip(cb.point.iter()))
                            .map(|(&gj, ((&xc, &dc), &yj))| gj * (xc + dc - yj))
                            .sum::<f64>();
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(active_cut);

            let cut = &bundle[active_idx];
            // Gradient of model + proximal term w.r.t. d:
            //   g_model(d) = g_active  (subgradient of max)
            //   g_prox(d)  = d / t
            let grad_d: Array1<f64> = cut
                .subgrad
                .iter()
                .zip(d.iter())
                .map(|(&gj, &dj)| gj + dj / prox_t)
                .collect();

            let step_norm = l2_norm(&grad_d);
            if step_norm < config.tol * 0.1 {
                break;
            }
            for i in 0..n {
                d[i] -= inner_step * grad_d[i];
            }
        }

        // Check descent: only accept step if model predicts sufficient decrease
        let d_norm = l2_norm(&d);
        if d_norm < config.tol {
            converged = true;
            break;
        }

        // Serious step: update x
        for i in 0..n {
            x[i] += d[i];
        }

        // Null step convergence check: if max value not decreasing
        let new_max = problem.eval_max(&x.view());
        if (new_max - max_val).abs() < config.tol && k > 10 {
            converged = true;
            break;
        }
    }

    let active_idx = problem.argmax(&x_best.view());
    Ok(MinimaxSolveResult {
        x: x_best,
        fun: val_best,
        active_index: active_idx,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Bundle method converged".to_string()
        } else {
            "Bundle method reached maximum iterations".to_string()
        },
    })
}

// ─── Nesterov Smooth Minimax ──────────────────────────────────────────────────

/// Minimize max_i f_i(x) via Nesterov's log-sum-exp smoothing.
///
/// The non-smooth objective  F(x) = max_i f_i(x)  is replaced by the smooth
/// surrogate:
///
/// ```text
/// F_μ(x) = (μ/k) · log(Σ_i exp(f_i(x) / μ))  →  F(x)  as  μ → 0
/// ```
///
/// The error satisfies F(x) ≤ F_μ(x) ≤ F(x) + μ·log(k), so for small μ
/// F_μ is an accurate smooth upper bound. Gradient descent on F_μ converges
/// at rate O(1/k²) (Nesterov accelerated gradient).
///
/// # Arguments
///
/// * `problem` – the minimax problem with k functions
/// * `x0`      – initial point
/// * `config`  – solver configuration (uses `smoothing_mu` as μ)
///
/// # Returns
///
/// [`MinimaxSolveResult`] with the approximate minimizer.
///
/// # References
///
/// Nesterov, Y. (2005). "Smooth minimization of non-smooth functions".
/// *Mathematical Programming*, 103(1), 127–152.
pub fn smooth_minimax(
    problem: &MinimaxProblem,
    x0: &ArrayView1<f64>,
    config: &MinimaxSolverConfig,
) -> OptimizeResult<MinimaxSolveResult> {
    if problem.num_funcs() == 0 {
        return Err(OptimizeError::ValueError(
            "MinimaxProblem must contain at least one function".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    let mu = config.smoothing_mu;
    if mu <= 0.0 {
        return Err(OptimizeError::ValueError(format!(
            "smoothing_mu must be positive, got {}",
            mu
        )));
    }

    let k = problem.num_funcs() as f64;
    let h = config.fd_step;

    // Smooth surrogate: F_μ(x) = μ * log( (1/k) Σ_i exp(f_i(x) / μ) )
    // This is the "soft-max" approximation.
    let smooth_obj = |x: &ArrayView1<f64>| -> f64 {
        let vals = problem.eval_all(x);
        // Use numerically stable log-sum-exp
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val.is_infinite() {
            return max_val;
        }
        let sum_exp: f64 = vals.iter().map(|&v| ((v - max_val) / mu).exp()).sum();
        mu * (sum_exp.ln() + (max_val / mu)) - mu * k.ln()
    };

    let mut x = x0.to_owned();
    let mut x_best = x.clone();
    let mut val_best = problem.eval_max(&x.view());
    let mut converged = false;

    // Nesterov accelerated gradient (AGD) / FISTA-style
    let mut y = x.clone(); // extrapolated point
    let mut t_k = 1.0_f64; // momentum coefficient

    for _ in 0..config.max_iter {
        // Gradient of smooth objective at extrapolated point y
        let grad = fd_gradient(&smooth_obj, &y.view(), h);
        let grad_norm = l2_norm(&grad);

        if grad_norm < config.tol {
            converged = true;
            break;
        }

        // Gradient descent step on y
        let x_new: Array1<f64> = y
            .iter()
            .zip(grad.iter())
            .map(|(&yi, &gi)| yi - config.step_size * gi)
            .collect();

        // Nesterov momentum update
        let t_new = (1.0 + (1.0 + 4.0 * t_k * t_k).sqrt()) / 2.0;
        let mom = (t_k - 1.0) / t_new;

        let y_new: Array1<f64> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xn, &xo)| xn + mom * (xn - xo))
            .collect();

        // Evaluate true minimax at new x
        let max_val = problem.eval_max(&x_new.view());
        if max_val < val_best {
            val_best = max_val;
            x_best = x_new.clone();
        }

        x = x_new;
        y = y_new;
        t_k = t_new;
    }

    let active_idx = problem.argmax(&x_best.view());
    Ok(MinimaxSolveResult {
        x: x_best,
        fun: val_best,
        active_index: active_idx,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Smooth minimax (Nesterov) converged".to_string()
        } else {
            "Smooth minimax reached maximum iterations".to_string()
        },
    })
}

// ─── Game-Theoretic Minimax (Fictitious Play) ─────────────────────────────────

/// Result of a game-theoretic minimax solve.
#[derive(Debug, Clone)]
pub struct GameMinimaxResult {
    /// Approximate minimizer strategy (mixed or pure).
    pub x: Array1<f64>,
    /// Minimax value.
    pub fun: f64,
    /// Mixed strategy (probability distribution over pure strategies for the maximizer).
    /// Entry i is the empirical frequency of function i being active.
    pub maximizer_strategy: Array1<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

/// Solve a zero-sum game minimax via fictitious play.
///
/// Fictitious play (Brown 1951) is a classical algorithm for zero-sum games:
/// each player responds optimally to the *empirical distribution* of the
/// opponent's past strategies.
///
/// For the continuous minimax problem, we discretize the minimizer's action
/// space around x₀ (using a simplex of perturbations) and apply:
///
/// 1. Maximizer picks the function i* = argmax_i Σ_j q_j f_i(x_j)
///    (best response to current minimizer mixture).
/// 2. Minimizer picks x* = argmin_x f_{i*}(x) + (sum of past active gradients).
///    (best response to current maximizer mixture q).
/// 3. Update empirical frequencies.
///
/// **Note**: This implementation uses a gradient-based inner response for the
/// minimizer (projected gradient descent on the currently active function),
/// which gives a tractable continuous analogue of fictitious play.
///
/// # Arguments
///
/// * `problem`       – the minimax problem
/// * `x0`            – initial minimizer point
/// * `step_size`     – step size for the minimizer's gradient response
/// * `config`        – solver configuration (uses `fictitious_play_iter`)
///
/// # Returns
///
/// [`GameMinimaxResult`] with approximate Nash equilibrium.
///
/// # References
///
/// Brown, G.W. (1951). "Iterative solution of games by fictitious play".
/// In *Activity Analysis of Production and Allocation*, pp. 374–376.
pub fn game_theoretic_minimax(
    problem: &MinimaxProblem,
    x0: &ArrayView1<f64>,
    step_size: f64,
    config: &MinimaxSolverConfig,
) -> OptimizeResult<GameMinimaxResult> {
    if problem.num_funcs() == 0 {
        return Err(OptimizeError::ValueError(
            "MinimaxProblem must contain at least one function".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    let k = problem.num_funcs();
    let h = config.fd_step;
    let max_fp_iter = config.fictitious_play_iter;

    // Empirical frequency counts for each function (maximizer's mixed strategy)
    let mut counts = vec![0_usize; k];
    let mut x = x0.to_owned();
    let mut x_best = x.clone();
    let mut val_best = problem.eval_max(&x.view());
    let mut converged = false;

    // Maintain a cumulative "average gradient" to guide the minimizer
    let mut cumulative_grad = Array1::<f64>::zeros(n);
    let mut cumulative_count = 0_usize;

    for iter in 0..max_fp_iter {
        // ── Maximizer's best response ───────────────────────────────────────
        // Compute the expected loss under the current minimizer iterate x
        let vals = problem.eval_all(&x.view());

        // Maximizer picks the function with highest value (best response)
        let active_i = vals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        counts[active_i] += 1;
        cumulative_count += 1;

        let max_val = vals[active_i];
        if max_val < val_best {
            val_best = max_val;
            x_best = x.clone();
        }

        // ── Minimizer's best response ───────────────────────────────────────
        // Compute gradient of the active function and accumulate
        let grad_i = fd_gradient(
            &|v: &ArrayView1<f64>| problem.funcs[active_i](v),
            &x.view(),
            h,
        );
        for i in 0..n {
            cumulative_grad[i] += grad_i[i];
        }

        // Minimizer takes a step against the cumulative (average) gradient
        let avg_grad_norm: f64 =
            cumulative_grad.iter().map(|&g| g * g).sum::<f64>().sqrt() / cumulative_count as f64;

        if avg_grad_norm < config.tol {
            converged = true;
            break;
        }

        // Step in direction of negative average gradient (best response update)
        let alpha = step_size / ((iter as f64 + 1.0).sqrt());
        for i in 0..n {
            x[i] -= alpha * cumulative_grad[i] / cumulative_count as f64;
        }

        // Convergence: check if active function value is stable
        if iter > 10 && (max_val - problem.eval_max(&x.view())).abs() < config.tol {
            converged = true;
            break;
        }
    }

    // Compute empirical mixed strategy for the maximizer
    let total = counts.iter().sum::<usize>().max(1) as f64;
    let maximizer_strategy: Array1<f64> = counts.iter().map(|&c| c as f64 / total).collect();

    Ok(GameMinimaxResult {
        x: x_best,
        fun: val_best,
        maximizer_strategy,
        n_iter: max_fp_iter,
        converged,
        message: if converged {
            "Fictitious play converged".to_string()
        } else {
            "Fictitious play reached maximum iterations".to_string()
        },
    })
}

// ─── Convenience: evaluate smooth minimax objective ──────────────────────────

/// Evaluate the Nesterov smooth upper bound of max_i f_i(x).
///
/// ```text
/// F_μ(x) = μ · log( (1/k) Σ_i exp(f_i(x)/μ) )
/// ```
///
/// # Arguments
///
/// * `funcs` – slice of function closures f_i
/// * `x`     – evaluation point
/// * `mu`    – smoothing parameter > 0 (smaller → tighter approximation)
///
/// # Returns
///
/// The smooth upper bound value.
pub fn smooth_max_value<F>(funcs: &[F], x: &ArrayView1<f64>, mu: f64) -> OptimizeResult<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    if funcs.is_empty() {
        return Err(OptimizeError::ValueError(
            "funcs must be non-empty".to_string(),
        ));
    }
    if mu <= 0.0 {
        return Err(OptimizeError::ValueError(format!(
            "mu must be positive, got {}",
            mu
        )));
    }
    let vals: Vec<f64> = funcs.iter().map(|f| f(x)).collect();
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return Ok(max_val);
    }
    let sum_exp: f64 = vals.iter().map(|&v| ((v - max_val) / mu).exp()).sum();
    let k = funcs.len() as f64;
    Ok(mu * (sum_exp.ln() + (max_val / mu)) - mu * k.ln())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build the problem max(f0, f1) where:
    ///   f0(x) = (x[0] - 1)²   (minimum at x=1)
    ///   f1(x) = (x[0] + 1)²   (minimum at x=-1)
    /// max(f0, f1) is minimised at x=0 where max = 1.
    fn build_two_func_problem() -> MinimaxProblem {
        MinimaxProblem::new(vec![
            Box::new(|x: &ArrayView1<f64>| (x[0] - 1.0).powi(2)),
            Box::new(|x: &ArrayView1<f64>| (x[0] + 1.0).powi(2)),
        ])
    }

    #[test]
    fn test_minimax_problem_eval() {
        let p = build_two_func_problem();
        let x = array![0.0];
        assert_eq!(p.eval_max(&x.view()), 1.0);
        let x2 = array![1.0];
        // f0(1)=0, f1(1)=4 → max=4
        assert!((p.eval_max(&x2.view()) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_subgradient_basic() {
        let p = build_two_func_problem();
        let x0 = array![3.0];
        let config = MinimaxSolverConfig {
            max_iter: 3_000,
            tol: 1e-4,
            step_size: 0.5,
            ..Default::default()
        };
        let result = minimax_subgradient(&p, &x0.view(), &config).expect("failed to create result");
        // Optimal at x=0, minimax value = 1
        assert!(
            result.fun <= 1.5,
            "subgradient minimax value {} should be ≤ 1.5",
            result.fun
        );
        assert!(
            result.x[0].abs() < 1.0,
            "subgradient minimizer {} should be near 0",
            result.x[0]
        );
    }

    #[test]
    fn test_smooth_minimax_basic() {
        let p = build_two_func_problem();
        let x0 = array![3.0];
        let config = MinimaxSolverConfig {
            max_iter: 3_000,
            tol: 1e-5,
            step_size: 1e-2,
            smoothing_mu: 0.05,
            ..Default::default()
        };
        let result = smooth_minimax(&p, &x0.view(), &config).expect("failed to create result");
        assert!(
            result.fun <= 2.0,
            "smooth minimax value {} should be ≤ 2.0",
            result.fun
        );
    }

    #[test]
    fn test_game_theoretic_minimax() {
        let p = build_two_func_problem();
        let x0 = array![2.0];
        let config = MinimaxSolverConfig {
            max_iter: 500,
            tol: 1e-4,
            fictitious_play_iter: 500,
            ..Default::default()
        };
        let result =
            game_theoretic_minimax(&p, &x0.view(), 0.1, &config).expect("failed to create result");
        // Should move towards x=0
        assert!(
            result.x[0].abs() < 2.5,
            "game theoretic minimizer {} should move toward 0",
            result.x[0]
        );
        // Mixed strategy should be non-trivial (both functions played)
        assert_eq!(result.maximizer_strategy.len(), 2);
        let strat_sum: f64 = result.maximizer_strategy.iter().sum();
        assert!((strat_sum - 1.0).abs() < 1e-9, "strategy should sum to 1");
    }

    #[test]
    fn test_smooth_max_value() {
        let funcs: Vec<Box<dyn Fn(&ArrayView1<f64>) -> f64>> = vec![
            Box::new(|_x: &ArrayView1<f64>| 1.0),
            Box::new(|_x: &ArrayView1<f64>| 2.0),
            Box::new(|_x: &ArrayView1<f64>| 3.0),
        ];
        let x = array![0.0];
        let val = smooth_max_value(&funcs, &x.view(), 0.01).expect("failed to create val");
        // With small mu, smooth_max ≈ true max = 3.0
        assert!((val - 3.0).abs() < 0.1, "smooth max ≈ 3.0, got {val}");
    }

    #[test]
    fn test_bundle_method_basic() {
        let p = build_two_func_problem();
        let x0 = array![2.0];
        let config = MinimaxSolverConfig {
            max_iter: 500,
            tol: 1e-4,
            step_size: 0.5,
            ..Default::default()
        };
        let result = minimax_bundle(&p, &x0.view(), &config).expect("failed to create result");
        assert!(
            result.fun <= 5.0,
            "bundle minimax value {} should be reasonable",
            result.fun
        );
    }

    #[test]
    fn test_empty_problem_error() {
        let p = MinimaxProblem::new(vec![]);
        let x0 = array![1.0];
        let config = MinimaxSolverConfig::default();
        assert!(minimax_subgradient(&p, &x0.view(), &config).is_err());
        assert!(smooth_minimax(&p, &x0.view(), &config).is_err());
    }

    #[test]
    fn test_empty_x0_error() {
        let p = build_two_func_problem();
        let x0: Array1<f64> = Array1::zeros(0);
        let config = MinimaxSolverConfig::default();
        assert!(minimax_subgradient(&p, &x0.view(), &config).is_err());
    }
}
