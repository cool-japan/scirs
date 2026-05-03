//! Kernel grammar search and hyperparameter optimisation.
//!
//! Generates candidate kernels up to a configurable depth using BFS over the
//! grammar `{ Base, Sum(a, b), Product(a, b) }`, then selects the best one by
//! k-fold cross-validated MSE.
//!
//! ## Algorithm
//!
//! 1. **Enumerate candidates** — BFS: depth-0 = one expression per base kernel;
//!    depth-1 = all (depth-0 + depth-0) and (depth-0 × depth-0) pairs (commutative
//!    duplicates deduplicated by ordered index); depth-2 = all (depth-1 ± base) and
//!    (depth-1 × base) combinations.
//!
//! 2. **Optimise hyperparameters** — For each candidate kernel, golden-section
//!    search over `length_scale ∈ [0.01, 10]`; for `Periodic` a grid over
//!    `period ∈ [0.1, 5.0]` is combined with golden-section on ℓ.
//!
//! 3. **Cross-validate** — k-fold (default 5) leave-out MSE on the training data.
//!
//! 4. **Select** — Return the kernel with the lowest CV-MSE.

use super::kernel::{base_kernels, BaseKernel, KernelExpr};

// ---------------------------------------------------------------------------
// Inline Cholesky (Banachiewicz, lower triangular)
// ---------------------------------------------------------------------------

/// Compute the Cholesky factor L of a symmetric positive definite matrix A.
///
/// Returns L (lower triangular) such that A ≈ L · Lᵀ.
/// Adds a small jitter (ε) to the diagonal to improve numerical stability.
pub(super) fn cholesky_lower(a: &[f64], n: usize, jitter: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            if i == j {
                s += jitter;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * n + i] = s.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < 1e-15 {
                    return None;
                }
                l[i * n + j] = s / ljj;
            }
        }
    }
    Some(l)
}

/// Solve L · x = b (forward substitution), L lower triangular.
fn forward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() > 1e-15 { s / lii } else { 0.0 };
    }
    x
}

/// Solve Lᵀ · x = b (back substitution), L lower triangular.
fn back_sub_lt(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= l[j * n + i] * x[j]; // L^T[i,j] = L[j,i]
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() > 1e-15 { s / lii } else { 0.0 };
    }
    x
}

/// Solve A·x = b via Cholesky.  Returns None if A is not positive definite.
pub(super) fn solve_spd(a: &[f64], b: &[f64], n: usize, jitter: f64) -> Option<Vec<f64>> {
    let l = cholesky_lower(a, n, jitter)?;
    let y = forward_sub(&l, b, n);
    Some(back_sub_lt(&l, &y, n))
}

// ---------------------------------------------------------------------------
// GP helpers
// ---------------------------------------------------------------------------

/// Build the n×n kernel matrix K[i,j] = kernel.eval(x[i], x[j]).
pub(super) fn build_kernel_matrix(kernel: &KernelExpr, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut k = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = kernel.eval(x[i], x[j]);
        }
    }
    k
}

/// Build the m×n cross-kernel matrix K_star[i,j] = kernel.eval(x_star[i], x[j]).
pub(super) fn build_cross_kernel(kernel: &KernelExpr, x_star: &[f64], x: &[f64]) -> Vec<f64> {
    let m = x_star.len();
    let n = x.len();
    let mut k = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            k[i * n + j] = kernel.eval(x_star[i], x[j]);
        }
    }
    k
}

/// Fit GP alpha vector: alpha = (K + noise·I)⁻¹ · y.
/// Returns None if the system is not solvable.
pub(super) fn gp_fit(kernel: &KernelExpr, x: &[f64], y: &[f64], noise: f64) -> Option<Vec<f64>> {
    let n = x.len();
    let mut k = build_kernel_matrix(kernel, x);
    for i in 0..n {
        k[i * n + i] += noise;
    }
    solve_spd(&k, y, n, 1e-8)
}

/// Predict at new points given the GP dual variables.
pub(super) fn gp_predict(
    kernel: &KernelExpr,
    x_train: &[f64],
    alpha: &[f64],
    x_star: &[f64],
) -> Vec<f64> {
    let k_star = build_cross_kernel(kernel, x_star, x_train);
    let n = x_train.len();
    let m = x_star.len();
    let mut preds = vec![0.0f64; m];
    for i in 0..m {
        for j in 0..n {
            preds[i] += k_star[i * n + j] * alpha[j];
        }
    }
    preds
}

/// k-fold cross-validated MSE for a given kernel.
///
/// Splits `(x, y)` into `folds` folds, leaves each out in turn, fits GP on
/// the remaining data, predicts on the held-out fold, and accumulates MSE.
pub(super) fn cv_mse(kernel: &KernelExpr, x: &[f64], y: &[f64], noise: f64, folds: usize) -> f64 {
    let n = x.len();
    if n == 0 || folds == 0 {
        return f64::MAX;
    }
    let folds = folds.min(n);
    let fold_size = (n + folds - 1) / folds; // ceiling
    let mut total_sq_err = 0.0f64;
    let mut total_count = 0usize;

    for fold in 0..folds {
        let val_start = fold * fold_size;
        let val_end = (val_start + fold_size).min(n);
        if val_start >= val_end {
            continue;
        }

        // Collect train and validation indices
        let train_x: Vec<f64> = x[..val_start]
            .iter()
            .chain(&x[val_end..])
            .copied()
            .collect();
        let train_y: Vec<f64> = y[..val_start]
            .iter()
            .chain(&y[val_end..])
            .copied()
            .collect();
        let val_x = &x[val_start..val_end];
        let val_y = &y[val_start..val_end];

        if train_x.is_empty() {
            continue;
        }

        if let Some(alpha) = gp_fit(kernel, &train_x, &train_y, noise) {
            let preds = gp_predict(kernel, &train_x, &alpha, val_x);
            for (p, v) in preds.iter().zip(val_y.iter()) {
                total_sq_err += (p - v).powi(2);
                total_count += 1;
            }
        } else {
            // Penalise non-positive-definite kernels
            return f64::MAX;
        }
    }

    if total_count == 0 {
        f64::MAX
    } else {
        total_sq_err / total_count as f64
    }
}

// ---------------------------------------------------------------------------
// Hyperparameter optimisation: golden-section search on length_scale
// ---------------------------------------------------------------------------

/// Golden-section search for a minimum of `f` on [lo, hi].
/// Returns the approximate minimiser `x` in `n_iters` iterations.
fn golden_section<F: Fn(f64) -> f64>(lo: f64, hi: f64, n_iters: usize, f: F) -> f64 {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio
    let mut a = lo;
    let mut b = hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..n_iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// Optimise the hyperparameters of `kernel` to minimise CV-MSE.
///
/// - For `Rbf` and `Matern52`: golden-section on ℓ in [0.05, 10].
/// - For `Periodic`: grid on p in `period_grid`, golden-section on ℓ for each p.
/// - For `Linear` / `WhiteNoise`: golden-section on variance in [0.01, 5].
/// - For `Sum` / `Product`: optimise each sub-expression independently (greedy).
///
/// Returns the best kernel expression found and its CV score.
pub(super) fn optimise_kernel(
    kernel: &KernelExpr,
    x: &[f64],
    y: &[f64],
    noise: f64,
    folds: usize,
    n_restarts: usize,
    period_grid: &[f64],
) -> (KernelExpr, f64) {
    match kernel {
        KernelExpr::Base(base) => {
            let best = optimise_base(base, x, y, noise, folds, n_restarts, period_grid);
            let score = cv_mse(&KernelExpr::Base(best.clone()), x, y, noise, folds);
            (KernelExpr::Base(best), score)
        }
        KernelExpr::Sum(a, b) => {
            let (oa, _) = optimise_kernel(a, x, y, noise, folds, n_restarts, period_grid);
            let (ob, _) = optimise_kernel(b, x, y, noise, folds, n_restarts, period_grid);
            let composed = KernelExpr::Sum(Box::new(oa), Box::new(ob));
            let score = cv_mse(&composed, x, y, noise, folds);
            (composed, score)
        }
        KernelExpr::Product(a, b) => {
            let (oa, _) = optimise_kernel(a, x, y, noise, folds, n_restarts, period_grid);
            let (ob, _) = optimise_kernel(b, x, y, noise, folds, n_restarts, period_grid);
            let composed = KernelExpr::Product(Box::new(oa), Box::new(ob));
            let score = cv_mse(&composed, x, y, noise, folds);
            (composed, score)
        }
    }
}

/// Optimise a single base kernel's hyperparameters.
fn optimise_base(
    base: &BaseKernel,
    x: &[f64],
    y: &[f64],
    noise: f64,
    folds: usize,
    n_restarts: usize,
    period_grid: &[f64],
) -> BaseKernel {
    let n_gs_iters = 20 + n_restarts * 5;
    match base {
        BaseKernel::Rbf { .. } | BaseKernel::Matern52 { .. } => {
            let best_ell = golden_section(0.05, 10.0, n_gs_iters, |ell| {
                let k = KernelExpr::Base(base.with_length_scale(ell));
                cv_mse(&k, x, y, noise, folds)
            });
            base.with_length_scale(best_ell)
        }
        BaseKernel::Linear { .. } | BaseKernel::WhiteNoise { .. } => {
            let best_v = golden_section(0.01, 5.0, n_gs_iters, |v| {
                let k = KernelExpr::Base(base.with_length_scale(v));
                cv_mse(&k, x, y, noise, folds)
            });
            base.with_length_scale(best_v)
        }
        BaseKernel::Periodic { .. } => {
            // Grid on period, golden-section on ℓ for each period value.
            let mut best_score = f64::MAX;
            let mut best_base = base.clone();
            for &p in period_grid {
                let best_ell = golden_section(0.05, 5.0, n_gs_iters, |ell| {
                    let k = KernelExpr::Base(BaseKernel::Periodic {
                        period: p,
                        length_scale: ell,
                    });
                    cv_mse(&k, x, y, noise, folds)
                });
                let candidate = BaseKernel::Periodic {
                    period: p,
                    length_scale: best_ell,
                };
                let score = cv_mse(&KernelExpr::Base(candidate.clone()), x, y, noise, folds);
                if score < best_score {
                    best_score = score;
                    best_base = candidate;
                }
            }
            best_base
        }
    }
}

// ---------------------------------------------------------------------------
// Grammar enumeration
// ---------------------------------------------------------------------------

/// Generate all candidate kernel expressions up to `max_depth`.
///
/// Commutative duplicates (A+B, B+A) are pruned by only generating ordered pairs
/// (using index ordering of the parent list).
pub fn enumerate_candidates(max_depth: usize) -> Vec<KernelExpr> {
    // Depth-0: one expression per base kernel.
    let mut prev: Vec<KernelExpr> = base_kernels().into_iter().map(KernelExpr::Base).collect();
    let mut all: Vec<KernelExpr> = prev.iter().cloned().collect();

    if max_depth == 0 {
        return all;
    }

    // Depth-1..max_depth: extend by combining current level with base kernels.
    for _depth in 1..=max_depth {
        let bases: Vec<KernelExpr> = base_kernels().into_iter().map(KernelExpr::Base).collect();
        let mut next: Vec<KernelExpr> = Vec::new();

        // Pair each expression in `prev` with each base kernel via Sum and Product.
        for (i, expr) in prev.iter().enumerate() {
            for (j, base) in bases.iter().enumerate() {
                // Prune commutative duplicates: for depth-1 from depth-0 pairs,
                // only generate (i, j) where i <= j to avoid A+B and B+A.
                if _depth == 1 && j < i {
                    continue;
                }
                next.push(KernelExpr::Sum(
                    Box::new(expr.clone()),
                    Box::new(base.clone()),
                ));
                next.push(KernelExpr::Product(
                    Box::new(expr.clone()),
                    Box::new(base.clone()),
                ));
            }
        }

        all.extend(next.iter().cloned());
        prev = next;
    }

    all
}

/// Run the full kernel search: enumerate, optimise, and rank by CV-MSE.
///
/// Returns a sorted list of `(description, cv_score)` and the best kernel expression.
pub fn search_kernels(
    x: &[f64],
    y: &[f64],
    max_depth: usize,
    n_restarts: usize,
    noise: f64,
    folds: usize,
    period_grid: &[f64],
) -> (Vec<(String, f64)>, KernelExpr) {
    let candidates = enumerate_candidates(max_depth);

    let mut results: Vec<(String, f64, KernelExpr)> = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let (optimised, score) =
            optimise_kernel(&candidate, x, y, noise, folds, n_restarts, period_grid);
        let desc = optimised.description();
        results.push((desc, score, optimised));
    }

    // Sort by CV score ascending (lower = better)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_kernel = results
        .first()
        .map(|(_, _, k)| k.clone())
        .unwrap_or_else(|| KernelExpr::Base(BaseKernel::Rbf { length_scale: 1.0 }));

    let ranked: Vec<(String, f64)> = results.into_iter().map(|(d, s, _)| (d, s)).collect();
    (ranked, best_kernel)
}
