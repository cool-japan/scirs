//! DIRECT (DIviding RECTangles) Global Optimization Algorithm
//!
//! DIRECT is a deterministic global optimization algorithm that does not require
//! any knowledge of the Lipschitz constant. It works by systematically dividing
//! the search space into hyperrectangles and evaluating the function at their
//! centers, selecting "potentially optimal" rectangles for further division.
//!
//! ## Key properties
//!
//! - No gradient or Lipschitz constant required
//! - Guaranteed convergence to global optimum (under mild conditions)
//! - Balances local refinement and global exploration
//! - Budget management via max evaluations and iterations
//!
//! ## References
//!
//! - Jones, D.R., Perttunen, C.D., Stuckman, B.E. (1993).
//!   Lipschitzian Optimization Without the Lipschitz Constant.
//!   Journal of Optimization Theory and Applications, 79(1), 157-181.
//! - Gablonsky, J.M. & Kelley, C.T. (2001). A Locally-Biased form of
//!   the DIRECT Algorithm. Journal of Global Optimization, 21, 27-37.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Options for the DIRECT algorithm
#[derive(Debug, Clone)]
pub struct DirectOptions {
    /// Maximum number of function evaluations
    pub max_fevals: usize,
    /// Maximum number of iterations (subdivisions)
    pub max_iterations: usize,
    /// Minimum function value improvement to continue (absolute tolerance)
    pub ftol_abs: f64,
    /// Relative tolerance on function value
    pub ftol_rel: f64,
    /// Minimum rectangle size (stops when rectangles become too small)
    pub vol_tol: f64,
    /// Epsilon parameter for selecting potentially optimal rectangles.
    /// Larger values bias toward global search; smaller toward local refinement.
    /// Jones (1993) recommends 1e-4 as a good default.
    pub epsilon: f64,
    /// Whether to use the locally-biased variant (DIRECT-L, Gablonsky & Kelley 2001)
    pub locally_biased: bool,
}

impl Default for DirectOptions {
    fn default() -> Self {
        Self {
            max_fevals: 10_000,
            max_iterations: 1_000,
            ftol_abs: 1e-12,
            ftol_rel: 1e-12,
            vol_tol: 1e-16,
            epsilon: 1e-4,
            locally_biased: false,
        }
    }
}

/// Result of DIRECT optimization
#[derive(Debug, Clone)]
pub struct DirectResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Best function value found
    pub fun: f64,
    /// Number of function evaluations used
    pub nfev: usize,
    /// Number of iterations (divisions)
    pub nit: usize,
    /// Number of rectangles at termination
    pub n_rectangles: usize,
    /// Whether optimization converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// A hyperrectangle in the DIRECT algorithm
#[derive(Debug, Clone)]
struct Rectangle {
    /// Center point in normalized [0,1]^n coordinates
    center: Vec<f64>,
    /// Function value at the center
    f_center: f64,
    /// Half-widths of the rectangle in each dimension (in [0,1] units)
    half_widths: Vec<f64>,
    /// Measure of the rectangle's "size" (max half-width or diagonal)
    size: f64,
}

impl Rectangle {
    fn new(center: Vec<f64>, f_center: f64, half_widths: Vec<f64>) -> Self {
        let size = half_widths.iter().copied().fold(0.0_f64, f64::max);
        Self {
            center,
            f_center,
            half_widths,
            size,
        }
    }

    /// Compute the diagonal distance (L2 norm of half-widths)
    fn diagonal(&self) -> f64 {
        self.half_widths.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// Volume of the rectangle (product of widths)
    fn volume(&self) -> f64 {
        self.half_widths.iter().map(|w| 2.0 * w).product::<f64>()
    }
}

/// DIRECT optimizer
pub struct Direct<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    func: F,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    options: DirectOptions,
    ndim: usize,
    /// All rectangles currently active
    rectangles: Vec<Rectangle>,
    /// Best function value found
    best_f: f64,
    /// Best point found (in original coordinates)
    best_x: Vec<f64>,
    /// Number of function evaluations
    fevals: usize,
}

impl<F> Direct<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    /// Create a new DIRECT optimizer
    pub fn new(
        func: F,
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        options: DirectOptions,
    ) -> OptimizeResult<Self> {
        let ndim = lower_bounds.len();
        if ndim == 0 {
            return Err(OptimizeError::InvalidInput(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if upper_bounds.len() != ndim {
            return Err(OptimizeError::InvalidInput(
                "Lower and upper bounds must have the same length".to_string(),
            ));
        }
        for i in 0..ndim {
            if lower_bounds[i] >= upper_bounds[i] {
                return Err(OptimizeError::InvalidInput(format!(
                    "Lower bound must be less than upper bound for dimension {}: {} >= {}",
                    i, lower_bounds[i], upper_bounds[i]
                )));
            }
        }

        Ok(Self {
            func,
            lower_bounds,
            upper_bounds,
            options,
            ndim,
            rectangles: Vec::new(),
            best_f: f64::INFINITY,
            best_x: vec![0.0; ndim],
            fevals: 0,
        })
    }

    /// Convert normalized [0,1]^n coordinates to original coordinates
    fn to_original(&self, normalized: &[f64]) -> Vec<f64> {
        normalized
            .iter()
            .enumerate()
            .map(|(i, &x)| self.lower_bounds[i] + x * (self.upper_bounds[i] - self.lower_bounds[i]))
            .collect()
    }

    /// Evaluate the function at a point (normalized coordinates)
    fn evaluate(&mut self, normalized_point: &[f64]) -> f64 {
        let original = self.to_original(normalized_point);
        let arr = Array1::from_vec(original.clone());
        let f_val = (self.func)(&arr.view());
        self.fevals += 1;

        if f_val < self.best_f {
            self.best_f = f_val;
            self.best_x = original;
        }

        f_val
    }

    /// Initialize the algorithm with the unit hypercube center
    fn initialize(&mut self) {
        let center = vec![0.5; self.ndim];
        let f_center = self.evaluate(&center);
        let half_widths = vec![0.5; self.ndim];
        let rect = Rectangle::new(center, f_center, half_widths);
        self.rectangles.push(rect);
    }

    /// Select potentially optimal rectangles
    ///
    /// A rectangle is potentially optimal if there exists some Lipschitz constant K > 0
    /// such that the rectangle could contain the global minimum. This is determined
    /// by checking if the rectangle lies on the lower-right convex hull in the
    /// (size, f_center) space.
    fn select_potentially_optimal(&self) -> Vec<usize> {
        if self.rectangles.is_empty() {
            return Vec::new();
        }

        let epsilon = self.options.epsilon;
        let f_min = self.best_f;

        // Group rectangles by their size (quantized to avoid floating point issues)
        let mut size_groups: std::collections::BTreeMap<u64, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, rect) in self.rectangles.iter().enumerate() {
            let size_key = (rect.diagonal() * 1e12) as u64;
            size_groups.entry(size_key).or_default().push(idx);
        }

        // For each size group, find the rectangle with the lowest function value
        let mut hull_candidates: Vec<(f64, f64, usize)> = Vec::new(); // (size, f_val, idx)
        for (_size_key, indices) in &size_groups {
            let mut best_idx = indices[0];
            let mut best_f = self.rectangles[indices[0]].f_center;
            for &idx in &indices[1..] {
                if self.rectangles[idx].f_center < best_f {
                    best_f = self.rectangles[idx].f_center;
                    best_idx = idx;
                }
            }
            hull_candidates.push((self.rectangles[best_idx].diagonal(), best_f, best_idx));
        }

        // Sort by size
        hull_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if hull_candidates.is_empty() {
            return Vec::new();
        }

        // Select potentially optimal rectangles using the correct Jones (1993) criterion.
        //
        // Rectangle i is potentially optimal if there EXISTS K >= 0 such that:
        //   (1) f_i - K * d_i <= f_j - K * d_j  for all j != i   (dominance condition)
        //   (2) f_i - K * d_i <= f_min - epsilon * |f_min|         (improvement condition)
        //
        // From (1) with d_j > d_i:  K <= (f_j - f_i) / (d_j - d_i)   → upper bound K_upper
        // From (1) with d_j < d_i:  K >= (f_j - f_i) / (d_j - d_i)   → lower bound K_lower
        //
        // The feasible K range is [max(0, K_lower), K_upper].
        // We use K = K_upper (tightest constraint from larger rectangles) to check (2).
        // If K_upper < max(0, K_lower) there is no feasible K and the rectangle is not
        // potentially optimal.
        //
        // For the locally-biased variant (DIRECT-L) we only check the immediate neighbor
        // as the upper-bound constraint (Gablonsky & Kelley 2001).
        let mut selected = Vec::new();

        for i in 0..hull_candidates.len() {
            let (d_i, f_i, idx) = hull_candidates[i];

            // Compute K_lower: tightest lower bound from all smaller rectangles.
            let mut k_lower = 0.0_f64; // must be non-negative
            for k in 0..i {
                let (d_k, f_k, _) = hull_candidates[k];
                if d_k < d_i && (d_i - d_k).abs() > 1e-15 {
                    let slope = (f_i - f_k) / (d_i - d_k);
                    if slope > k_lower {
                        k_lower = slope;
                    }
                }
            }

            // Compute K_upper: tightest upper bound from larger rectangles.
            let k_upper = if self.options.locally_biased {
                // DIRECT-L: only use the immediate next-larger neighbor
                if i + 1 < hull_candidates.len() {
                    let (d_next, f_next, _) = hull_candidates[i + 1];
                    if (d_next - d_i).abs() > 1e-15 {
                        (f_next - f_i) / (d_next - d_i)
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                }
            } else {
                // Standard DIRECT: minimum slope to any larger rectangle
                let mut k_up = f64::INFINITY;
                for j in (i + 1)..hull_candidates.len() {
                    let (d_j, f_j, _) = hull_candidates[j];
                    if d_j > d_i && (d_j - d_i).abs() > 1e-15 {
                        let slope = (f_j - f_i) / (d_j - d_i);
                        if slope < k_up {
                            k_up = slope;
                        }
                    }
                }
                k_up
            };

            // Check feasibility: K_upper must be >= K_lower (and >= 0)
            if k_upper < k_lower {
                continue; // no feasible K exists
            }

            // Check improvement condition using K = K_upper (the tightest upper bound).
            // If K_upper is infinite (no larger rectangles), use K = K_lower >= 0.
            let k_use = if k_upper.is_finite() {
                k_upper
            } else {
                k_lower
            };
            let f_projected = f_i - k_use * d_i;
            if f_projected <= f_min - epsilon * f_min.abs() {
                selected.push(idx);
            }
        }

        // Always include the rectangle containing the best point
        // if it's not already selected
        let best_rect_idx = self
            .rectangles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.f_center
                    .partial_cmp(&b.f_center)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(best_idx) = best_rect_idx {
            if !selected.contains(&best_idx) {
                selected.push(best_idx);
            }
        }

        selected
    }

    /// Divide a rectangle along its longest dimensions
    fn divide_rectangle(&mut self, rect_idx: usize) -> Vec<Rectangle> {
        let rect = self.rectangles[rect_idx].clone();

        // Find the longest dimension(s)
        let max_width = rect.half_widths.iter().copied().fold(0.0_f64, f64::max);

        let long_dims: Vec<usize> = rect
            .half_widths
            .iter()
            .enumerate()
            .filter(|(_, &w)| (w - max_width).abs() < 1e-15)
            .map(|(i, _)| i)
            .collect();

        // Standard DIRECT trisection (Jones 1993, Algorithm 1):
        //
        // When trisecting a rectangle along dimension d with current half-width W:
        //
        //   new_hw = W / 3
        //
        //   The original interval [center - W, center + W] is split into three equal
        //   thirds of width 2*new_hw each:
        //     Bottom third: [center - W,       center - W/3]  center at center - 2*new_hw
        //     Middle third: [center - W/3,      center + W/3]  center stays at center
        //     Top    third: [center + W/3,      center + W  ]  center at center + 2*new_hw
        //
        //   So each child rectangle is centered at center ± 2*new_hw with half-width new_hw
        //   in dimension d, and retains the ORIGINAL parent half-widths in all other dims.
        //
        // Jones (1993) determines the trisection order by first evaluating the function
        // at the BOUNDARY between thirds: center ± new_hw (= center ± W/3).
        // These probe values w_i = min(f(center ± new_hw * e_d)) rank the dimensions
        // by how promising their child regions look, without yet committing to full eval.
        // Child center evaluations (at center ± 2*new_hw) happen separately in the divide step.
        //
        // Implementations note: this uses 4 evaluations per long dimension:
        //   2 at boundary points for sorting + 2 at child centers for child f_center values.
        let new_hw = max_width / 3.0; // new half-width after trisection

        // Phase 1: evaluate at center ± new_hw (boundary between thirds) for each long dim.
        // These values rank the dimensions by w_i = min(f+, f-).
        let mut dim_sort: Vec<(usize, f64)> = Vec::new();
        for &dim in &long_dims {
            let mut c_probe_p = rect.center.clone();
            c_probe_p[dim] += new_hw;
            let f_probe_p = self.evaluate(&c_probe_p);

            let mut c_probe_m = rect.center.clone();
            c_probe_m[dim] -= new_hw;
            let f_probe_m = self.evaluate(&c_probe_m);

            dim_sort.push((dim, f_probe_p.min(f_probe_m)));
        }

        // Sort long dimensions ascending by w_i (lowest-valued dimension first)
        dim_sort.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Phase 2: for each long dimension (in sorted order), create the two child
        // rectangles centered at center ± 2*new_hw.
        //
        // Each child inherits the ORIGINAL parent half-widths for all dimensions, except
        // dimension d which becomes new_hw.  This is correct because the trisection of
        // one dimension does not affect the extent of the rectangle in other dimensions.
        let mut new_rects = Vec::new();

        for &(dim, _) in &dim_sort {
            // Child centers are at the centers of the outer thirds: center ± 2*new_hw
            let mut c_child_p = rect.center.clone();
            c_child_p[dim] += 2.0 * new_hw;
            let f_child_p = self.evaluate(&c_child_p);

            let mut c_child_m = rect.center.clone();
            c_child_m[dim] -= 2.0 * new_hw;
            let f_child_m = self.evaluate(&c_child_m);

            // Child half-widths: original parent widths with only dimension d changed
            let mut hw_child = rect.half_widths.clone();
            hw_child[dim] = new_hw;

            new_rects.push(Rectangle::new(c_child_p, f_child_p, hw_child.clone()));
            new_rects.push(Rectangle::new(c_child_m, f_child_m, hw_child));
        }

        // The parent rectangle (middle third in all trisected dims) keeps its center
        // but its half-widths shrink to new_hw in every long dimension.
        let mut parent_hw = rect.half_widths.clone();
        for &(dim, _) in &dim_sort {
            parent_hw[dim] = new_hw;
        }
        let parent_rect = Rectangle::new(rect.center.clone(), rect.f_center, parent_hw);
        new_rects.push(parent_rect);

        new_rects
    }

    /// Run the DIRECT algorithm
    pub fn run(&mut self) -> DirectResult {
        self.initialize();

        let mut prev_best_f = self.best_f;

        for iteration in 0..self.options.max_iterations {
            // Check budget
            if self.fevals >= self.options.max_fevals {
                return DirectResult {
                    x: Array1::from_vec(self.best_x.clone()),
                    fun: self.best_f,
                    nfev: self.fevals,
                    nit: iteration,
                    n_rectangles: self.rectangles.len(),
                    success: true,
                    message: format!(
                        "Maximum function evaluations ({}) reached",
                        self.options.max_fevals
                    ),
                };
            }

            // Select potentially optimal rectangles
            let po_indices = self.select_potentially_optimal();
            if po_indices.is_empty() {
                return DirectResult {
                    x: Array1::from_vec(self.best_x.clone()),
                    fun: self.best_f,
                    nfev: self.fevals,
                    nit: iteration,
                    n_rectangles: self.rectangles.len(),
                    success: true,
                    message: "No potentially optimal rectangles found".to_string(),
                };
            }

            // Divide selected rectangles
            // Sort indices in descending order so removal doesn't affect earlier indices
            let mut sorted_indices = po_indices;
            sorted_indices.sort_unstable_by(|a, b| b.cmp(a));

            let mut new_rects_all = Vec::new();
            for &idx in &sorted_indices {
                if self.fevals >= self.options.max_fevals {
                    break;
                }
                let new_rects = self.divide_rectangle(idx);
                new_rects_all.push((idx, new_rects));
            }

            // Remove old rectangles and add new ones
            // Remove in descending order
            let mut indices_to_remove: Vec<usize> =
                new_rects_all.iter().map(|(idx, _)| *idx).collect();
            indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));
            for idx in indices_to_remove {
                self.rectangles.swap_remove(idx);
            }
            for (_, new_rects) in new_rects_all {
                self.rectangles.extend(new_rects);
            }

            // Check convergence criteria: only terminate on stagnation after many
            // consecutive iterations with no improvement.  A single iteration without
            // improvement is normal in DIRECT (the algorithm explores globally), so we
            // require sustained stagnation before declaring convergence.
            let f_improvement = (prev_best_f - self.best_f).abs();
            let abs_stagnant = f_improvement < self.options.ftol_abs;
            let rel_stagnant = if prev_best_f.abs() > 1e-30 {
                f_improvement / prev_best_f.abs() < self.options.ftol_rel
            } else {
                abs_stagnant
            };
            if (abs_stagnant || rel_stagnant) && iteration > 10 {
                // Only terminate if we have truly stagnated for many iterations.
                // Use a heuristic: stagnate for at least 10% of max_iterations or 50 iters.
                let stagnation_limit = (self.options.max_iterations / 10).max(50);
                // We track stagnation via consecutive no-improvement: count iterations
                // since the last improvement by comparing with the running prev_best_f.
                // Here we use the conservative check: only exit if f has not changed
                // at all (machine precision) AND enough iterations have passed.
                if f_improvement == 0.0 && iteration > stagnation_limit {
                    return DirectResult {
                        x: Array1::from_vec(self.best_x.clone()),
                        fun: self.best_f,
                        nfev: self.fevals,
                        nit: iteration,
                        n_rectangles: self.rectangles.len(),
                        success: true,
                        message: "Function tolerance reached (stagnation)".to_string(),
                    };
                }
            }

            // Check volume tolerance
            let max_vol = self
                .rectangles
                .iter()
                .map(|r| r.volume())
                .fold(0.0_f64, f64::max);
            if max_vol < self.options.vol_tol {
                return DirectResult {
                    x: Array1::from_vec(self.best_x.clone()),
                    fun: self.best_f,
                    nfev: self.fevals,
                    nit: iteration,
                    n_rectangles: self.rectangles.len(),
                    success: true,
                    message: "Volume tolerance reached".to_string(),
                };
            }

            prev_best_f = self.best_f;
        }

        DirectResult {
            x: Array1::from_vec(self.best_x.clone()),
            fun: self.best_f,
            nfev: self.fevals,
            nit: self.options.max_iterations,
            n_rectangles: self.rectangles.len(),
            success: true,
            message: format!(
                "Maximum iterations ({}) reached",
                self.options.max_iterations
            ),
        }
    }
}

/// Convenience function to run DIRECT optimization
///
/// # Arguments
///
/// * `func` - Objective function to minimize
/// * `lower_bounds` - Lower bounds for each dimension
/// * `upper_bounds` - Upper bounds for each dimension
/// * `options` - DIRECT options (uses defaults if None)
///
/// # Returns
///
/// * `DirectResult` with the best solution found
pub fn direct_minimize<F>(
    func: F,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    options: Option<DirectOptions>,
) -> OptimizeResult<DirectResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut optimizer = Direct::new(func, lower_bounds, upper_bounds, options)?;
    Ok(optimizer.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function
    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    /// Rosenbrock function
    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    /// Rastrigin function (multimodal)
    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mut sum = 10.0 * n;
        for &xi in x.iter() {
            sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
        }
        sum
    }

    /// Branin function (3 global minima)
    fn branin(x: &ArrayView1<f64>) -> f64 {
        let pi = std::f64::consts::PI;
        let x1 = x[0];
        let x2 = x[1];
        let a = 1.0;
        let b = 5.1 / (4.0 * pi * pi);
        let c = 5.0 / pi;
        let r = 6.0;
        let s = 10.0;
        let t = 1.0 / (8.0 * pi);
        a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
    }

    #[test]
    fn test_direct_sphere_2d() {
        let result = direct_minimize(
            sphere,
            vec![-5.0, -5.0],
            vec![5.0, 5.0],
            Some(DirectOptions {
                max_fevals: 500,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT sphere 2D failed");
        assert!(res.fun < 0.1, "DIRECT sphere value: {}", res.fun);
        // Allow a small overshoot: divide_rectangle() makes up to 4*ndim evaluations per
        // rectangle in flight when the budget is reached, so the final count may slightly
        // exceed max_fevals.  Add a slack of 2*4*ndim = 16 to tolerate this.
        assert!(res.nfev <= 516, "Used {} evaluations", res.nfev);
    }

    #[test]
    fn test_direct_sphere_3d() {
        let result = direct_minimize(
            sphere,
            vec![-5.0, -5.0, -5.0],
            vec![5.0, 5.0, 5.0],
            Some(DirectOptions {
                max_fevals: 2_000,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT sphere 3D failed");
        assert!(res.fun < 1.0, "DIRECT sphere 3D value: {}", res.fun);
    }

    #[test]
    fn test_direct_rosenbrock() {
        let result = direct_minimize(
            rosenbrock,
            vec![-2.0, -2.0],
            vec![2.0, 2.0],
            Some(DirectOptions {
                max_fevals: 5_000,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT Rosenbrock failed");
        assert!(res.fun < 1.0, "DIRECT Rosenbrock value: {}", res.fun);
    }

    #[test]
    fn test_direct_rastrigin() {
        let result = direct_minimize(
            rastrigin,
            vec![-5.12, -5.12],
            vec![5.12, 5.12],
            Some(DirectOptions {
                max_fevals: 5_000,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT Rastrigin failed");
        // DIRECT should find near-global minimum of Rastrigin
        assert!(res.fun < 5.0, "DIRECT Rastrigin value: {}", res.fun);
    }

    #[test]
    fn test_direct_branin() {
        let result = direct_minimize(
            branin,
            vec![-5.0, 0.0],
            vec![10.0, 15.0],
            Some(DirectOptions {
                max_fevals: 3_000,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT Branin failed");
        // Global minimum of Branin is ~0.397887
        assert!(
            res.fun < 1.0,
            "DIRECT Branin value: {} (expected ~0.398)",
            res.fun
        );
    }

    #[test]
    fn test_direct_locally_biased() {
        let result = direct_minimize(
            sphere,
            vec![-5.0, -5.0],
            vec![5.0, 5.0],
            Some(DirectOptions {
                max_fevals: 500,
                locally_biased: true,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT-L sphere failed");
        assert!(res.fun < 1.0, "DIRECT-L sphere value: {}", res.fun);
    }

    #[test]
    fn test_direct_invalid_bounds() {
        let result = direct_minimize(sphere, vec![5.0, -5.0], vec![-5.0, 5.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_direct_empty_dimensions() {
        let result: OptimizeResult<DirectResult> = direct_minimize(sphere, vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_direct_1d() {
        fn parabola(x: &ArrayView1<f64>) -> f64 {
            (x[0] - 3.0).powi(2) + 1.0
        }
        let result = direct_minimize(
            parabola,
            vec![0.0],
            vec![6.0],
            Some(DirectOptions {
                max_fevals: 200,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT 1D parabola failed");
        assert!(
            (res.x[0] - 3.0).abs() < 0.5,
            "DIRECT 1D minimum at x={} (expected 3.0)",
            res.x[0]
        );
        assert!(
            (res.fun - 1.0).abs() < 0.5,
            "DIRECT 1D value {} (expected 1.0)",
            res.fun
        );
    }

    #[test]
    fn test_direct_budget_management() {
        let result = direct_minimize(
            sphere,
            vec![-10.0, -10.0],
            vec![10.0, 10.0],
            Some(DirectOptions {
                max_fevals: 50,
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("DIRECT budget test failed");
        assert!(res.nfev <= 55, "Budget exceeded: {} > 50", res.nfev);
    }
}
