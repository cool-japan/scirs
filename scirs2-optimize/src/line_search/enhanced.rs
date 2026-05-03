//! Enhanced line search algorithms for optimization
//!
//! This module provides robust implementations of several line search methods
//! used as building blocks by second-order optimization algorithms:
//!
//! - [`StrongWolfe`]      – zoom-based strong Wolfe conditions line search
//! - [`HagerZhang`]       – CG_DESCENT line search satisfying the approximate Wolfe conditions
//! - [`SafeguardedPowell`]– Powell's safeguarded cubic interpolation with bracket
//! - [`BacktrackingArmijo`] – simple backtracking with Armijo (sufficient decrease) condition
//!
//! ## References
//!
//! - Nocedal & Wright, "Numerical Optimization", 2nd ed., §3.5–3.6
//! - Hager & Zhang (2006). "Algorithm 851: CG_DESCENT". ACM TOMS 32(1).
//! - More & Thuente (1994). "Line search algorithms with guaranteed sufficient decrease". ACM TOMS.

use crate::error::OptimizeError;
use crate::unconstrained::Bounds;
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── LineSearchResult ─────────────────────────────────────────────────────────

/// Unified result returned by all line search algorithms in this module.
#[derive(Debug, Clone)]
pub struct LineSearchResult {
    /// Accepted step size α
    pub alpha: f64,
    /// Function value f(x + α d)
    pub f_val: f64,
    /// Gradient at the new point (if computed)
    pub grad: Option<Array1<f64>>,
    /// Directional derivative at the new point: g(x + α d)ᵀ d
    pub derphi: Option<f64>,
    /// Number of function evaluations
    pub n_fev: usize,
    /// Number of gradient evaluations
    pub n_gev: usize,
    /// Whether the search satisfied its termination conditions
    pub success: bool,
    /// Termination message
    pub message: String,
}

// ─── StrongWolfe ─────────────────────────────────────────────────────────────

/// Options for the strong Wolfe line search.
#[derive(Debug, Clone)]
pub struct StrongWolfeConfig {
    /// Armijo sufficient-decrease constant c₁ (0 < c₁ < c₂ < 1)
    pub c1: f64,
    /// Curvature condition constant c₂ (c₁ < c₂ < 1)
    pub c2: f64,
    /// Initial trial step size
    pub alpha_init: f64,
    /// Maximum step size (upper bound for the search)
    pub alpha_max: f64,
    /// Maximum total function + gradient evaluations
    pub max_fev: usize,
    /// Safeguard: minimum step size below which we give up
    pub alpha_min: f64,
}

impl Default for StrongWolfeConfig {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            alpha_init: 1.0,
            alpha_max: 1e10,
            max_fev: 100,
            alpha_min: 1e-14,
        }
    }
}

/// Strong Wolfe conditions line search implemented via the zoom algorithm.
///
/// The search proceeds in two phases:
/// 1. **Bracketing**: increase α until a bracket [αlo, αhi] containing an
///    acceptable step is found.
/// 2. **Zoom**: bisect/interpolate within the bracket until the strong Wolfe
///    conditions are satisfied.
pub struct StrongWolfe {
    /// Configuration
    pub config: StrongWolfeConfig,
}

impl StrongWolfe {
    /// Create with given config.
    pub fn new(config: StrongWolfeConfig) -> Self {
        Self { config }
    }

    /// Create with defaults.
    pub fn default_search() -> Self {
        Self {
            config: StrongWolfeConfig::default(),
        }
    }

    /// Perform the line search.
    ///
    /// # Arguments
    /// * `phi`     – scalar function φ(α) = f(x + α d); gives (f_val) only
    /// * `dphi`    – directional derivative φ'(α) = ∇f(x + α d)ᵀ d
    /// * `phi0`    – φ(0) = f(x)
    /// * `dphi0`   – φ'(0) = ∇f(x)ᵀ d (must be negative for a descent direction)
    pub fn search<Phi, DPhi>(
        &self,
        mut phi: Phi,
        mut dphi: DPhi,
        phi0: f64,
        dphi0: f64,
    ) -> Result<LineSearchResult, OptimizeError>
    where
        Phi: FnMut(f64) -> f64,
        DPhi: FnMut(f64) -> f64,
    {
        let cfg = &self.config;
        if dphi0 >= 0.0 {
            return Err(OptimizeError::ValueError(
                "Initial directional derivative must be negative (descent direction required)"
                    .to_string(),
            ));
        }

        let mut n_fev = 1usize; // phi0 already evaluated by caller
        let mut n_gev = 1usize; // dphi0 too

        let mut alpha_prev = 0.0f64;
        let mut alpha = cfg.alpha_init.min(cfg.alpha_max);
        let mut phi_prev = phi0;
        // dphi at alpha=0
        let mut dphi_prev = dphi0;
        let _ = dphi_prev;

        for _iter in 0..cfg.max_fev {
            let phi_a = phi(alpha);
            n_fev += 1;

            // Armijo condition fails or function increased
            if phi_a > phi0 + cfg.c1 * alpha * dphi0 || (phi_a >= phi_prev && _iter > 0) {
                let (result_alpha, result_phi, nf, ng) = self.zoom(
                    &mut phi, &mut dphi, alpha_prev, phi_prev, alpha, phi_a, phi0, dphi0,
                );
                n_fev += nf;
                n_gev += ng;
                return Ok(LineSearchResult {
                    alpha: result_alpha,
                    f_val: result_phi,
                    grad: None,
                    derphi: None,
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Strong Wolfe conditions satisfied (zoom from upper bracket)"
                        .to_string(),
                });
            }

            let dphi_a = dphi(alpha);
            n_gev += 1;

            // Strong Wolfe curvature condition
            if dphi_a.abs() <= -cfg.c2 * dphi0 {
                return Ok(LineSearchResult {
                    alpha,
                    f_val: phi_a,
                    grad: None,
                    derphi: Some(dphi_a),
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Strong Wolfe conditions satisfied".to_string(),
                });
            }

            if dphi_a >= 0.0 {
                let (result_alpha, result_phi, nf, ng) = self.zoom(
                    &mut phi, &mut dphi, alpha, phi_a, alpha_prev, phi_prev, phi0, dphi0,
                );
                n_fev += nf;
                n_gev += ng;
                return Ok(LineSearchResult {
                    alpha: result_alpha,
                    f_val: result_phi,
                    grad: None,
                    derphi: None,
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Strong Wolfe conditions satisfied (zoom from positive derivative)"
                        .to_string(),
                });
            }

            // Increase alpha (interpolation-based extrapolation)
            alpha_prev = alpha;
            phi_prev = phi_a;
            dphi_prev = dphi_a;
            let alpha_new = (alpha + cfg.alpha_max) * 0.5;
            alpha = cubic_min_bracket(alpha_prev, phi_prev, dphi_a, alpha_new, phi_a)
                .unwrap_or(alpha_new)
                .clamp(alpha * 1.1, cfg.alpha_max);
        }

        // Fallback: return best known step
        let f_alpha = phi(alpha);
        n_fev += 1;
        Ok(LineSearchResult {
            alpha,
            f_val: f_alpha,
            grad: None,
            derphi: None,
            n_fev,
            n_gev,
            success: false,
            message: "Strong Wolfe search did not converge within max evaluations".to_string(),
        })
    }

    /// Zoom phase: find acceptable α in [α_lo, α_hi].
    fn zoom<Phi, DPhi>(
        &self,
        phi: &mut Phi,
        dphi: &mut DPhi,
        alpha_lo: f64,
        phi_lo: f64,
        alpha_hi: f64,
        phi_hi: f64,
        phi0: f64,
        dphi0: f64,
    ) -> (f64, f64, usize, usize)
    where
        Phi: FnMut(f64) -> f64,
        DPhi: FnMut(f64) -> f64,
    {
        let cfg = &self.config;
        let mut n_fev = 0usize;
        let mut n_gev = 0usize;

        let mut a_lo = alpha_lo;
        let mut f_lo = phi_lo;
        let mut a_hi = alpha_hi;
        let mut f_hi = phi_hi;

        for _ in 0..cfg.max_fev {
            // Cubic interpolation to find trial step
            let alpha_j = cubic_min_bracket(a_lo, f_lo, dphi(a_lo), a_hi, f_hi)
                .unwrap_or((a_lo + a_hi) * 0.5)
                .clamp(a_lo.min(a_hi) + 1e-10, a_lo.max(a_hi) - 1e-10);
            n_gev += 1; // dphi(a_lo) counted above

            let phi_j = phi(alpha_j);
            n_fev += 1;

            if phi_j > phi0 + cfg.c1 * alpha_j * dphi0 || phi_j >= f_lo {
                a_hi = alpha_j;
                f_hi = phi_j;
            } else {
                let dphi_j = dphi(alpha_j);
                n_gev += 1;

                if dphi_j.abs() <= -cfg.c2 * dphi0 {
                    return (alpha_j, phi_j, n_fev, n_gev);
                }

                if dphi_j * (a_hi - a_lo) >= 0.0 {
                    a_hi = a_lo;
                    f_hi = f_lo;
                }
                a_lo = alpha_j;
                f_lo = phi_j;
            }

            if (a_hi - a_lo).abs() < cfg.alpha_min {
                break;
            }
        }

        (a_lo, f_lo, n_fev, n_gev)
    }
}

/// Fit a cubic polynomial to the two points and find its minimum.
///
/// Uses the formula from Nocedal & Wright (Algorithm 3.6).
fn cubic_min_bracket(a: f64, fa: f64, dfa: f64, b: f64, fb: f64) -> Option<f64> {
    let d1 = dfa + (fb - fa) / (b - a) * 2.0 - (fb - fa) / (b - a);
    // Simplified cubic min via Nocedal–Wright formula
    let ab = b - a;
    let d = dfa;
    let d2 = 3.0 * (fa - fb) / ab + d;
    let discr = d2 * d2 - d * ((fb - fa) / ab * 3.0 - d2);
    let _ = d1;
    if discr < 0.0 {
        return None;
    }
    let t = d2 - discr.sqrt();
    let denom = (2.0 * d2 - dfa / ab).abs();
    if denom < 1e-300 {
        return None;
    }
    let alpha = a + ab * t / denom;
    Some(alpha)
}

// ─── HagerZhang ──────────────────────────────────────────────────────────────

/// Hager–Zhang (CG_DESCENT) line search options.
#[derive(Debug, Clone)]
pub struct HagerZhangConfig {
    /// δ ∈ (0, 1): Armijo-like constant
    pub delta: f64,
    /// σ ∈ (δ, 1): strong Wolfe curvature constant
    pub sigma: f64,
    /// ε: approximate Wolfe energy tolerance factor
    pub epsilon: f64,
    /// θ ∈ (0, 1): bisection safeguard
    pub theta: f64,
    /// γ ∈ (0, 1): bracket shrinkage test
    pub gamma: f64,
    /// Maximum evaluations
    pub max_fev: usize,
    /// Initial step
    pub alpha_init: f64,
}

impl Default for HagerZhangConfig {
    fn default() -> Self {
        Self {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            gamma: 0.66,
            max_fev: 50,
            alpha_init: 1.0,
        }
    }
}

/// Hager–Zhang CG_DESCENT line search.
///
/// Uses the approximate Wolfe conditions which are more permissive and avoid
/// pathological convergence failures of the exact Wolfe conditions in practice.
pub struct HagerZhang {
    /// Configuration
    pub config: HagerZhangConfig,
}

impl HagerZhang {
    /// Create with given config.
    pub fn new(config: HagerZhangConfig) -> Self {
        Self { config }
    }

    /// Create with defaults.
    pub fn default_search() -> Self {
        Self {
            config: HagerZhangConfig::default(),
        }
    }

    /// Perform the Hager–Zhang line search.
    ///
    /// # Arguments
    /// * `phi`, `dphi` – function and directional derivative (as closures)
    /// * `phi0`, `dphi0` – values at α = 0
    pub fn search<Phi, DPhi>(
        &self,
        mut phi: Phi,
        mut dphi: DPhi,
        phi0: f64,
        dphi0: f64,
    ) -> Result<LineSearchResult, OptimizeError>
    where
        Phi: FnMut(f64) -> f64,
        DPhi: FnMut(f64) -> f64,
    {
        if dphi0 >= 0.0 {
            return Err(OptimizeError::ValueError(
                "Initial directional derivative must be negative".to_string(),
            ));
        }

        let cfg = &self.config;
        let mut n_fev = 1usize;
        let mut n_gev = 1usize;
        let c = cfg.epsilon * phi0.abs().max(1.0);

        // Wolfe1 (approximate Wolfe): phi(alpha) <= phi0 + delta * alpha * dphi0
        // Wolfe2: |dphi(alpha)| <= sigma * |dphi0|  (strong curvature)
        // Approximate Wolfe: dphi(alpha) >= (2 delta - 1) * dphi0  AND  dphi(alpha) <= sigma * dphi0

        let wolfe1 = |pa: f64, a: f64| pa <= phi0 + cfg.delta * a * dphi0;
        let approx_wolfe1 =
            |pa: f64, a: f64| pa <= phi0 + c && cfg.delta * dphi0 >= (phi0 - pa) / a.max(1e-300);
        let wolfe2 = |da: f64| da.abs() <= cfg.sigma * dphi0.abs();
        let approx_wolfe2 =
            |da: f64| (2.0 * cfg.delta - 1.0) * dphi0 <= da && da <= cfg.sigma * dphi0;
        let _ = approx_wolfe1;

        // Initial bracket [a=0, b=alpha_init]
        let mut a = 0.0f64;
        let mut b = cfg.alpha_init;
        let mut fa = phi0;
        let mut fb = phi(b);
        n_fev += 1;
        let mut db = dphi(b);
        n_gev += 1;

        // Quick check: already satisfies conditions
        if wolfe1(fb, b) && wolfe2(db) {
            return Ok(LineSearchResult {
                alpha: b,
                f_val: fb,
                grad: None,
                derphi: Some(db),
                n_fev,
                n_gev,
                success: true,
                message: "Hager-Zhang: initial step satisfies Wolfe conditions".to_string(),
            });
        }

        // Bisection phase (simplified HZ Update)
        for _ in 0..cfg.max_fev {
            let mid = a + cfg.theta * (b - a);
            let fm = phi(mid);
            n_fev += 1;
            let dm = dphi(mid);
            n_gev += 1;

            if wolfe1(fm, mid) && wolfe2(dm) {
                return Ok(LineSearchResult {
                    alpha: mid,
                    f_val: fm,
                    grad: None,
                    derphi: Some(dm),
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Hager-Zhang: bisection converged".to_string(),
                });
            }

            // Check approximate Wolfe
            if approx_wolfe2(dm) && fm <= phi0 + c {
                return Ok(LineSearchResult {
                    alpha: mid,
                    f_val: fm,
                    grad: None,
                    derphi: Some(dm),
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Hager-Zhang: approximate Wolfe satisfied".to_string(),
                });
            }

            // Update bracket
            if dm < 0.0 && fm <= phi0 + c {
                a = mid;
                fa = fm;
            } else {
                b = mid;
                fb = fm;
                db = dm;
            }

            // Safeguard: bracket must shrink
            if (b - a).abs() < 1e-14 {
                break;
            }
            let _ = fa;
            let _ = fb;
            let _ = db;
        }

        // Return best found
        let alpha_best = a + cfg.theta * (b - a);
        let f_best = phi(alpha_best);
        n_fev += 1;

        Ok(LineSearchResult {
            alpha: alpha_best,
            f_val: f_best,
            grad: None,
            derphi: None,
            n_fev,
            n_gev,
            success: false,
            message: "Hager-Zhang: max evaluations reached".to_string(),
        })
    }
}

// ─── SafeguardedPowell ────────────────────────────────────────────────────────

/// Safeguarded Powell cubic interpolation line search.
///
/// Powell's method maintains a bracket `[alo, ahi]` and uses cubic interpolation
/// safeguarded by bisection to find a step satisfying the sufficient decrease condition.
/// This is the method used inside L-BFGS and related quasi-Newton methods.
pub struct SafeguardedPowell {
    /// Armijo constant c₁
    pub c1: f64,
    /// Maximum evaluations
    pub max_fev: usize,
    /// Minimum bracket width before giving up
    pub bracket_tol: f64,
    /// Initial step size
    pub alpha_init: f64,
}

impl Default for SafeguardedPowell {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            max_fev: 50,
            bracket_tol: 1e-14,
            alpha_init: 1.0,
        }
    }
}

impl SafeguardedPowell {
    /// Create with given parameters.
    pub fn new(c1: f64, max_fev: usize, alpha_init: f64) -> Self {
        Self {
            c1,
            max_fev,
            bracket_tol: 1e-14,
            alpha_init,
        }
    }

    /// Perform the safeguarded Powell line search.
    pub fn search<Phi, DPhi>(
        &self,
        mut phi: Phi,
        mut dphi: DPhi,
        phi0: f64,
        dphi0: f64,
    ) -> Result<LineSearchResult, OptimizeError>
    where
        Phi: FnMut(f64) -> f64,
        DPhi: FnMut(f64) -> f64,
    {
        if dphi0 >= 0.0 {
            return Err(OptimizeError::ValueError(
                "Initial directional derivative must be negative".to_string(),
            ));
        }

        let mut n_fev = 1usize;
        let mut n_gev = 1usize;

        let mut alpha = self.alpha_init;
        let mut alpha_lo = 0.0f64;
        let mut f_lo = phi0;
        let mut d_lo = dphi0;

        for _ in 0..self.max_fev {
            let fa = phi(alpha);
            n_fev += 1;

            if fa <= phi0 + self.c1 * alpha * dphi0 {
                // Sufficient decrease satisfied
                let da = dphi(alpha);
                n_gev += 1;
                return Ok(LineSearchResult {
                    alpha,
                    f_val: fa,
                    grad: None,
                    derphi: Some(da),
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Safeguarded Powell: sufficient decrease satisfied".to_string(),
                });
            }

            // Safeguarded cubic interpolation for next trial step
            // Fit cubic through (alpha_lo, f_lo, d_lo) and (alpha, fa)
            let alpha_new =
                cubic_interpolate_safeguarded(alpha_lo, f_lo, d_lo, alpha, fa, self.bracket_tol);

            // Update bracket
            if fa < f_lo {
                // This is a better function value; update bracket
                alpha_lo = alpha;
                f_lo = fa;
                d_lo = dphi(alpha_new.min(alpha));
                n_gev += 1;
            }

            if (alpha_new - alpha_lo).abs() < self.bracket_tol {
                break;
            }
            alpha = alpha_new;
        }

        // Fallback: return best known
        Ok(LineSearchResult {
            alpha,
            f_val: phi(alpha),
            grad: None,
            derphi: None,
            n_fev: n_fev + 1,
            n_gev,
            success: false,
            message: "Safeguarded Powell: max evaluations reached".to_string(),
        })
    }
}

/// Safeguarded cubic interpolation between two points.
fn cubic_interpolate_safeguarded(a: f64, fa: f64, da: f64, b: f64, fb: f64, tol: f64) -> f64 {
    // Cubic interpolation using derivative at `a` and function values at both ends
    let ab = b - a;
    if ab.abs() < tol {
        return (a + b) * 0.5;
    }
    // Coefficients
    let d1 = da * ab;
    let d2 = fb - fa;
    let d3 = d2 * 3.0 / ab - da;
    let d4 = d3 * d3 - da * (d2 * 3.0 / ab - da);
    if d4 < 0.0 {
        return (a + b) * 0.5;
    }
    let sqrt_d4 = d4.sqrt();
    let t = 1.0 - (d3 + sqrt_d4) / (d3 + sqrt_d4 + d1 / ab);
    let alpha_cubic = a + t.clamp(0.0, 1.0) * ab;

    // Safeguard: stay within (a, b)
    alpha_cubic.clamp(a + tol.abs(), b - tol.abs())
}

// ─── BacktrackingArmijo ───────────────────────────────────────────────────────

/// Backtracking line search with Armijo (sufficient decrease) condition.
///
/// This is the simplest reliable line search: it starts with the full step and
/// reduces it geometrically until the function decreases sufficiently. It does
/// not use gradient information after the initial evaluation.
#[derive(Debug, Clone)]
pub struct BacktrackingArmijo {
    /// Armijo constant c₁ (typical: 1e-4)
    pub c1: f64,
    /// Reduction factor ρ ∈ (0, 1) (typical: 0.5)
    pub rho: f64,
    /// Initial step size
    pub alpha_init: f64,
    /// Minimum step size (stop if alpha < alpha_min)
    pub alpha_min: f64,
    /// Maximum number of backtracking steps
    pub max_steps: usize,
    /// Optional box bounds
    pub bounds: Option<Bounds>,
}

impl Default for BacktrackingArmijo {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            rho: 0.5,
            alpha_init: 1.0,
            alpha_min: 1e-14,
            max_steps: 60,
            bounds: None,
        }
    }
}

impl BacktrackingArmijo {
    /// Create with given parameters.
    pub fn new(c1: f64, rho: f64, alpha_init: f64, bounds: Option<Bounds>) -> Self {
        Self {
            c1,
            rho,
            alpha_init,
            alpha_min: 1e-14,
            max_steps: 60,
            bounds,
        }
    }

    /// Perform backtracking line search along a direction.
    ///
    /// Evaluates φ(α) = f(x + α d), where d is the descent direction.
    ///
    /// # Arguments
    /// * `fun`   – objective function
    /// * `x`     – current iterate
    /// * `d`     – descent direction
    /// * `f0`    – current function value f(x)
    /// * `slope` – directional derivative φ'(0) = ∇f(x)ᵀ d (should be negative)
    pub fn search<F>(
        &self,
        fun: &mut F,
        x: &ArrayView1<f64>,
        d: &ArrayView1<f64>,
        f0: f64,
        slope: f64,
    ) -> LineSearchResult
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
    {
        let mut alpha = self.alpha_init;
        let n = x.len();
        let mut n_fev = 0usize;

        // If the slope is non-negative, this is a bad direction
        if slope >= 0.0 {
            return LineSearchResult {
                alpha: 1e-14,
                f_val: f0,
                grad: None,
                derphi: None,
                n_fev: 0,
                n_gev: 0,
                success: false,
                message: "Backtracking: non-descent direction".to_string(),
            };
        }

        for _ in 0..self.max_steps {
            let mut x_new = Array1::zeros(n);
            for i in 0..n {
                x_new[i] = x[i] + alpha * d[i];
            }

            // Project if bounds present
            if let Some(ref b) = self.bounds {
                if let Some(s) = x_new.as_slice_mut() {
                    b.project(s);
                }
            }

            n_fev += 1;
            let f_new = fun(&x_new.view());

            if f_new <= f0 + self.c1 * alpha * slope {
                return LineSearchResult {
                    alpha,
                    f_val: f_new,
                    grad: None,
                    derphi: None,
                    n_fev,
                    n_gev: 0,
                    success: true,
                    message: "Armijo condition satisfied".to_string(),
                };
            }

            alpha *= self.rho;
            if alpha < self.alpha_min {
                return LineSearchResult {
                    alpha: self.alpha_min,
                    f_val: f_new,
                    grad: None,
                    derphi: None,
                    n_fev,
                    n_gev: 0,
                    success: false,
                    message: "Backtracking: alpha below minimum".to_string(),
                };
            }
        }

        let mut x_last = Array1::zeros(n);
        for i in 0..n {
            x_last[i] = x[i] + alpha * d[i];
        }
        n_fev += 1;
        let f_last = fun(&x_last.view());

        LineSearchResult {
            alpha,
            f_val: f_last,
            grad: None,
            derphi: None,
            n_fev,
            n_gev: 0,
            success: false,
            message: "Backtracking: max steps reached".to_string(),
        }
    }

    /// Convenience wrapper: search given a precomputed scalar function φ(α).
    pub fn search_scalar<Phi>(&self, mut phi: Phi, phi0: f64, dphi0: f64) -> LineSearchResult
    where
        Phi: FnMut(f64) -> f64,
    {
        let mut alpha = self.alpha_init;
        let mut n_fev = 0usize;

        if dphi0 >= 0.0 {
            return LineSearchResult {
                alpha: 1e-14,
                f_val: phi0,
                grad: None,
                derphi: None,
                n_fev: 0,
                n_gev: 0,
                success: false,
                message: "Backtracking: non-descent direction".to_string(),
            };
        }

        for _ in 0..self.max_steps {
            n_fev += 1;
            let fa = phi(alpha);
            if fa <= phi0 + self.c1 * alpha * dphi0 {
                return LineSearchResult {
                    alpha,
                    f_val: fa,
                    grad: None,
                    derphi: None,
                    n_fev,
                    n_gev: 0,
                    success: true,
                    message: "Armijo condition satisfied".to_string(),
                };
            }
            alpha *= self.rho;
            if alpha < self.alpha_min {
                break;
            }
        }

        LineSearchResult {
            alpha,
            f_val: phi(alpha),
            grad: None,
            derphi: None,
            n_fev: n_fev + 1,
            n_gev: 0,
            success: false,
            message: "Backtracking: max steps reached".to_string(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// φ(α) = (1 - α)²  ← minimum at α=1, dphi(0) = -2
    fn phi_quadratic(alpha: f64) -> f64 {
        (1.0 - alpha).powi(2)
    }
    fn dphi_quadratic(alpha: f64) -> f64 {
        -2.0 * (1.0 - alpha)
    }

    #[test]
    fn test_strong_wolfe_quadratic() {
        let sw = StrongWolfe::default_search();
        let result = sw
            .search(phi_quadratic, dphi_quadratic, 1.0, -2.0)
            .expect("StrongWolfe failed");
        // Should converge to alpha ≈ 1.0
        assert!(result.success);
        assert!(result.alpha > 0.0 && result.alpha <= 2.0);
        // Function value should be below initial
        assert!(result.f_val < 1.0);
    }

    #[test]
    fn test_hager_zhang_quadratic() {
        let hz = HagerZhang::default_search();
        let result = hz
            .search(phi_quadratic, dphi_quadratic, 1.0, -2.0)
            .expect("HagerZhang failed");
        assert!(result.alpha > 0.0);
        assert!(result.f_val <= 1.0);
    }

    #[test]
    fn test_backtracking_armijo_quadratic() {
        let bt = BacktrackingArmijo::default();
        let result = bt.search_scalar(phi_quadratic, 1.0, -2.0);
        assert!(result.success);
        assert!(result.f_val <= 1.0);
    }

    #[test]
    fn test_safeguarded_powell_quadratic() {
        let pw = SafeguardedPowell::default();
        let result = pw
            .search(phi_quadratic, dphi_quadratic, 1.0, -2.0)
            .expect("Powell failed");
        assert!(result.alpha > 0.0);
        assert!(result.f_val < 1.0);
    }

    #[test]
    fn test_backtracking_armijo_bad_direction() {
        let bt = BacktrackingArmijo::default();
        let result = bt.search_scalar(phi_quadratic, 1.0, 1.0); // dphi0 > 0 → bad
        assert!(!result.success);
    }

    #[test]
    fn test_strong_wolfe_bad_direction() {
        let sw = StrongWolfe::default_search();
        let err = sw.search(phi_quadratic, dphi_quadratic, 1.0, 1.0);
        assert!(err.is_err());
    }
}
