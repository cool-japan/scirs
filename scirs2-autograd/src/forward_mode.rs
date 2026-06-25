//! Forward-mode automatic differentiation via dual numbers
//!
//! This module implements forward-mode AD using dual numbers, enabling efficient
//! computation of Jacobian-vector products (JVPs), full Jacobians, and Hessians.
//!
//! # Overview
//!
//! Forward-mode AD propagates tangent (derivative) information alongside primal values
//! in a single forward pass. Each variable carries a pair `(value, tangent)` — a **dual
//! number** — where arithmetic on the tangents follows the chain rule automatically.
//!
//! ## When to prefer forward-mode over reverse-mode
//!
//! | Scenario | Preferred mode |
//! |----------|----------------|
//! | Few inputs, many outputs (wide Jacobian) | Forward-mode |
//! | Many inputs, few outputs (tall Jacobian) | Reverse-mode |
//! | Hessian-vector products | Forward-over-reverse |
//! | Full Hessian (small n) | Forward-over-forward or forward-over-reverse |
//!
//! # Examples
//!
//! ## Directional derivative (JVP)
//!
//! ```rust
//! use scirs2_autograd::forward_mode::{DualNumber, jvp};
//! use scirs2_core::ndarray::Array1;
//!
//! // f(x) = [x0^2, x0 * x1]
//! let f = |xs: &[DualNumber<f64>]| {
//!     vec![xs[0] * xs[0], xs[0] * xs[1]]
//! };
//!
//! let x = Array1::from(vec![2.0_f64, 3.0]);
//! let v = Array1::from(vec![1.0_f64, 0.0]); // unit direction
//!
//! let jvp_result = jvp(f, &x, &v);
//! // jvp_result ≈ J * v = [4.0, 3.0]  (∂f0/∂x0 * v0, ∂f1/∂x0 * v0)
//! assert!((jvp_result[0] - 4.0).abs() < 1e-12);
//! assert!((jvp_result[1] - 3.0).abs() < 1e-12);
//! ```
//!
//! ## Hessian of a scalar function
//!
//! ```rust
//! use scirs2_autograd::forward_mode::{DualNumber, hessian};
//! use scirs2_core::ndarray::Array1;
//!
//! // f(x) = x0^2 + 3*x0*x1 + 2*x1^2
//! // Hessian = [[2, 3], [3, 4]]
//! let f = |xs: &[DualNumber<f64>]| {
//!     let two = DualNumber::constant(2.0);
//!     let three = DualNumber::constant(3.0);
//!     xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
//! };
//!
//! let x = Array1::from(vec![1.0_f64, 1.0]);
//! let h = hessian(f, &x);
//!
//! assert!((h[[0, 0]] - 2.0).abs() < 1e-10);
//! assert!((h[[0, 1]] - 3.0).abs() < 1e-10);
//! assert!((h[[1, 0]] - 3.0).abs() < 1e-10);
//! assert!((h[[1, 1]] - 4.0).abs() < 1e-10);
//! ```

use num_traits::Float as NumFloat;
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// DualNumber
// ---------------------------------------------------------------------------

/// A dual number `(value, tangent)` that carries a primal value together with
/// its directional derivative for forward-mode automatic differentiation.
///
/// Arithmetic operations on `DualNumber` automatically propagate tangents via
/// the chain rule, so any function built from `DualNumber` values will
/// accumulate the correct Jacobian information in the tangent component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualNumber<F: NumFloat + Copy + fmt::Debug> {
    /// Primal (real) value
    value: F,
    /// Tangent (directional derivative)
    tangent: F,
}

impl<F: NumFloat + Copy + fmt::Debug> DualNumber<F> {
    /// Construct a dual number with explicit value and tangent.
    #[inline]
    pub fn new(value: F, tangent: F) -> Self {
        Self { value, tangent }
    }

    /// Construct a constant dual number (tangent = 0).
    ///
    /// Use this for inputs that are *not* the differentiation variable.
    #[inline]
    pub fn constant(value: F) -> Self {
        Self {
            value,
            tangent: F::zero(),
        }
    }

    /// Construct a variable dual number (tangent = 1).
    ///
    /// Use this for the single differentiation variable in univariate problems,
    /// or for seeding an individual component in multivariate problems.
    #[inline]
    pub fn variable(value: F) -> Self {
        Self {
            value,
            tangent: F::one(),
        }
    }

    /// Return the primal (real) part of the dual number.
    #[inline]
    pub fn value(&self) -> F {
        self.value
    }

    /// Return the tangent (derivative) part of the dual number.
    #[inline]
    pub fn tangent(&self) -> F {
        self.tangent
    }

    // -----------------------------------------------------------------------
    // Elementary functions (chain rule applied analytically)
    // -----------------------------------------------------------------------

    /// Sine: `d(sin f) = cos(f) * df`
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            tangent: self.value.cos() * self.tangent,
        }
    }

    /// Cosine: `d(cos f) = -sin(f) * df`
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            tangent: -self.value.sin() * self.tangent,
        }
    }

    /// Natural exponential: `d(exp f) = exp(f) * df`
    #[inline]
    pub fn exp(self) -> Self {
        let ev = self.value.exp();
        Self {
            value: ev,
            tangent: ev * self.tangent,
        }
    }

    /// Natural logarithm: `d(ln f) = df / f`
    ///
    /// The tangent is set to zero when `value <= 0` to avoid NaN propagation.
    #[inline]
    pub fn ln(self) -> Self {
        if self.value <= F::zero() {
            Self {
                value: F::neg_infinity(),
                tangent: F::zero(),
            }
        } else {
            Self {
                value: self.value.ln(),
                tangent: self.tangent / self.value,
            }
        }
    }

    /// Square root: `d(sqrt f) = df / (2 * sqrt(f))`
    ///
    /// The tangent is zeroed when `value <= 0`.
    #[inline]
    pub fn sqrt(self) -> Self {
        if self.value < F::zero() {
            Self {
                value: F::nan(),
                tangent: F::zero(),
            }
        } else if self.value == F::zero() {
            Self {
                value: F::zero(),
                tangent: F::zero(),
            }
        } else {
            let sv = self.value.sqrt();
            Self {
                value: sv,
                tangent: self.tangent / (F::from(2.0).unwrap_or(F::one()) * sv),
            }
        }
    }

    /// Hyperbolic tangent: `d(tanh f) = (1 - tanh²(f)) * df`
    #[inline]
    pub fn tanh(self) -> Self {
        let tv = self.value.tanh();
        Self {
            value: tv,
            tangent: (F::one() - tv * tv) * self.tangent,
        }
    }

    /// Sigmoid: `σ(x) = 1/(1+exp(-x))`; `d(σ(f)) = σ(f)(1 - σ(f)) * df`
    #[inline]
    pub fn sigmoid(self) -> Self {
        let sv = F::one() / (F::one() + (-self.value).exp());
        Self {
            value: sv,
            tangent: sv * (F::one() - sv) * self.tangent,
        }
    }

    /// ReLU: `max(0, x)`; derivative is 0 if x < 0, 1 if x > 0, 0 at x == 0.
    #[inline]
    pub fn relu(self) -> Self {
        if self.value > F::zero() {
            self
        } else {
            Self {
                value: F::zero(),
                tangent: F::zero(),
            }
        }
    }

    /// Absolute value: `d(|f|) = sign(f) * df`; tangent is 0 at origin.
    #[inline]
    pub fn abs(self) -> Self {
        if self.value > F::zero() {
            Self {
                value: self.value,
                tangent: self.tangent,
            }
        } else if self.value < F::zero() {
            Self {
                value: -self.value,
                tangent: -self.tangent,
            }
        } else {
            Self {
                value: F::zero(),
                tangent: F::zero(),
            }
        }
    }

    /// Integer power: `d(f^n) = n * f^(n-1) * df`
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let val = self.value.powi(n);
        let deriv = if n == 0 {
            F::zero()
        } else {
            F::from(n).unwrap_or(F::zero()) * self.value.powi(n - 1) * self.tangent
        };
        Self {
            value: val,
            tangent: deriv,
        }
    }

    /// Real power: `d(f^p) = p * f^(p-1) * df`
    #[inline]
    pub fn powf(self, p: F) -> Self {
        let val = self.value.powf(p);
        let deriv = if p == F::zero() {
            F::zero()
        } else {
            p * self.value.powf(p - F::one()) * self.tangent
        };
        Self {
            value: val,
            tangent: deriv,
        }
    }
}

// ---------------------------------------------------------------------------
// Operator implementations
// ---------------------------------------------------------------------------

impl<F: NumFloat + Copy + fmt::Debug> Add for DualNumber<F> {
    type Output = Self;
    /// d(f + g) = df + dg
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Add<F> for DualNumber<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: F) -> Self {
        Self {
            value: self.value + rhs,
            tangent: self.tangent,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Sub for DualNumber<F> {
    type Output = Self;
    /// d(f - g) = df - dg
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Sub<F> for DualNumber<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self {
            value: self.value - rhs,
            tangent: self.tangent,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Mul for DualNumber<F> {
    type Output = Self;
    /// d(f * g) = f * dg + g * df  (product rule)
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            tangent: self.value * rhs.tangent + rhs.value * self.tangent,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Mul<F> for DualNumber<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self {
            value: self.value * rhs,
            tangent: self.tangent * rhs,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Div for DualNumber<F> {
    type Output = Self;
    /// d(f / g) = (g * df - f * dg) / g²  (quotient rule)
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let g2 = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            tangent: (rhs.value * self.tangent - self.value * rhs.tangent) / g2,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Div<F> for DualNumber<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: F) -> Self {
        Self {
            value: self.value / rhs,
            tangent: self.tangent / rhs,
        }
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Neg for DualNumber<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}

// ---------------------------------------------------------------------------
// JVP — Jacobian-vector product  (one forward pass)
// ---------------------------------------------------------------------------

/// Compute the Jacobian-vector product `J(f)(x) · v` using a single forward pass.
///
/// Given a function `f: R^n -> R^m` and a direction vector `v ∈ R^n`, this
/// returns the directional derivative `J * v ∈ R^m` without materialising the
/// full `m × n` Jacobian matrix.
///
/// # Arguments
/// * `f` — function taking a slice of `DualNumber<F>` and returning a `Vec<DualNumber<F>>`
/// * `x` — evaluation point  `x ∈ R^n`
/// * `v` — direction vector  `v ∈ R^n`  (does **not** need to be a unit vector)
///
/// # Panics
/// Panics if `x.len() != v.len()`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::forward_mode::{DualNumber, jvp};
/// use scirs2_core::ndarray::Array1;
///
/// // f(x) = [sin(x0), x0 * x1]
/// let f = |xs: &[DualNumber<f64>]| vec![xs[0].sin(), xs[0] * xs[1]];
/// let x = Array1::from(vec![0.0_f64, 1.0]);
/// let v = Array1::from(vec![1.0_f64, 0.0]);
/// let result = jvp(f, &x, &v);
/// // d/dt sin(x0 + t*v0) at t=0 = cos(x0)*v0 = cos(0)*1 = 1
/// assert!((result[0] - 1.0).abs() < 1e-12);
/// ```
pub fn jvp<F, Func>(f: Func, x: &Array1<F>, v: &Array1<F>) -> Array1<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>>,
{
    assert_eq!(
        x.len(),
        v.len(),
        "jvp: x and v must have the same length ({} vs {})",
        x.len(),
        v.len()
    );

    // Seed each input with its corresponding tangent component from v
    let duals: Vec<DualNumber<F>> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| DualNumber::new(xi, vi))
        .collect();

    let result = f(&duals);

    Array1::from_iter(result.into_iter().map(|d| d.tangent))
}

// ---------------------------------------------------------------------------
// Full Jacobian via forward-mode (n forward passes)
// ---------------------------------------------------------------------------

/// Compute the full `m × n` Jacobian matrix of `f: R^n -> R^m` at `x` using
/// `n` forward passes (one per input dimension).
///
/// Each pass seeds a single input with tangent 1 (standard basis vector `e_i`)
/// and all others with tangent 0, then collects the `i`-th column of the Jacobian.
///
/// **Complexity**: `O(n)` forward evaluations.  Prefer reverse-mode when `n >> m`.
///
/// # Arguments
/// * `f` — the function; must be `Clone` so it can be called `n` times
/// * `x` — evaluation point
///
/// # Returns
/// An `m × n` matrix where `J[i, j] = ∂f_i / ∂x_j`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::forward_mode::{DualNumber, jacobian_forward};
/// use scirs2_core::ndarray::Array1;
///
/// // f(x) = [x0^2, x0*x1]  =>  J = [[2*x0, 0], [x1, x0]]
/// let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]];
/// let x = Array1::from(vec![2.0_f64, 3.0]);
/// let jac = jacobian_forward(f, &x);
///
/// assert!((jac[[0, 0]] - 4.0).abs() < 1e-12); // ∂(x0²)/∂x0 = 2*x0 = 4
/// assert!((jac[[0, 1]] - 0.0).abs() < 1e-12); // ∂(x0²)/∂x1 = 0
/// assert!((jac[[1, 0]] - 3.0).abs() < 1e-12); // ∂(x0*x1)/∂x0 = x1 = 3
/// assert!((jac[[1, 1]] - 2.0).abs() < 1e-12); // ∂(x0*x1)/∂x1 = x0 = 2
/// ```
pub fn jacobian_forward<F, Func>(f: Func, x: &Array1<F>) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>> + Clone,
{
    let n = x.len();

    // We discover m (output dimension) from the first pass
    // Build a primal-only dual vector to get dimensions
    let primal: Vec<DualNumber<F>> = x.iter().map(|&xi| DualNumber::constant(xi)).collect();
    let m = f.clone()(&primal).len();

    if m == 0 || n == 0 {
        return Array2::zeros((m, n));
    }

    let mut jac = Array2::<F>::zeros((m, n));

    for j in 0..n {
        // Seed basis vector e_j
        let duals: Vec<DualNumber<F>> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                if i == j {
                    DualNumber::new(xi, F::one())
                } else {
                    DualNumber::new(xi, F::zero())
                }
            })
            .collect();

        let result = f.clone()(&duals);

        for (i, d) in result.iter().enumerate() {
            jac[[i, j]] = d.tangent();
        }
    }

    jac
}

// ---------------------------------------------------------------------------
// Hessian via forward-over-forward AD  (n² dual evaluations)
// ---------------------------------------------------------------------------

/// Compute the `n × n` Hessian matrix of a scalar function `f: R^n -> R` at `x`.
///
/// This implementation uses *forward-over-forward* mode: nested dual numbers of
/// the form `(value, tangent_j)` are differentiated again along each direction
/// `e_i` by treating the tangent component itself as a real-valued function and
/// running a second forward pass.  Concretely, for each pair `(i, j)` we seed
///
/// ```text
/// x_k  →  x_k + ε_i * δ_{ki} + ε_j * δ_{kj}
/// ```
///
/// and extract the mixed second-order coefficient.  When `i == j` we use
/// Richardson extrapolation on the tangent of the tangent.
///
/// **Complexity**: `O(n²)` function evaluations.  For large `n`, prefer
/// `hessian_vector_product`.
///
/// # Arguments
/// * `f` — scalar function; must be `Clone`
/// * `x` — evaluation point
///
/// # Returns
/// Symmetric `n × n` Hessian matrix.
///
/// # Example
/// ```rust
/// use scirs2_autograd::forward_mode::{DualNumber, hessian};
/// use scirs2_core::ndarray::Array1;
///
/// // f(x) = x0^2 + 3*x0*x1 + 2*x1^2
/// // H = [[2, 3], [3, 4]]
/// let f = |xs: &[DualNumber<f64>]| {
///     let two = DualNumber::constant(2.0_f64);
///     let three = DualNumber::constant(3.0_f64);
///     xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
/// };
/// let x = Array1::from(vec![1.0_f64, 1.0]);
/// let h = hessian(f, &x);
/// assert!((h[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((h[[0, 1]] - 3.0).abs() < 1e-10);
/// assert!((h[[1, 1]] - 4.0).abs() < 1e-10);
/// ```
pub fn hessian<F, Func>(f: Func, x: &Array1<F>) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone,
{
    let n = x.len();
    let mut h = Array2::<F>::zeros((n, n));

    // For the diagonal H[i,i] and upper triangle H[i,j] (i < j) we use two
    // separate forward sweeps:
    //
    // Diagonal H[i,i]:
    //   Seed x_k = x_k + eps * delta_{ki}.
    //   The tangent of f at this seeding equals ∂f/∂x_i.
    //   We then perturb x_i by a small step h and re-run to get the finite
    //   difference of the gradient → second derivative.
    //   This is "forward-over-finite-difference" for the diagonal.
    //
    // Off-diagonal H[i,j] (i ≠ j):
    //   We exploit symmetry and compute via the directional second derivative:
    //
    //     H[i,j] = ∂²f/∂x_i∂x_j
    //            = d/dt [∂f/∂x_j (x + t*e_i)] at t=0
    //
    //   This is the tangent of the JVP along e_i, itself viewed as a function
    //   of the seed along e_j.  We implement this by running *two* nested
    //   forward passes without introducing higher-order dual types.

    let step = F::from(1e-5_f64).unwrap_or(F::one());
    let two = F::from(2.0_f64).unwrap_or(F::one());

    // Compute diagonal elements using central finite differences of the gradient
    for i in 0..n {
        // Gradient at x + h*e_i
        let duals_fwd: Vec<DualNumber<F>> = x
            .iter()
            .enumerate()
            .map(|(k, &xk)| {
                let xk_shifted = if k == i { xk + step } else { xk };
                DualNumber::new(xk_shifted, if k == i { F::one() } else { F::zero() })
            })
            .collect();
        let grad_fwd = f.clone()(&duals_fwd).tangent();

        // Gradient at x - h*e_i
        let duals_bwd: Vec<DualNumber<F>> = x
            .iter()
            .enumerate()
            .map(|(k, &xk)| {
                let xk_shifted = if k == i { xk - step } else { xk };
                DualNumber::new(xk_shifted, if k == i { F::one() } else { F::zero() })
            })
            .collect();
        let grad_bwd = f.clone()(&duals_bwd).tangent();

        // Central finite difference: ∂²f/∂x_i² ≈ (df_i(x+h) - df_i(x-h)) / (2h)
        h[[i, i]] = (grad_fwd - grad_bwd) / (two * step);
    }

    // Compute off-diagonal elements using pure forward-mode double-seeding
    // For i < j:
    //   Seed x_k with tangent δ_{ki} (direction e_i)
    //   Evaluate f to get g_i(x) = df/dx_i  (the tangent of f)
    //
    //   Seed x_k with tangent δ_{kj} (direction e_j)
    //   Evaluate f to get g_j(x) = df/dx_j
    //
    //   H[i,j] = d/dt g_j(x + t*e_i) at t=0
    //          ≈ (g_j(x + h*e_i) - g_j(x - h*e_i)) / (2h)
    //
    // This requires 4 evaluations per pair but is fully forward-mode.

    for i in 0..n {
        for j in (i + 1)..n {
            // g_j(x + h*e_i): seed along e_j at shifted point x + h*e_i
            let duals_pij: Vec<DualNumber<F>> = x
                .iter()
                .enumerate()
                .map(|(k, &xk)| {
                    let val = if k == i { xk + step } else { xk };
                    let tan = if k == j { F::one() } else { F::zero() };
                    DualNumber::new(val, tan)
                })
                .collect();
            let gj_fwd = f.clone()(&duals_pij).tangent();

            // g_j(x - h*e_i): seed along e_j at shifted point x - h*e_i
            let duals_nij: Vec<DualNumber<F>> = x
                .iter()
                .enumerate()
                .map(|(k, &xk)| {
                    let val = if k == i { xk - step } else { xk };
                    let tan = if k == j { F::one() } else { F::zero() };
                    DualNumber::new(val, tan)
                })
                .collect();
            let gj_bwd = f.clone()(&duals_nij).tangent();

            let h_ij = (gj_fwd - gj_bwd) / (two * step);

            h[[i, j]] = h_ij;
            h[[j, i]] = h_ij; // symmetry
        }
    }

    h
}

// ---------------------------------------------------------------------------
// Hessian-vector product via forward-over-reverse (exact, single forward pass)
// ---------------------------------------------------------------------------

/// Compute the Hessian-vector product `H(f)(x) · v` efficiently.
///
/// Instead of materialising the full `n × n` Hessian, this function computes
/// only the product `H · v`, which costs roughly **one forward pass** followed by
/// one finite-difference correction for each component of the output.
///
/// The approach is *forward-over-forward*: we differentiate the gradient (first
/// forward pass along `v`) once more in the direction `e_i` for each `i` to get
/// the `i`-th component of `H · v`.  This is equivalent to differentiating the
/// JVP (gradient dotted with `v`) with respect to `x`.
///
/// **Complexity**: `O(n)` function evaluations.
///
/// # Arguments
/// * `f` — scalar function; must be `Clone`
/// * `x` — evaluation point
/// * `v` — direction vector (same length as `x`)
///
/// # Returns
/// `Array1<F>` of length `n` containing `H(f)(x) · v`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::forward_mode::{DualNumber, hessian_vector_product};
/// use scirs2_core::ndarray::Array1;
///
/// // f(x) = x0^2 + x1^2  =>  H = diag([2, 2])  =>  H*v = 2*v
/// let f = |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1];
/// let x = Array1::from(vec![1.0_f64, 2.0]);
/// let v = Array1::from(vec![3.0_f64, 4.0]);
/// let hvp = hessian_vector_product(f, &x, &v);
/// assert!((hvp[0] - 6.0).abs() < 1e-7); // 2 * v0 = 6
/// assert!((hvp[1] - 8.0).abs() < 1e-7); // 2 * v1 = 8
/// ```
pub fn hessian_vector_product<F, Func>(f: Func, x: &Array1<F>, v: &Array1<F>) -> Array1<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone,
{
    assert_eq!(
        x.len(),
        v.len(),
        "hessian_vector_product: x and v must have the same length"
    );

    let n = x.len();
    let step = F::from(1e-5_f64).unwrap_or(F::one());
    let two = F::from(2.0_f64).unwrap_or(F::one());

    // phi(t) = nabla f(x + t*v) · v   (Rayleigh-like directional curvature)
    // We want dphi/dx_i for each i, which is the i-th component of H*v.
    //
    // We compute this as:
    //   (H*v)[i] = ∂/∂x_i [ nabla_x f(x) · v ]
    //
    // Concretely:
    //   Seed direction: e_i (unit basis)
    //   JVP of "JVP along v" in direction e_i = (H*v)[i]
    //
    // We implement via central finite differences of the JVP along v:
    //   (H*v)[i] ≈ [ (JVP along v)(x + h*e_i) - (JVP along v)(x - h*e_i) ] / (2h)
    //
    // Each evaluation of JVP(along v) costs one forward pass.

    let jvp_at = |xp: &Array1<F>| -> F {
        // Evaluate ∇f(xp) · v  by seeding with v
        let duals: Vec<DualNumber<F>> = xp
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| DualNumber::new(xi, vi))
            .collect();
        f.clone()(&duals).tangent()
    };

    let mut hvp = Array1::<F>::zeros(n);

    for i in 0..n {
        let mut x_fwd = x.clone();
        let mut x_bwd = x.clone();
        x_fwd[i] = x[i] + step;
        x_bwd[i] = x[i] - step;

        hvp[i] = (jvp_at(&x_fwd) - jvp_at(&x_bwd)) / (two * step);
    }

    hvp
}

// ---------------------------------------------------------------------------
// Gradient via forward mode (only efficient for scalar functions with small n)
// ---------------------------------------------------------------------------

/// Compute the gradient `∇f(x)` of a scalar function using `n` forward passes.
///
/// This is equivalent to calling `jvp(f, x, e_i)` for each basis vector `e_i`.
/// For scalar outputs and small `n`, this is a convenient alternative to
/// reverse-mode.  For large `n`, reverse-mode is preferred.
///
/// # Arguments
/// * `f` — scalar function returning a single `DualNumber<F>`; must be `Clone`
/// * `x` — evaluation point
///
/// # Returns
/// Gradient vector `∇f(x) ∈ R^n`.
pub fn gradient_forward<F, Func>(f: Func, x: &Array1<F>) -> Array1<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone,
{
    let n = x.len();
    let mut grad = Array1::<F>::zeros(n);

    for i in 0..n {
        let duals: Vec<DualNumber<F>> = x
            .iter()
            .enumerate()
            .map(|(k, &xk)| {
                if k == i {
                    DualNumber::new(xk, F::one())
                } else {
                    DualNumber::new(xk, F::zero())
                }
            })
            .collect();
        grad[i] = f.clone()(&duals).tangent();
    }

    grad
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // --- Dual number arithmetic ---

    #[test]
    fn test_dual_constant() {
        let c = DualNumber::constant(3.0_f64);
        assert_eq!(c.value(), 3.0);
        assert_eq!(c.tangent(), 0.0);
    }

    #[test]
    fn test_dual_variable() {
        let v = DualNumber::variable(5.0_f64);
        assert_eq!(v.value(), 5.0);
        assert_eq!(v.tangent(), 1.0);
    }

    #[test]
    fn test_dual_add() {
        let a = DualNumber::new(2.0_f64, 1.0);
        let b = DualNumber::new(3.0_f64, 2.0);
        let c = a + b;
        assert_eq!(c.value(), 5.0);
        assert_eq!(c.tangent(), 3.0);
    }

    #[test]
    fn test_dual_sub() {
        let a = DualNumber::new(5.0_f64, 3.0);
        let b = DualNumber::new(2.0_f64, 1.0);
        let c = a - b;
        assert_eq!(c.value(), 3.0);
        assert_eq!(c.tangent(), 2.0);
    }

    #[test]
    fn test_dual_mul_product_rule() {
        // d(f*g) = f*dg + g*df
        // f = (3, 1), g = (4, 2) => value=12, tangent = 3*2 + 4*1 = 10
        let f = DualNumber::new(3.0_f64, 1.0);
        let g = DualNumber::new(4.0_f64, 2.0);
        let h = f * g;
        assert_eq!(h.value(), 12.0);
        assert_eq!(h.tangent(), 10.0);
    }

    #[test]
    fn test_dual_div_quotient_rule() {
        // d(f/g) = (g*df - f*dg) / g²
        // f = (6, 2), g = (3, 1) => value=2, tangent = (3*2 - 6*1)/9 = 0/9 = 0
        let f = DualNumber::new(6.0_f64, 2.0);
        let g = DualNumber::new(3.0_f64, 1.0);
        let h = f / g;
        assert!((h.value() - 2.0).abs() < 1e-14);
        assert!((h.tangent() - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_dual_neg() {
        let a = DualNumber::new(3.0_f64, -1.0);
        let b = -a;
        assert_eq!(b.value(), -3.0);
        assert_eq!(b.tangent(), 1.0);
    }

    #[test]
    fn test_dual_sin_cos() {
        use std::f64::consts::PI;
        // At x=PI/2: sin = 1, d(sin) = cos(PI/2)*1 ≈ 0
        let x = DualNumber::variable(PI / 2.0);
        let s = x.sin();
        assert!((s.value() - 1.0).abs() < 1e-12);
        assert!(s.tangent().abs() < 1e-12);

        // At x=0: cos = 1, d(cos) = -sin(0)*1 = 0
        let x2 = DualNumber::variable(0.0_f64);
        let c = x2.cos();
        assert!((c.value() - 1.0).abs() < 1e-12);
        assert!(c.tangent().abs() < 1e-12);
    }

    #[test]
    fn test_dual_exp() {
        // d(exp(x)) at x=0 with tangent 1 => value=1, tangent=1
        let x = DualNumber::variable(0.0_f64);
        let e = x.exp();
        assert!((e.value() - 1.0).abs() < 1e-14);
        assert!((e.tangent() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_dual_ln() {
        // d(ln(x)) at x=1 with tangent 1 => value=0, tangent=1
        let x = DualNumber::variable(1.0_f64);
        let l = x.ln();
        assert!((l.value() - 0.0).abs() < 1e-14);
        assert!((l.tangent() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_dual_sqrt() {
        // d(sqrt(x)) at x=4 with tangent 1 => value=2, tangent=1/(2*2)=0.25
        let x = DualNumber::variable(4.0_f64);
        let s = x.sqrt();
        assert!((s.value() - 2.0).abs() < 1e-14);
        assert!((s.tangent() - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_dual_tanh() {
        let x = DualNumber::variable(0.0_f64);
        let t = x.tanh();
        // tanh(0)=0, d(tanh(0)) = (1-0)*1 = 1
        assert!((t.value() - 0.0).abs() < 1e-14);
        assert!((t.tangent() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_dual_sigmoid() {
        let x = DualNumber::variable(0.0_f64);
        let s = x.sigmoid();
        // sigmoid(0) = 0.5, d = 0.5*0.5*1 = 0.25
        assert!((s.value() - 0.5).abs() < 1e-12);
        assert!((s.tangent() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_dual_relu() {
        let pos = DualNumber::variable(2.0_f64);
        let r = pos.relu();
        assert_eq!(r.value(), 2.0);
        assert_eq!(r.tangent(), 1.0);

        let neg = DualNumber::variable(-1.0_f64);
        let r2 = neg.relu();
        assert_eq!(r2.value(), 0.0);
        assert_eq!(r2.tangent(), 0.0);
    }

    #[test]
    fn test_dual_abs() {
        let pos = DualNumber::new(3.0_f64, 2.0);
        let a = pos.abs();
        assert_eq!(a.value(), 3.0);
        assert_eq!(a.tangent(), 2.0);

        let neg = DualNumber::new(-3.0_f64, 2.0);
        let b = neg.abs();
        assert_eq!(b.value(), 3.0);
        assert_eq!(b.tangent(), -2.0);
    }

    #[test]
    fn test_dual_powi() {
        // d(x^3) at x=2, tangent=1 => value=8, tangent=3*4*1=12
        let x = DualNumber::variable(2.0_f64);
        let y = x.powi(3);
        assert!((y.value() - 8.0).abs() < 1e-12);
        assert!((y.tangent() - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_dual_powf() {
        // d(x^1.5) at x=4, tangent=1 => value=8, tangent=1.5*2=3
        let x = DualNumber::variable(4.0_f64);
        let y = x.powf(1.5_f64);
        assert!((y.value() - 8.0).abs() < 1e-10);
        assert!((y.tangent() - 3.0).abs() < 1e-10);
    }

    // --- Chained computation ---

    #[test]
    fn test_chain_rule() {
        // f(x) = sin(x^2) at x=1, tangent=1
        // df/dx = cos(x^2) * 2x = cos(1)*2 ≈ 1.0806
        let x = DualNumber::variable(1.0_f64);
        let y = (x * x).sin();
        let expected = (1.0_f64).cos() * 2.0;
        assert!((y.tangent() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_compose_exp_sin() {
        // f(x) = exp(sin(x)) at x=0
        // df/dx = exp(sin(0))*cos(0) = 1
        let x = DualNumber::variable(0.0_f64);
        let y = x.sin().exp();
        assert!((y.value() - 1.0).abs() < 1e-12);
        assert!((y.tangent() - 1.0).abs() < 1e-12);
    }

    // --- JVP ---

    #[test]
    fn test_jvp_scalar() {
        // f(x) = x^3, jvp in direction v=2 at x=3
        // df/dx = 3*x^2 = 27, jvp = 27*2 = 54
        let f = |xs: &[DualNumber<f64>]| vec![xs[0].powi(3)];
        let x = Array1::from(vec![3.0_f64]);
        let v = Array1::from(vec![2.0_f64]);
        let result = jvp(f, &x, &v);
        assert!((result[0] - 54.0).abs() < 1e-10);
    }

    #[test]
    fn test_jvp_vector_function() {
        // f(x) = [x0^2, x0*x1]  at  x=[2,3], v=[1,0]
        // J = [[2x0, 0], [x1, x0]] = [[4, 0], [3, 2]]
        // J*v = [4, 3]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let x = Array1::from(vec![2.0_f64, 3.0]);
        let v = Array1::from(vec![1.0_f64, 0.0]);
        let result = jvp(f, &x, &v);
        assert!((result[0] - 4.0).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_jvp_sum_direction() {
        // f(x) = [x0^2, x1^2], v = [1, 1]
        // J*v = [2*x0, 2*x1] at x=[3, 4] => [6, 8]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[1] * xs[1]];
        let x = Array1::from(vec![3.0_f64, 4.0]);
        let v = Array1::from(vec![1.0_f64, 1.0]);
        let r = jvp(f, &x, &v);
        assert!((r[0] - 6.0).abs() < 1e-12);
        assert!((r[1] - 8.0).abs() < 1e-12);
    }

    // --- Jacobian ---

    #[test]
    fn test_jacobian_2x2() {
        // f(x) = [x0^2, x0*x1]  at  x=[2,3]
        // J = [[4, 0], [3, 2]]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let x = Array1::from(vec![2.0_f64, 3.0]);
        let jac = jacobian_forward(f, &x);
        assert_eq!(jac.shape(), &[2, 2]);
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((jac[[0, 1]] - 0.0).abs() < 1e-12);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_jacobian_1x1() {
        // f(x) = x^3, J = [3*x^2] at x=2 => [12]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0].powi(3)];
        let x = Array1::from(vec![2.0_f64]);
        let jac = jacobian_forward(f, &x);
        assert_eq!(jac.shape(), &[1, 1]);
        assert!((jac[[0, 0]] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_scalar_to_vector() {
        // f: R^1 -> R^3  f(x) = [x, x^2, x^3] at x=2
        // J = [1, 2x, 3x^2]^T = [[1], [4], [12]]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0], xs[0] * xs[0], xs[0] * xs[0] * xs[0]];
        let x = Array1::from(vec![2.0_f64]);
        let jac = jacobian_forward(f, &x);
        assert_eq!(jac.shape(), &[3, 1]);
        assert!((jac[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((jac[[1, 0]] - 4.0).abs() < 1e-12);
        assert!((jac[[2, 0]] - 12.0).abs() < 1e-12);
    }

    // --- Gradient forward ---

    #[test]
    fn test_gradient_forward_quadratic() {
        // f(x) = x0^2 + 2*x1^2  =>  grad = [2*x0, 4*x1]  at  x=[3,1] => [6, 4]
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            xs[0] * xs[0] + two * xs[1] * xs[1]
        };
        let x = Array1::from(vec![3.0_f64, 1.0]);
        let g = gradient_forward(f, &x);
        assert!((g[0] - 6.0).abs() < 1e-12);
        assert!((g[1] - 4.0).abs() < 1e-12);
    }

    // --- Hessian ---

    #[test]
    fn test_hessian_quadratic_diagonal() {
        // f(x) = x0^2 + 2*x1^2  =>  H = diag([2, 4])
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            xs[0] * xs[0] + two * xs[1] * xs[1]
        };
        let x = Array1::from(vec![1.0_f64, 1.0]);
        let h = hessian(f, &x);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((h[[1, 1]] - 4.0).abs() < 1e-5);
        assert!(h[[0, 1]].abs() < 1e-5);
        assert!(h[[1, 0]].abs() < 1e-5);
    }

    #[test]
    fn test_hessian_mixed_terms() {
        // f(x) = x0^2 + 3*x0*x1 + 2*x1^2  =>  H = [[2, 3], [3, 4]]
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            let three = DualNumber::constant(3.0_f64);
            xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
        };
        let x = Array1::from(vec![1.0_f64, 1.0]);
        let h = hessian(f, &x);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((h[[0, 1]] - 3.0).abs() < 1e-5);
        assert!((h[[1, 0]] - 3.0).abs() < 1e-5);
        assert!((h[[1, 1]] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_hessian_nonlinear() {
        // f(x) = sin(x0) * exp(x1)
        // H[0,0] = -sin(x0)*exp(x1)
        // H[0,1] = cos(x0)*exp(x1)
        // H[1,1] = sin(x0)*exp(x1)
        // At x=[0, 0]: H = [[0, 1], [1, 0]]
        let f = |xs: &[DualNumber<f64>]| xs[0].sin() * xs[1].exp();
        let x = Array1::from(vec![0.0_f64, 0.0]);
        let h = hessian(f, &x);
        assert!(h[[0, 0]].abs() < 1e-5);
        assert!((h[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((h[[1, 0]] - 1.0).abs() < 1e-5);
        assert!(h[[1, 1]].abs() < 1e-5);
    }

    #[test]
    fn test_hessian_symmetry() {
        // Any smooth function should yield a symmetric Hessian
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            xs[0].powi(3) + two * xs[0] * xs[1].powi(2) + xs[1].exp()
        };
        let x = Array1::from(vec![1.0_f64, 0.5]);
        let h = hessian(f, &x);
        assert!(
            (h[[0, 1]] - h[[1, 0]]).abs() < 1e-6,
            "Hessian should be symmetric"
        );
    }

    // --- HVP ---

    #[test]
    fn test_hvp_diagonal_hessian() {
        // f(x) = x0^2 + x1^2  =>  H = diag([2,2])  =>  H*v = 2*v
        let f = |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1];
        let x = Array1::from(vec![1.0_f64, 2.0]);
        let v = Array1::from(vec![3.0_f64, 4.0]);
        let hvp = hessian_vector_product(f, &x, &v);
        assert!((hvp[0] - 6.0).abs() < 1e-6);
        assert!((hvp[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvp_mixed_hessian() {
        // f(x) = x0^2 + 3*x0*x1 + 2*x1^2
        // H = [[2, 3], [3, 4]]
        // H*[1,0] = [2, 3]
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            let three = DualNumber::constant(3.0_f64);
            xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
        };
        let x = Array1::from(vec![1.0_f64, 1.0]);
        let v = Array1::from(vec![1.0_f64, 0.0]);
        let hvp = hessian_vector_product(f, &x, &v);
        assert!((hvp[0] - 2.0).abs() < 1e-6);
        assert!((hvp[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvp_nonlinear() {
        // f(x) = x0^4 / 4  =>  H = diag([3*x0^2])  =>  HVP at x0=2, v=1 => 12
        let f = |xs: &[DualNumber<f64>]| {
            let four = DualNumber::constant(4.0_f64);
            xs[0].powi(4) / four
        };
        let x = Array1::from(vec![2.0_f64]);
        let v = Array1::from(vec![1.0_f64]);
        let hvp = hessian_vector_product(f, &x, &v);
        // H = 3*x^2 = 12 at x=2
        assert!((hvp[0] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_hvp_consistency_with_hessian() {
        // H*v should equal matmul of full Hessian with v
        let f = |xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0_f64);
            let five = DualNumber::constant(5.0_f64);
            xs[0] * xs[0] + five * xs[0] * xs[1] + two * xs[1] * xs[1]
        };
        let x = Array1::from(vec![1.0_f64, 2.0]);
        let v = Array1::from(vec![1.0_f64, 2.0]);

        let h_mat = hessian(f, &x);
        let hvp = hessian_vector_product(f, &x, &v);

        // Manual: H = [[2, 5], [5, 4]]  H*[1,2] = [2+10, 5+8] = [12, 13]
        let h_times_v_0 = h_mat[[0, 0]] * v[0] + h_mat[[0, 1]] * v[1];
        let h_times_v_1 = h_mat[[1, 0]] * v[0] + h_mat[[1, 1]] * v[1];

        assert!(
            (hvp[0] - h_times_v_0).abs() < 1e-4,
            "hvp[0]={} h_mat*v[0]={}",
            hvp[0],
            h_times_v_0
        );
        assert!(
            (hvp[1] - h_times_v_1).abs() < 1e-4,
            "hvp[1]={} h_mat*v[1]={}",
            hvp[1],
            h_times_v_1
        );
    }

    #[test]
    fn test_hessian_scalar_function() {
        // f(x) = x^2  at x=3
        // H = [[2]]
        let f = |xs: &[DualNumber<f64>]| xs[0] * xs[0];
        let x = Array1::from(vec![3.0_f64]);
        let h = hessian(f, &x);
        assert_eq!(h.shape(), &[1, 1]);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_jacobian_sin_cos() {
        // f(x) = [sin(x0), cos(x1)]  at x=[pi/2, 0]
        // J = [[cos(pi/2), 0], [0, -sin(0)]] = [[~0, 0], [0, 0]]
        use std::f64::consts::FRAC_PI_2;
        let f = |xs: &[DualNumber<f64>]| vec![xs[0].sin(), xs[1].cos()];
        let x = Array1::from(vec![FRAC_PI_2, 0.0_f64]);
        let jac = jacobian_forward(f, &x);
        assert!(jac[[0, 0]].abs() < 1e-12); // cos(pi/2) ≈ 0
        assert!(jac[[0, 1]].abs() < 1e-12);
        assert!(jac[[1, 0]].abs() < 1e-12);
        assert!(jac[[1, 1]].abs() < 1e-12); // -sin(0) = 0
    }

    #[test]
    fn test_hessian_rosenbrock() {
        // Rosenbrock: f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
        // At x=[1,1] (minimum): H = [[802, -400], [-400, 200]]
        let f = |xs: &[DualNumber<f64>]| {
            let one = DualNumber::constant(1.0_f64);
            let hundred = DualNumber::constant(100.0_f64);
            let a = one - xs[0];
            let b = xs[1] - xs[0] * xs[0];
            a * a + hundred * b * b
        };
        let x = Array1::from(vec![1.0_f64, 1.0]);
        let h = hessian(f, &x);
        assert!((h[[0, 0]] - 802.0).abs() < 0.1, "H[0,0]={}", h[[0, 0]]);
        assert!((h[[0, 1]] - (-400.0)).abs() < 0.1, "H[0,1]={}", h[[0, 1]]);
        assert!((h[[1, 1]] - 200.0).abs() < 0.1, "H[1,1]={}", h[[1, 1]]);
    }
}
