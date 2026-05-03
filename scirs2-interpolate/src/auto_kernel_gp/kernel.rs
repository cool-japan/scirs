//! Kernel expressions for automatic kernel structure discovery.
//!
//! Provides a closed grammar of base kernels and composite expressions formed
//! by sum and product operators.  Each [`KernelExpr`] can be evaluated at a
//! pair of scalar inputs and described as a human-readable string.
//!
//! ## Base kernels
//!
//! | Variant | Formula |
//! |---------|---------|
//! | `Rbf` | exp(-‖x-y‖² / (2 ℓ²)) |
//! | `Matern52` | (1 + √5·r/ℓ + 5r²/(3ℓ²)) exp(-√5·r/ℓ) |
//! | `Periodic` | exp(-2 sin²(π‖x-y‖/p) / ℓ²) |
//! | `Linear` | σ² · x · y |
//! | `WhiteNoise` | σ² · 𝟙{x == y} |

/// One of the atomic kernel functions in the grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum BaseKernel {
    /// Squared-exponential (RBF) kernel with length scale ℓ > 0.
    Rbf { length_scale: f64 },
    /// Matérn 5/2 kernel with length scale ℓ > 0.
    Matern52 { length_scale: f64 },
    /// Periodic kernel with period p > 0 and length scale ℓ > 0.
    Periodic { period: f64, length_scale: f64 },
    /// Linear kernel with variance σ² > 0: k(x, y) = σ² · x · y.
    Linear { variance: f64 },
    /// White-noise kernel: k(x, x) = σ², k(x, y) = 0 for x ≠ y.
    WhiteNoise { variance: f64 },
}

impl BaseKernel {
    /// Evaluate k(x1, x2).
    pub fn eval(&self, x1: f64, x2: f64) -> f64 {
        match self {
            BaseKernel::Rbf { length_scale } => {
                let ell = length_scale.max(1e-10);
                let d = x1 - x2;
                (-d * d / (2.0 * ell * ell)).exp()
            }
            BaseKernel::Matern52 { length_scale } => {
                let ell = length_scale.max(1e-10);
                let r = (x1 - x2).abs();
                let s = 5.0_f64.sqrt() * r / ell;
                (1.0 + s + s * s / 3.0) * (-s).exp()
            }
            BaseKernel::Periodic {
                period,
                length_scale,
            } => {
                let p = period.max(1e-10);
                let ell = length_scale.max(1e-10);
                let d = (x1 - x2).abs();
                let sin_val = (std::f64::consts::PI * d / p).sin();
                (-2.0 * sin_val * sin_val / (ell * ell)).exp()
            }
            BaseKernel::Linear { variance } => variance * x1 * x2,
            BaseKernel::WhiteNoise { variance } => {
                if (x1 - x2).abs() < 1e-12 {
                    *variance
                } else {
                    0.0
                }
            }
        }
    }

    /// Short human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            BaseKernel::Rbf { .. } => "RBF",
            BaseKernel::Matern52 { .. } => "Matern52",
            BaseKernel::Periodic { .. } => "Periodic",
            BaseKernel::Linear { .. } => "Linear",
            BaseKernel::WhiteNoise { .. } => "WhiteNoise",
        }
    }

    /// Return a version of this kernel with updated hyperparameters.
    pub fn with_length_scale(&self, new_ell: f64) -> Self {
        match self {
            BaseKernel::Rbf { .. } => BaseKernel::Rbf {
                length_scale: new_ell,
            },
            BaseKernel::Matern52 { .. } => BaseKernel::Matern52 {
                length_scale: new_ell,
            },
            BaseKernel::Periodic { period, .. } => BaseKernel::Periodic {
                period: *period,
                length_scale: new_ell,
            },
            BaseKernel::Linear { .. } => BaseKernel::Linear { variance: new_ell },
            BaseKernel::WhiteNoise { .. } => BaseKernel::WhiteNoise { variance: new_ell },
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel expression tree
// ---------------------------------------------------------------------------

/// A compositional kernel expression formed by summing or multiplying sub-kernels.
///
/// The expression depth is defined recursively:
/// - `Base(k)` has depth 0.
/// - `Sum(a, b)` and `Product(a, b)` have depth 1 + max(depth(a), depth(b)).
#[derive(Debug, Clone)]
pub enum KernelExpr {
    /// A single base kernel.
    Base(BaseKernel),
    /// Sum of two kernel expressions: k(x, y) = a(x, y) + b(x, y).
    Sum(Box<KernelExpr>, Box<KernelExpr>),
    /// Product of two kernel expressions: k(x, y) = a(x, y) · b(x, y).
    Product(Box<KernelExpr>, Box<KernelExpr>),
}

impl KernelExpr {
    /// Evaluate the kernel at `(x1, x2)`.
    pub fn eval(&self, x1: f64, x2: f64) -> f64 {
        match self {
            KernelExpr::Base(k) => k.eval(x1, x2),
            KernelExpr::Sum(a, b) => a.eval(x1, x2) + b.eval(x1, x2),
            KernelExpr::Product(a, b) => a.eval(x1, x2) * b.eval(x1, x2),
        }
    }

    /// Human-readable description, e.g. `"RBF + Periodic × Linear"`.
    pub fn description(&self) -> String {
        match self {
            KernelExpr::Base(k) => k.name().to_string(),
            KernelExpr::Sum(a, b) => format!("{} + {}", a.description(), b.description()),
            KernelExpr::Product(a, b) => {
                // Add parens around sum sub-expressions to make precedence clear.
                let ad = a.description();
                let bd = b.description();
                let left = if matches!(a.as_ref(), KernelExpr::Sum(_, _)) {
                    format!("({ad})")
                } else {
                    ad
                };
                let right = if matches!(b.as_ref(), KernelExpr::Sum(_, _)) {
                    format!("({bd})")
                } else {
                    bd
                };
                format!("{left} × {right}")
            }
        }
    }

    /// Compute the full depth of this expression tree.
    pub fn depth(&self) -> usize {
        match self {
            KernelExpr::Base(_) => 0,
            KernelExpr::Sum(a, b) | KernelExpr::Product(a, b) => 1 + a.depth().max(b.depth()),
        }
    }
}

/// Enumerate all base kernels with initial hyperparameters.
pub fn base_kernels() -> Vec<BaseKernel> {
    vec![
        BaseKernel::Rbf { length_scale: 1.0 },
        BaseKernel::Matern52 { length_scale: 1.0 },
        BaseKernel::Periodic {
            period: 1.0,
            length_scale: 1.0,
        },
        BaseKernel::Linear { variance: 1.0 },
        BaseKernel::WhiteNoise { variance: 0.1 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_expr_eval_rbf_is_symmetric() {
        let k = KernelExpr::Base(BaseKernel::Rbf { length_scale: 1.0 });
        let a = k.eval(0.5, 1.5);
        let b = k.eval(1.5, 0.5);
        assert!(
            (a - b).abs() < 1e-12,
            "RBF must be symmetric: k(0.5,1.5)={a}, k(1.5,0.5)={b}"
        );
    }

    #[test]
    fn kernel_expr_sum_is_additive() {
        let k1 = KernelExpr::Base(BaseKernel::Rbf { length_scale: 1.0 });
        let k2 = KernelExpr::Base(BaseKernel::Matern52 { length_scale: 1.0 });
        let sum = KernelExpr::Sum(Box::new(k1.clone()), Box::new(k2.clone()));
        let x = 0.3;
        let y = 1.2;
        let expected = k1.eval(x, y) + k2.eval(x, y);
        let actual = sum.eval(x, y);
        assert!(
            (actual - expected).abs() < 1e-12,
            "Sum kernel must equal sum of parts: {actual} vs {expected}"
        );
    }

    #[test]
    fn kernel_expr_product_is_multiplicative() {
        let k1 = KernelExpr::Base(BaseKernel::Rbf { length_scale: 0.8 });
        let k2 = KernelExpr::Base(BaseKernel::Linear { variance: 2.0 });
        let prod = KernelExpr::Product(Box::new(k1.clone()), Box::new(k2.clone()));
        let x = 1.5;
        let y = 0.5;
        let expected = k1.eval(x, y) * k2.eval(x, y);
        let actual = prod.eval(x, y);
        assert!(
            (actual - expected).abs() < 1e-12,
            "Product kernel must equal product of parts: {actual} vs {expected}"
        );
    }

    #[test]
    fn kernel_description_nonempty() {
        let k = KernelExpr::Sum(
            Box::new(KernelExpr::Base(BaseKernel::Rbf { length_scale: 1.0 })),
            Box::new(KernelExpr::Product(
                Box::new(KernelExpr::Base(BaseKernel::Periodic {
                    period: 1.0,
                    length_scale: 0.5,
                })),
                Box::new(KernelExpr::Base(BaseKernel::Linear { variance: 1.0 })),
            )),
        );
        let desc = k.description();
        assert!(!desc.is_empty(), "description must not be empty");
        assert!(desc.contains("RBF"), "description should contain RBF");
        assert!(
            desc.contains("Periodic"),
            "description should contain Periodic"
        );
    }
}
