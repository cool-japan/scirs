//! `UnitAware` — a [`LoweredOp`] paired with a [`Dimension`].
//!
//! See [`super`] for module-level overview.

use crate::eml::op::LoweredOp;
use crate::units::dimension::Dimension;

/// Errors returned by dimensional arithmetic.
#[derive(Debug, Clone, thiserror::Error)]
pub enum UnitError {
    /// Operands of an additive operator have different dimensions.
    #[error("dimension mismatch: {0} vs {1}")]
    Mismatch(Dimension, Dimension),
    /// A transcendental (sin/cos/exp/ln/...) was given a dimensional argument.
    #[error("transcendental `{op}` requires dimensionless argument; got {dim}")]
    NonDimensionlessTranscendental {
        /// Name of the transcendental operator (e.g. `"sin"`, `"exp"`).
        op: &'static str,
        /// The dimension of the argument.
        dim: Dimension,
    },
    /// A non-integer power was applied to a dimensional quantity, or
    /// `sqrt` was applied to a dimension whose exponents are not all even.
    #[error(
        "non-integer power on dimensional quantity ({0}^{1}); only integer powers preserve dimensional structure"
    )]
    NonIntegerPower(Dimension, f64),
    /// A `Sqrt` was applied to a dimension with odd exponents, which would
    /// produce a fractional (non-representable) dimension.
    #[error("sqrt applied to non-square dimension {0}; all exponents must be even")]
    NonSquareSqrt(Dimension),
    /// A `Var(i)` index exceeds the provided `var_dims` slice length.
    #[error("Var({0}) index out of range for var_dims slice")]
    VarIndexOutOfRange(usize),
    /// An internal stack invariant was violated during iterative traversal.
    #[error("internal dimensional-inference error: {0}")]
    Internal(&'static str),
}

/// A [`LoweredOp`] paired with its [`Dimension`] (the units of its result).
///
/// Use the constructor methods ([`UnitAware::constant`], [`UnitAware::var`])
/// to build leaves and the arithmetic methods (`add`, `mul`, ...) to
/// compose them; the dimensional bookkeeping is automatic.
#[derive(Clone, Debug)]
pub struct UnitAware {
    /// The underlying expression.
    pub op: LoweredOp,
    /// The dimensional units of `op`'s result.
    pub dim: Dimension,
}

impl UnitAware {
    /// Construct from op + dimension.
    pub fn new(op: LoweredOp, dim: Dimension) -> Self {
        Self { op, dim }
    }

    /// Construct a dimensionless constant.
    pub fn constant(c: f64) -> Self {
        Self {
            op: LoweredOp::Const(c),
            dim: Dimension::dimensionless(),
        }
    }

    /// Construct a variable with the given dimension.
    pub fn var(idx: usize, dim: Dimension) -> Self {
        Self {
            op: LoweredOp::Var(idx),
            dim,
        }
    }

    /// Add: dimensions must match.
    pub fn add(&self, other: &Self) -> Result<Self, UnitError> {
        if self.dim != other.dim {
            return Err(UnitError::Mismatch(self.dim, other.dim));
        }
        Ok(Self {
            op: LoweredOp::Add(Box::new(self.op.clone()), Box::new(other.op.clone())),
            dim: self.dim,
        })
    }

    /// Sub: dimensions must match.
    pub fn sub(&self, other: &Self) -> Result<Self, UnitError> {
        if self.dim != other.dim {
            return Err(UnitError::Mismatch(self.dim, other.dim));
        }
        Ok(Self {
            op: LoweredOp::Sub(Box::new(self.op.clone()), Box::new(other.op.clone())),
            dim: self.dim,
        })
    }

    /// Mul: dimensions multiply.
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            op: LoweredOp::Mul(Box::new(self.op.clone()), Box::new(other.op.clone())),
            dim: self.dim.product(&other.dim),
        }
    }

    /// Div: dimensions divide.
    pub fn div(&self, other: &Self) -> Self {
        Self {
            op: LoweredOp::Div(Box::new(self.op.clone()), Box::new(other.op.clone())),
            dim: self.dim.quotient(&other.dim),
        }
    }

    /// Power by an integer constant: dimensions multiply by the integer.
    ///
    /// Non-integer powers on dimensional quantities are an error per
    /// dimensional analysis; use [`UnitAware::pow_dimensionless`] when
    /// the base is dimensionless and the exponent is fractional.
    pub fn pow_int(&self, n: i32) -> Self {
        Self {
            op: LoweredOp::Pow(
                Box::new(self.op.clone()),
                Box::new(LoweredOp::Const(n as f64)),
            ),
            dim: self.dim.power_int(n),
        }
    }

    /// Power for dimensionless base + dimensionless exponent (`f64`).
    ///
    /// Both base and exponent must be dimensionless; the result is
    /// dimensionless.  Errors otherwise.
    pub fn pow_dimensionless(&self, exponent: f64) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonIntegerPower(self.dim, exponent));
        }
        Ok(Self {
            op: LoweredOp::Pow(
                Box::new(self.op.clone()),
                Box::new(LoweredOp::Const(exponent)),
            ),
            dim: Dimension::dimensionless(),
        })
    }

    /// Sqrt: only valid if all exponents are even (in which case they halve).
    pub fn sqrt(&self) -> Result<Self, UnitError> {
        if self.dim.exp.iter().any(|&e| e % 2 != 0) {
            return Err(UnitError::NonIntegerPower(self.dim, 0.5));
        }
        let mut new_exp = [0; 7];
        for (i, e) in new_exp.iter_mut().enumerate() {
            *e = self.dim.exp[i] / 2;
        }
        Ok(Self {
            op: LoweredOp::Sqrt(Box::new(self.op.clone())),
            dim: Dimension::new(new_exp),
        })
    }

    /// Sin: argument must be dimensionless.
    pub fn sin(&self) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonDimensionlessTranscendental {
                op: "sin",
                dim: self.dim,
            });
        }
        Ok(Self {
            op: LoweredOp::Sin(Box::new(self.op.clone())),
            dim: Dimension::dimensionless(),
        })
    }

    /// Cos: argument must be dimensionless.
    pub fn cos(&self) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonDimensionlessTranscendental {
                op: "cos",
                dim: self.dim,
            });
        }
        Ok(Self {
            op: LoweredOp::Cos(Box::new(self.op.clone())),
            dim: Dimension::dimensionless(),
        })
    }

    /// Tan: argument must be dimensionless.
    pub fn tan(&self) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonDimensionlessTranscendental {
                op: "tan",
                dim: self.dim,
            });
        }
        Ok(Self {
            op: LoweredOp::Tan(Box::new(self.op.clone())),
            dim: Dimension::dimensionless(),
        })
    }

    /// Exp: argument must be dimensionless.
    pub fn exp(&self) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonDimensionlessTranscendental {
                op: "exp",
                dim: self.dim,
            });
        }
        Ok(Self {
            op: LoweredOp::Exp(Box::new(self.op.clone())),
            dim: Dimension::dimensionless(),
        })
    }

    /// Ln: argument must be dimensionless.
    pub fn ln(&self) -> Result<Self, UnitError> {
        if !self.dim.is_dimensionless() {
            return Err(UnitError::NonDimensionlessTranscendental {
                op: "ln",
                dim: self.dim,
            });
        }
        Ok(Self {
            op: LoweredOp::Ln(Box::new(self.op.clone())),
            dim: Dimension::dimensionless(),
        })
    }

    /// Negation preserves dimension.
    pub fn neg(&self) -> Self {
        Self {
            op: LoweredOp::Neg(Box::new(self.op.clone())),
            dim: self.dim,
        }
    }

    /// Absolute value preserves dimension.
    pub fn abs(&self) -> Self {
        Self {
            op: LoweredOp::Abs(Box::new(self.op.clone())),
            dim: self.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::dimension::Dimension;

    #[test]
    fn add_compatible_dimensions() {
        let v1 = UnitAware::var(0, Dimension::length());
        let v2 = UnitAware::var(1, Dimension::length());
        let s = v1.add(&v2).expect("compatible add");
        assert_eq!(s.dim, Dimension::length());
    }

    #[test]
    fn add_incompatible_returns_err() {
        let v1 = UnitAware::var(0, Dimension::length());
        let v2 = UnitAware::var(1, Dimension::time());
        assert!(matches!(v1.add(&v2), Err(UnitError::Mismatch(_, _))));
    }

    #[test]
    fn sub_compatible_dimensions() {
        let v1 = UnitAware::var(0, Dimension::mass());
        let v2 = UnitAware::var(1, Dimension::mass());
        let d = v1.sub(&v2).expect("compatible sub");
        assert_eq!(d.dim, Dimension::mass());
    }

    #[test]
    fn sub_incompatible_returns_err() {
        let v1 = UnitAware::var(0, Dimension::mass());
        let v2 = UnitAware::var(1, Dimension::length());
        assert!(matches!(v1.sub(&v2), Err(UnitError::Mismatch(_, _))));
    }

    #[test]
    fn mul_combines_dimensions() {
        let l = UnitAware::var(0, Dimension::length());
        let t = UnitAware::var(1, Dimension::time());
        let lt = l.mul(&t);
        assert_eq!(lt.dim.exp[0], 1);
        assert_eq!(lt.dim.exp[2], 1);
    }

    #[test]
    fn velocity_from_length_div_time() {
        let l = UnitAware::var(0, Dimension::length());
        let t = UnitAware::var(1, Dimension::time());
        let v = l.div(&t);
        assert_eq!(v.dim, Dimension::velocity());
    }

    #[test]
    fn sin_dimensional_returns_err() {
        let l = UnitAware::var(0, Dimension::length());
        assert!(matches!(
            l.sin(),
            Err(UnitError::NonDimensionlessTranscendental { .. })
        ));
    }

    #[test]
    fn sin_dimensionless_works() {
        let theta = UnitAware::var(0, Dimension::dimensionless());
        let s = theta.sin().expect("dimensionless sin");
        assert!(s.dim.is_dimensionless());
    }

    #[test]
    fn cos_dimensional_returns_err() {
        let l = UnitAware::var(0, Dimension::length());
        assert!(matches!(
            l.cos(),
            Err(UnitError::NonDimensionlessTranscendental { .. })
        ));
    }

    #[test]
    fn tan_dimensionless_works() {
        let theta = UnitAware::var(0, Dimension::dimensionless());
        let t = theta.tan().expect("dimensionless tan");
        assert!(t.dim.is_dimensionless());
    }

    #[test]
    fn exp_of_dimensional_errs() {
        let m = UnitAware::var(0, Dimension::mass());
        assert!(matches!(
            m.exp(),
            Err(UnitError::NonDimensionlessTranscendental { .. })
        ));
    }

    #[test]
    fn ln_dimensionless_works() {
        let x = UnitAware::var(0, Dimension::dimensionless());
        assert!(x.ln().is_ok());
    }

    #[test]
    fn pow_int_squares_dimension() {
        let v = UnitAware::var(0, Dimension::velocity());
        let v_sq = v.pow_int(2);
        // v^2 has units m^2·s^-2
        assert_eq!(v_sq.dim.exp[0], 2);
        assert_eq!(v_sq.dim.exp[2], -2);
    }

    #[test]
    fn pow_int_negative_inverts() {
        let l = UnitAware::var(0, Dimension::length());
        let inv = l.pow_int(-1);
        assert_eq!(inv.dim.exp[0], -1);
    }

    #[test]
    fn pow_dimensionless_on_dimensional_errs() {
        let l = UnitAware::var(0, Dimension::length());
        assert!(matches!(
            l.pow_dimensionless(0.5),
            Err(UnitError::NonIntegerPower(_, _))
        ));
    }

    #[test]
    fn pow_dimensionless_on_dimensionless_ok() {
        let x = UnitAware::var(0, Dimension::dimensionless());
        let y = x.pow_dimensionless(0.5).expect("dimensionless pow");
        assert!(y.dim.is_dimensionless());
    }

    #[test]
    fn sqrt_of_area_is_length() {
        let area = UnitAware::var(0, Dimension::length().power_int(2));
        let l = area.sqrt().expect("sqrt of area");
        assert_eq!(l.dim, Dimension::length());
    }

    #[test]
    fn sqrt_of_odd_dim_returns_err() {
        let v = UnitAware::var(0, Dimension::length()); // [m^1]
        assert!(matches!(v.sqrt(), Err(UnitError::NonIntegerPower(_, _))));
    }

    #[test]
    fn sqrt_of_dimensionless_is_dimensionless() {
        let x = UnitAware::var(0, Dimension::dimensionless());
        let r = x.sqrt().expect("sqrt of dimensionless");
        assert!(r.dim.is_dimensionless());
    }

    #[test]
    fn neg_preserves_dimension() {
        let m = UnitAware::var(0, Dimension::mass());
        let nm = m.neg();
        assert_eq!(nm.dim, Dimension::mass());
        match nm.op {
            LoweredOp::Neg(_) => {}
            _ => panic!("expected Neg op"),
        }
    }

    #[test]
    fn abs_preserves_dimension() {
        let m = UnitAware::var(0, Dimension::force());
        let am = m.abs();
        assert_eq!(am.dim, Dimension::force());
    }

    #[test]
    fn constant_is_dimensionless() {
        let c = UnitAware::constant(2.71);
        assert!(c.dim.is_dimensionless());
        assert!(matches!(c.op, LoweredOp::Const(_)));
    }

    #[test]
    fn new_constructor_passes_through() {
        let u = UnitAware::new(LoweredOp::Var(3), Dimension::time());
        assert_eq!(u.dim, Dimension::time());
    }

    #[test]
    fn kinetic_energy_pipeline() {
        // KE = (1/2)·m·v^2 should have units of energy
        let half = UnitAware::constant(0.5);
        let m = UnitAware::var(0, Dimension::mass());
        let v = UnitAware::var(1, Dimension::velocity());
        let v_sq = v.pow_int(2);
        let mv_sq = m.mul(&v_sq);
        let ke = half.mul(&mv_sq);
        assert_eq!(ke.dim, Dimension::energy());
    }

    #[test]
    fn gravitational_pe_pipeline() {
        // U = m·g·h should have units of energy.
        let m = UnitAware::var(0, Dimension::mass());
        let g = UnitAware::var(1, Dimension::acceleration());
        let h = UnitAware::var(2, Dimension::length());
        let mg = m.mul(&g);
        let pe = mg.mul(&h);
        assert_eq!(pe.dim, Dimension::energy());
    }

    #[test]
    fn velocity_squared_then_sqrt_recovers_velocity() {
        let v = UnitAware::var(0, Dimension::velocity());
        let v_sq = v.pow_int(2);
        let recovered = v_sq.sqrt().expect("sqrt of v^2");
        assert_eq!(recovered.dim, Dimension::velocity());
    }
}
