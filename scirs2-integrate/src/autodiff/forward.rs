//! Forward mode automatic differentiation
//!
//! Forward mode AD is efficient for computing gradients when the number of
//! inputs is small compared to the number of outputs.

use super::dual::{Dual, DualVector};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Forward mode automatic differentiation engine
pub struct ForwardAD<F: IntegrateFloat> {
    /// Number of independent variables
    n_vars: usize,
    /// Tolerance for numerical operations
    tolerance: F,
}

impl<F: IntegrateFloat> ForwardAD<F> {
    /// Create a new forward AD engine
    pub fn new(n_vars: usize) -> Self {
        ForwardAD {
            n_vars,
            tolerance: F::from(1e-12).unwrap(),
        }
    }

    /// Set the tolerance
    pub fn with_tolerance(mut self, tol: F) -> Self {
        self.tolerance = tol;
        self
    }

    /// Compute gradient using forward mode AD
    pub fn gradient<Func>(&self, f: Func, x: ArrayView1<F>) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&[Dual<F>]) -> Dual<F>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        let mut gradient = Array1::zeros(self.n_vars);

        // Compute partial derivatives one at a time
        for i in 0..self.n_vars {
            let mut dual_x = Vec::with_capacity(self.n_vars);
            for (j, &val) in x.iter().enumerate() {
                if i == j {
                    dual_x.push(Dual::variable(val));
                } else {
                    dual_x.push(Dual::constant(val));
                }
            }

            let result = f(&dual_x);
            gradient[i] = result.derivative();
        }

        Ok(gradient)
    }

    /// Compute Jacobian using forward mode AD
    pub fn jacobian<Func>(&self, f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(&[Dual<F>]) -> Vec<Dual<F>>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        // First, determine output dimension
        let constant_x: Vec<_> = x.iter().map(|&v| Dual::constant(v)).collect();
        let output = f(&constant_x);
        let m = output.len();

        let mut jacobian = Array2::zeros((m, self.n_vars));

        // Compute column by column
        for j in 0..self.n_vars {
            let mut dual_x = Vec::with_capacity(self.n_vars);
            for (k, &val) in x.iter().enumerate() {
                if j == k {
                    dual_x.push(Dual::variable(val));
                } else {
                    dual_x.push(Dual::constant(val));
                }
            }

            let result = f(&dual_x);
            for (i, res) in result.iter().enumerate() {
                jacobian[[i, j]] = res.derivative();
            }
        }

        Ok(jacobian)
    }

    /// Compute directional derivative
    pub fn directional_derivative<Func>(
        &self,
        f: Func,
        x: ArrayView1<F>,
        direction: ArrayView1<F>,
    ) -> IntegrateResult<F>
    where
        Func: Fn(&[Dual<F>]) -> Dual<F>,
    {
        if x.len() != self.n_vars || direction.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got x: {}, direction: {}",
                self.n_vars,
                x.len(),
                direction.len()
            )));
        }

        let dual_x: Vec<_> = x
            .iter()
            .zip(direction.iter())
            .map(|(&val, &der)| Dual::new(val, der))
            .collect();

        let result = f(&dual_x);
        Ok(result.derivative())
    }
}

/// Compute gradient using forward mode AD (convenience function)
#[allow(dead_code)]
pub fn forward_gradient<F, Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(&[Dual<F>]) -> Dual<F>,
{
    let ad = ForwardAD::new(x.len());
    ad.gradient(f, x)
}

/// Compute Jacobian using forward mode AD (convenience function)
#[allow(dead_code)]
pub fn forward_jacobian<F, Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(&[Dual<F>]) -> Vec<Dual<F>>,
{
    let ad = ForwardAD::new(x.len());
    ad.jacobian(f, x)
}

/// Example: Rosenbrock function gradient
#[allow(dead_code)]
pub fn example_rosenbrock_gradient<F: IntegrateFloat>() -> IntegrateResult<()> {
    // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    let rosenbrock = |x: &[Dual<F>]| {
        let one = Dual::constant(F::one());
        let hundred = Dual::constant(F::from(100.0).unwrap());

        let term1 = (one - x[0]) * (one - x[0]);
        let term2 = hundred * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);

        term1 + term2
    };

    let x = Array1::from_vec(vec![F::from(1.0).unwrap(), F::from(2.0).unwrap()]);
    let grad = forward_gradient(rosenbrock, x.view())?;

    println!("Gradient at (1,2): {grad:?}");
    Ok(())
}

/// Forward mode AD for ODE right-hand side functions
pub struct ForwardODEJacobian<F: IntegrateFloat> {
    _n_states: usize,
    ad_engine: ForwardAD<F>,
}

impl<F: IntegrateFloat> ForwardODEJacobian<F> {
    /// Create a new ODE Jacobian computer
    pub fn new(_nstates: usize) -> Self {
        ForwardODEJacobian {
            _n_states: _nstates,
            ad_engine: ForwardAD::new(_nstates),
        }
    }

    /// Compute Jacobian for ODE system dy/dt = f(t, y)
    pub fn compute<Func>(&self, f: Func, t: F, y: ArrayView1<F>) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(F, &[Dual<F>]) -> Vec<Dual<F>>,
    {
        self.ad_engine.jacobian(|dual_y| f(t, dual_y), y)
    }
}

/// Vectorized forward mode AD for computing multiple directional derivatives
pub struct VectorizedForwardAD<F: IntegrateFloat> {
    n_vars: usize,
    n_directions: usize,
    #[allow(dead_code)]
    tolerance: F,
}

impl<F: IntegrateFloat> VectorizedForwardAD<F> {
    /// Create a new vectorized forward AD engine
    pub fn new(n_vars: usize, n_directions: usize) -> Self {
        VectorizedForwardAD {
            n_vars,
            n_directions,
            tolerance: F::from(1e-12).unwrap(),
        }
    }

    /// Compute multiple directional derivatives simultaneously
    pub fn directional_derivatives<Func>(
        &self,
        f: Func,
        x: ArrayView1<F>,
        directions: &[ArrayView1<F>],
    ) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&[DualVector<F>]) -> DualVector<F>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        if directions.len() != self.n_directions {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} directions, got {}",
                self.n_directions,
                directions.len()
            )));
        }

        // Create dual vectors with multiple derivative components
        let mut dual_x = Vec::with_capacity(self.n_vars);
        for i in 0..self.n_vars {
            let mut derivatives = Array1::zeros(self.n_directions);
            for (j, dir) in directions.iter().enumerate() {
                derivatives[j] = dir[i];
            }
            dual_x.push(DualVector {
                values: Array1::from_elem(1, x[i]),
                jacobian: Array1::from_elem(1, derivatives),
            });
        }

        let result = f(&dual_x);
        Ok(result.jacobian[0].clone())
    }

    /// Compute Jacobian using vectorized forward mode
    pub fn jacobian_vectorized<Func>(
        &self,
        f: Func,
        x: ArrayView1<F>,
        chunk_size: usize,
    ) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(&[DualVector<F>]) -> Vec<DualVector<F>> + Clone,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        // First, determine output dimension
        let constant_x: Vec<_> = (0..self.n_vars)
            .map(|i| DualVector::constant(Array1::from_elem(1, x[i])))
            .collect();
        let output = f(&constant_x);
        let m = output.len();

        let mut jacobian = Array2::zeros((m, self.n_vars));

        // Process columns in chunks for better cache efficiency
        for chunk_start in (0..self.n_vars).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(self.n_vars);
            let chunk_width = chunk_end - chunk_start;

            // Create dual vectors for this chunk
            let mut dual_x = Vec::with_capacity(self.n_vars);
            for i in 0..self.n_vars {
                let mut derivatives = Array1::zeros(chunk_width);
                if i >= chunk_start && i < chunk_end {
                    derivatives[i - chunk_start] = F::one();
                }
                dual_x.push(DualVector {
                    values: Array1::from_elem(1, x[i]),
                    jacobian: Array1::from_elem(1, derivatives),
                });
            }

            let result = f(&dual_x);
            for (i, res) in result.iter().enumerate() {
                for j in 0..chunk_width {
                    jacobian[[i, chunk_start + j]] = res.jacobian[0][j];
                }
            }
        }

        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_gradient() {
        // Test gradient of f(x,y) = x^2 + y^2
        let f = |x: &[Dual<f64>]| x[0] * x[0] + x[1] * x[1];

        let x = Array1::from_vec(vec![3.0, 4.0]);
        let grad = forward_gradient(f, x.view()).unwrap();

        // Gradient should be [2x, 2y] = [6, 8]
        assert!((grad[0] - 6.0).abs() < 1e-10);
        assert!((grad[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_forward_jacobian() {
        // Test Jacobian of f(x,y) = [x^2, x*y, y^2]
        let f = |x: &[Dual<f64>]| vec![x[0] * x[0], x[0] * x[1], x[1] * x[1]];

        let x = Array1::from_vec(vec![2.0, 3.0]);
        let jac = forward_jacobian(f, x.view()).unwrap();

        // Jacobian should be:
        // [[2x, 0 ],
        //  [y,  x ],
        //  [0,  2y]]
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-10); // 2*2
        assert!((jac[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-10); // y
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-10); // x
        assert!((jac[[2, 0]] - 0.0).abs() < 1e-10);
        assert!((jac[[2, 1]] - 6.0).abs() < 1e-10); // 2*3
    }
}
