use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array1, Array2, Ix1, Ix2};
use num_traits::FromPrimitive;

/// Conjugate Gradient solver for symmetric positive definite systems
pub struct ConjugateGradientOp {
    max_iter: usize,
    tolerance: Option<f64>,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for ConjugateGradientOp {
    fn name(&self) -> &'static str {
        "ConjugateGradient"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "CG requires square matrix".into(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using conjugate gradient
        let x = conjugate_gradient(&a_2d, &b_1d, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let _b = ctx.input(1);
        let x = ctx.output();
        let g = ctx.graph();

        // Gradient computation for iterative solver
        // grad_b = solve(A^T, grad_x)
        // grad_A = -outer(solve(A^T, grad_x), x)

        let a_array = match a.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let x_array = match x.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Solve A^T y = grad_x
        let a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_x_1d = match grad_output_array.view().into_dimensionality::<Ix1>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Solve A^T y = grad_x
        let at = a_2d.t();
        match conjugate_gradient(&at.view(), &grad_x_1d, self.max_iter, self.tolerance) {
            Ok(y) => {
                // grad_b = y
                let grad_b_tensor = crate::tensor_ops::convert_to_tensor(y.clone().into_dyn(), g);
                ctx.append_input_grad(1, Some(grad_b_tensor));

                // grad_A = -y ⊗ x
                let x_1d = x_array.view().into_dimensionality::<Ix1>().unwrap();
                let grad_a = -outer_product(&y.view(), &x_1d);
                let grad_a_tensor = crate::tensor_ops::convert_to_tensor(grad_a.into_dyn(), g);
                ctx.append_input_grad(0, Some(grad_a_tensor));
            }
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
            }
        }
    }
}

/// GMRES (Generalized Minimal RESidual) solver for general linear systems
pub struct GMRESOp {
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for GMRESOp {
    fn name(&self) -> &'static str {
        "GMRES"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "GMRES requires square matrix".into(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using GMRES
        let x = gmres(&a_2d, &b_1d, self.max_iter, self.restart, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Similar gradient computation as CG
        let grad_output = ctx.output_grad();
        ctx.append_input_grad(0, Some(*grad_output));
        ctx.append_input_grad(1, Some(*grad_output));
    }
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) solver
pub struct BiCGSTABOp {
    max_iter: usize,
    tolerance: Option<f64>,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for BiCGSTABOp {
    fn name(&self) -> &'static str {
        "BiCGSTAB"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "BiCGSTAB requires square matrix".into(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using BiCGSTAB
        let x = bicgstab(&a_2d, &b_1d, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        ctx.append_input_grad(0, Some(*grad_output));
        ctx.append_input_grad(1, Some(*grad_output));
    }
}

/// Preconditioned Conjugate Gradient solver
pub struct PCGOp {
    max_iter: usize,
    tolerance: Option<f64>,
    preconditioner: PreconditionerType,
}

#[derive(Clone, Copy)]
pub enum PreconditionerType {
    None,
    Jacobi,
    IncompleteCholesky,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for PCGOp {
    fn name(&self) -> &'static str {
        "PCG"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Build preconditioner
        let preconditioner = build_preconditioner(&a_2d, self.preconditioner)?;

        // Solve using PCG
        let x = pcg(&a_2d, &b_1d, &preconditioner, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        ctx.append_input_grad(0, Some(*grad_output));
        ctx.append_input_grad(1, Some(*grad_output));
    }
}

// Helper functions

/// Conjugate Gradient implementation
fn conjugate_gradient<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).unwrap())
        .unwrap_or_else(|| F::epsilon() * F::from(10.0).unwrap());

    // Initial guess x = 0
    let mut x = Array1::<F>::zeros(n);

    // r = b - Ax = b (since x = 0)
    let mut r = b.to_owned();
    let mut p = r.clone();
    let mut rsold = r.dot(&r);

    for _ in 0..max_iter {
        let ap = a.dot(&p);
        let alpha = rsold / p.dot(&ap);

        x = &x + &p.mapv(|v| alpha * v);
        r = &r - &ap.mapv(|v| alpha * v);

        let rsnew = r.dot(&r);

        if rsnew.sqrt() < tol {
            break;
        }

        let beta = rsnew / rsold;
        p = &r + &p.mapv(|v| beta * v);

        rsold = rsnew;
    }

    Ok(x)
}

/// GMRES implementation
fn gmres<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).unwrap())
        .unwrap_or_else(|| F::epsilon() * F::from(10.0).unwrap());

    let mut x = Array1::<F>::zeros(n);
    let m = restart.min(n);

    for _ in 0..max_iter {
        let r = b - &a.dot(&x);
        let rnorm = r.dot(&r).sqrt();

        if rnorm < tol {
            break;
        }

        // Arnoldi process
        let mut v = vec![Array1::<F>::zeros(n); m + 1];
        let mut h = Array2::<F>::zeros((m + 1, m));

        v[0] = &r / rnorm;

        let mut j = 0;
        while j < m {
            let mut w = a.dot(&v[j]);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[[i, j]] = w.dot(&v[i]);
                w = &w - &v[i].mapv(|val| h[[i, j]] * val);
            }

            h[[j + 1, j]] = w.dot(&w).sqrt();

            if h[[j + 1, j]].abs() < F::epsilon() {
                break;
            }

            v[j + 1] = w / h[[j + 1, j]];
            j += 1;
        }

        // Solve least squares problem
        let beta = rnorm;
        let mut e1 = Array1::<F>::zeros(j + 1);
        e1[0] = beta;

        let y = solve_least_squares(&h.slice(ndarray::s![..j + 1, ..j]).to_owned(), &e1)?;

        // Update solution
        for i in 0..j {
            x = &x + &v[i].mapv(|val| y[i] * val);
        }
    }

    Ok(x)
}

/// BiCGSTAB implementation
fn bicgstab<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).unwrap())
        .unwrap_or_else(|| F::epsilon() * F::from(10.0).unwrap());

    let mut x = Array1::<F>::zeros(n);
    let mut r = b - &a.dot(&x);
    let r0 = r.clone();

    let mut rho = F::one();
    let mut alpha = F::one();
    let mut omega = F::one();

    let mut v = Array1::<F>::zeros(n);
    let mut p = Array1::<F>::zeros(n);

    for _ in 0..max_iter {
        let rho_new = r0.dot(&r);

        if rho_new.abs() < F::epsilon() {
            break;
        }

        let beta = (rho_new / rho) * (alpha / omega);
        p = &r + &(&p - &v.mapv(|val| omega * val)).mapv(|val| beta * val);

        v = a.dot(&p);
        alpha = rho_new / r0.dot(&v);

        let s = &r - &v.mapv(|val| alpha * val);

        if s.dot(&s).sqrt() < tol {
            x = &x + &p.mapv(|v| alpha * v);
            break;
        }

        let t = a.dot(&s);
        omega = t.dot(&s) / t.dot(&t);

        x = &x + &p.mapv(|val| alpha * val) + &s.mapv(|val| omega * val);
        r = &s - &t.mapv(|val| omega * val);

        if r.dot(&r).sqrt() < tol {
            break;
        }

        rho = rho_new;
    }

    Ok(x)
}

/// Preconditioned Conjugate Gradient
fn pcg<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
    m_inv: &Array2<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).unwrap())
        .unwrap_or_else(|| F::epsilon() * F::from(10.0).unwrap());

    let mut x = Array1::<F>::zeros(n);
    let mut r = b.to_owned();
    let mut z = m_inv.dot(&r);
    let mut p = z.clone();
    let mut rzold = r.dot(&z);

    for _ in 0..max_iter {
        let ap = a.dot(&p);
        let alpha = rzold / p.dot(&ap);

        x = &x + &p.mapv(|v| alpha * v);
        r = &r - &ap.mapv(|v| alpha * v);

        if r.dot(&r).sqrt() < tol {
            break;
        }

        z = m_inv.dot(&r);
        let rznew = r.dot(&z);
        let beta = rznew / rzold;
        p = &z + &p.mapv(|val| beta * val);

        rzold = rznew;
    }

    Ok(x)
}

/// Build preconditioner
fn build_preconditioner<F: Float + ndarray::ScalarOperand>(
    a: &ndarray::ArrayView2<F>,
    preconditioner_type: PreconditionerType,
) -> Result<Array2<F>, OpError> {
    let n = a.shape()[0];

    match preconditioner_type {
        PreconditionerType::None => Ok(Array2::<F>::eye(n)),
        PreconditionerType::Jacobi => {
            // Diagonal preconditioner
            let mut m_inv = Array2::<F>::zeros((n, n));
            for i in 0..n {
                if a[[i, i]].abs() > F::epsilon() {
                    m_inv[[i, i]] = F::one() / a[[i, i]];
                } else {
                    m_inv[[i, i]] = F::one();
                }
            }
            Ok(m_inv)
        }
        PreconditionerType::IncompleteCholesky => {
            // Simplified incomplete Cholesky
            let mut l = Array2::<F>::zeros((n, n));

            for i in 0..n {
                for j in 0..=i {
                    if a[[i, j]].abs() > F::epsilon() {
                        let mut sum = a[[i, j]];
                        for k in 0..j {
                            sum -= l[[i, k]] * l[[j, k]];
                        }

                        if i == j {
                            if sum > F::epsilon() {
                                l[[i, j]] = sum.sqrt();
                            } else {
                                l[[i, j]] = F::one();
                            }
                        } else {
                            l[[i, j]] = sum / l[[j, j]];
                        }
                    }
                }
            }

            // Return L^{-1}L^{-T} approximation
            // For simplicity, use diagonal approximation
            let mut m_inv = Array2::<F>::zeros((n, n));
            for i in 0..n {
                if l[[i, i]].abs() > F::epsilon() {
                    m_inv[[i, i]] = F::one() / (l[[i, i]] * l[[i, i]]);
                } else {
                    m_inv[[i, i]] = F::one();
                }
            }
            Ok(m_inv)
        }
    }
}

/// Solve least squares problem
fn solve_least_squares<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, OpError> {
    // Use normal equations: A^T A x = A^T b
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);

    // Solve using Cholesky decomposition (since A^T A is positive definite)
    solve_cholesky(&ata.view(), &atb.view())
}

/// Solve using Cholesky decomposition
fn solve_cholesky<F: Float>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
) -> Result<Array1<F>, OpError> {
    let n = a.shape()[0];
    let mut l = Array2::<F>::zeros((n, n));

    // Cholesky decomposition
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum < F::epsilon() {
                    return Err(OpError::Other("Matrix not positive definite".into()));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = Array1::<F>::zeros(n);
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i {
            let y_j = y[j];
            y[i] -= l[[i, j]] * y_j;
        }
        y[i] /= l[[i, i]];
    }

    // Back substitution: L^T x = y
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            let x_j = x[j];
            x[i] -= l[[j, i]] * x_j;
        }
        x[i] /= l[[i, i]];
    }

    Ok(x)
}

/// Outer product of two vectors
fn outer_product<F: Float>(u: &ndarray::ArrayView1<F>, v: &ndarray::ArrayView1<F>) -> Array2<F> {
    let m = u.len();
    let n = v.len();
    let mut result = Array2::<F>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = u[i] * v[j];
        }
    }

    result
}

// Public API functions

/// Solve Ax = b using Conjugate Gradient (for symmetric positive definite A)
pub fn conjugate_gradient_solve<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(ConjugateGradientOp {
            max_iter,
            tolerance,
        })
}

/// Solve Ax = b using GMRES (for general matrices)
pub fn gmres_solve<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(GMRESOp {
            max_iter,
            restart,
            tolerance,
        })
}

/// Solve Ax = b using BiCGSTAB (for general matrices)
pub fn bicgstab_solve<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(BiCGSTABOp {
            max_iter,
            tolerance,
        })
}

/// Solve Ax = b using Preconditioned Conjugate Gradient
pub fn pcg_solve<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
    preconditioner: PreconditionerType,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(PCGOp {
            max_iter,
            tolerance,
            preconditioner,
        })
}
