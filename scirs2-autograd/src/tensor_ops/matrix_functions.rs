use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

// ─────────────────────────────────────────────────────────────────────────────
// Public Op structs
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix square root operation (SPD-restricted).
///
/// # Restriction
///
/// This operation is restricted to **symmetric positive-definite (SPD)** input
/// matrices.  During the forward pass the matrix is verified to be symmetric
/// (`||A - Aᵀ||_F / ||A||_F < 1e-8`) and to have all strictly positive
/// eigenvalues (verified via the Jacobi eigenvalue algorithm).  Non-SPD inputs
/// return an `OpError`.
///
/// # Gradient
///
/// The reverse-mode gradient is derived from the Sylvester equation.  Given
/// `S = √A` and upstream cotangent `dS`, we solve
///
///   `S · X + X · S = dS`
///
/// for `X`, and the gradient w.r.t. `A` is `dA = X`.
///
/// This is well-posed because for an SPD matrix `S` all eigenvalues are
/// strictly positive, so the spectrum of `S` never intersects the negated
/// spectrum of `S`.
pub struct MatrixSqrtOp;

/// Matrix logarithm operation.
///
/// # Gradient
///
/// Uses the Daleckii-Krein spectral-expansion formula for symmetric inputs.
/// Given `A = V Λ Vᵀ` (Jacobi eigendecomposition), the reverse-mode gradient
/// with upstream cotangent `dB` is:
///
///   `dA = V · (Φ ⊙ (Vᵀ · dB · V)) · Vᵀ`
///
/// where the Loewner matrix `Φ_{ij} = (log λ_i − log λ_j) / (λ_i − λ_j)`
/// with the L'Hôpital limit `1/λ_i` when `i == j` (or `|λ_i - λ_j| < ε`).
pub struct MatrixLogOp;

/// Matrix power operation.
pub struct MatrixPowOp {
    pub power: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Op implementations
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSqrtOp {
    fn name(&self) -> &'static str {
        "MatrixSqrt"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix square root requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Verify SPD: symmetric check.
        let n = input_2d.shape()[0];
        let frob_sq_a: F = input_2d.iter().fold(F::zero(), |acc, &x| acc + x * x);
        let frob_a = frob_sq_a.sqrt();
        let mut sym_err = F::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = input_2d[[i, j]] - input_2d[[j, i]];
                sym_err += diff * diff;
            }
        }
        let sym_err = sym_err.sqrt();
        let sym_rel = if frob_a > F::epsilon() {
            sym_err / frob_a
        } else {
            sym_err
        };
        let sym_tol = F::from(1e-8).unwrap_or_else(F::epsilon);
        if sym_rel > sym_tol {
            return Err(OpError::Other(
                "sqrtm (SPD): matrix is not symmetric; use sqrtm_pd on SPD inputs only".into(),
            ));
        }

        // Verify SPD: smallest eigenvalue > 0 via Jacobi eigh.
        let (eigenvalues, _) = jacobi_eigh_f64_from::<F>(&input_2d);
        let min_ev = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_ev <= 0.0 {
            return Err(OpError::Other(format!(
                "sqrtm (SPD): matrix has non-positive eigenvalue {min_ev:.6e}; \
                 use sqrtm_pd on SPD inputs only"
            )));
        }

        // Compute √A via Jacobi eigh (correct for all n).
        let result = compute_matrix_sqrt_spd(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let g = ctx.graph();
        let output = ctx.output(); // S = √A
        let grad_output = ctx.output_grad(); // dS (upstream cotangent)

        let output_arr = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let grad_arr = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let s = match output_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let ds = match grad_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to f64 for numerical stability.
        let s_f64 = to_f64_mat(&s);
        let ds_f64 = to_f64_mat(&ds);

        // Solve S·X + X·S = dS  (Sylvester / Lyapunov equation).
        let da_f64 = match solve_sylvester_local(&s_f64, &s_f64, &ds_f64) {
            Ok(x) => x,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let da_f = from_f64_mat::<F>(&da_f64);
        let grad_tensor = crate::tensor_ops::convert_to_tensor(da_f.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixLogOp {
    fn name(&self) -> &'static str {
        "MatrixLog"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix logarithm requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute matrix logarithm
        let result = compute_matrix_log(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let g = ctx.graph();
        let input = ctx.input(0); // A
        let grad_output = ctx.output_grad(); // dB (upstream cotangent of log A)

        let input_arr = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let grad_arr = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let a = match input_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let db = match grad_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let a_f64 = to_f64_mat(&a);
        let db_f64 = to_f64_mat(&db);

        let da_f64 = logm_daleckii_krein(&a_f64, &db_f64);

        let da_f = from_f64_mat::<F>(&da_f64);
        let grad_tensor = crate::tensor_ops::convert_to_tensor(da_f.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixPowOp {
    fn name(&self) -> &'static str {
        "MatrixPow"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        // Required so gradient.rs can recover the exponent `p` for the backward
        // op via `downcast_ref::<MatrixPowOp>()`.
        Some(self)
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix power requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute matrix power
        let result = compute_matrix_pow(&input_2d, self.power)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Exact VJP of the symmetric matrix power via the spectral
        // (Daleckii-Krein) divided-difference formula:
        //   dA = V · (Φ ⊙ (Vᵀ·dB·V)) · Vᵀ,  Φ_{ij} = (λ_i^p − λ_j^p)/(λ_i − λ_j).
        //
        // NOTE: the live gradient path in gradient.rs uses MatrixPowBackwardOp
        // (dispatched by the "MatrixPow" op name); this method is kept correct
        // and consistent for any direct Op::grad invocation.
        let g = ctx.graph();
        let input = ctx.input(0); // A
        let grad_output = ctx.output_grad(); // dB

        let input_arr = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let grad_arr = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let a = match input_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };
        let db = match grad_arr.view().into_dimensionality::<Ix2>() {
            Ok(v) => v,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        if !is_symmetric_matrix(&a) {
            // Honest failure rather than a fabricated zero gradient.
            ctx.append_input_grad(0, None);
            return;
        }

        let a_f64 = to_f64_mat(&a);
        let db_f64 = to_f64_mat(&db);
        let da_f64 = powm_spectral_backward(&a_f64, &db_f64, self.power);
        let da = from_f64_mat::<F>(&da_f64);
        let grad_tensor = crate::tensor_ops::convert_to_tensor(da.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-conversion helpers (generic F ↔ f64)
// ─────────────────────────────────────────────────────────────────────────────

fn to_f64_mat<F: Float>(m: &scirs2_core::ndarray::ArrayView2<F>) -> Array2<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Array2::<f64>::zeros((r, c));
    for i in 0..r {
        for j in 0..c {
            out[[i, j]] = m[[i, j]].to_f64().unwrap_or(0.0);
        }
    }
    out
}

fn from_f64_mat<F: Float + FromPrimitive>(m: &Array2<f64>) -> Array2<F> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Array2::<F>::zeros((r, c));
    for i in 0..r {
        for j in 0..c {
            out[[i, j]] = F::from(m[[i, j]]).unwrap_or_else(F::zero);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi eigenvalue algorithm for symmetric real matrices (f64)
// ─────────────────────────────────────────────────────────────────────────────

/// Jacobi eigenvalue algorithm for symmetric `n × n` (f64).
///
/// Returns `(eigenvalues, V)` where columns of `V` are orthonormal eigenvectors
/// and `A = V · diag(λ) · Vᵀ`.
///
/// Uses cyclic Jacobi sweeps with Givens rotations until the sum of squared
/// off-diagonal entries is below `1e-28` (or 200 sweeps are exhausted).
fn jacobi_eigh(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut a = a.clone();
    let mut v = Array2::<f64>::eye(n);
    let max_sweeps = 200;

    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off < 1e-28 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[[p, p]] = app - t * apq;
                a[[q, q]] = aqq + t * apq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[[k, p]];
                        let akq = a[[k, q]];
                        a[[k, p]] = c * akp - s * akq;
                        a[[p, k]] = a[[k, p]];
                        a[[k, q]] = s * akp + c * akq;
                        a[[q, k]] = a[[k, q]];
                    }
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eigenvalues, v)
}

/// Convert generic matrix to f64, run Jacobi eigh, return (eigenvalues, V) in f64.
fn jacobi_eigh_f64_from<F: Float>(
    m: &scirs2_core::ndarray::ArrayView2<F>,
) -> (Vec<f64>, Array2<f64>) {
    let m_f64 = to_f64_mat(m);
    jacobi_eigh(&m_f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD matrix square root (via Jacobi eigh, correct for all n)
// ─────────────────────────────────────────────────────────────────────────────

fn compute_matrix_sqrt_spd<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let m_f64 = to_f64_mat(matrix);
    let (eigenvalues, v) = jacobi_eigh(&m_f64);

    // Compute V · diag(√λ) · Vᵀ.
    let mut vd = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let sq = eigenvalues[i].max(0.0).sqrt();
        for k in 0..n {
            vd[[k, i]] = v[[k, i]] * sq;
        }
    }
    // result = vd · Vᵀ
    let mut result_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..n {
                s += vd[[i, l]] * v[[j, l]];
            }
            result_f64[[i, j]] = s;
        }
    }
    Ok(from_f64_mat(&result_f64))
}

// ─────────────────────────────────────────────────────────────────────────────
// Local Sylvester solver: AX + XB = C (via Kronecker vectorisation)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve `A · X + X · B = C` for `X` using Kronecker-product vectorisation.
///
/// Forms the `mn × mn` linear system `(I_n ⊗ A + Bᵀ ⊗ I_m) vec(X) = vec(C)`
/// and solves via Gauss-Jordan elimination.
fn solve_sylvester_local(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
) -> Result<Array2<f64>, OpError> {
    let m = a.nrows();
    let n = b.nrows();
    debug_assert_eq!(a.ncols(), m, "A must be square");
    debug_assert_eq!(b.ncols(), n, "B must be square");
    debug_assert_eq!(c.nrows(), m);
    debug_assert_eq!(c.ncols(), n);

    let mn = m * n;
    let mut coeff = Array2::<f64>::zeros((mn, mn));

    // I_n ⊗ A: block-diagonal of A repeated n times.
    for col_block in 0..n {
        for i in 0..m {
            for j in 0..m {
                coeff[[col_block * m + i, col_block * m + j]] += a[[i, j]];
            }
        }
    }

    // Bᵀ ⊗ I_m: for each (rb, cb) in Bᵀ (= B[cb,rb]), add B[cb,rb] * I_m.
    for rb in 0..n {
        for cb in 0..n {
            let b_val = b[[cb, rb]];
            for d in 0..m {
                coeff[[rb * m + d, cb * m + d]] += b_val;
            }
        }
    }

    // vec(C): column-major stacking.
    let mut rhs = vec![0.0_f64; mn];
    for col in 0..n {
        for row in 0..m {
            rhs[col * m + row] = c[[row, col]];
        }
    }

    // Solve coeff · x = rhs via Gauss-Jordan.
    let x_vec = gauss_jordan_f64(&coeff, &rhs)?;

    // Reshape vec(X) back to m × n (column-major).
    let mut x = Array2::<f64>::zeros((m, n));
    for col in 0..n {
        for row in 0..m {
            x[[row, col]] = x_vec[col * m + row];
        }
    }
    Ok(x)
}

/// Gauss-Jordan elimination solving `A · x = b` for `x`.
fn gauss_jordan_f64(a: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>, OpError> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    // Build augmented matrix [A | b].
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot.
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-300 {
            return Err(OpError::Other(
                "Sylvester: singular coefficient matrix".into(),
            ));
        }

        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }
    }

    Ok((0..n).map(|i| aug[i][n]).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Daleckii-Krein logm backward (f64, symmetric case)
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix log backward via Daleckii-Krein spectral expansion.
///
/// For symmetric `A = V Λ Vᵀ` (all λ_i > 0):
///
///   `dA = V · (Φ ⊙ (Vᵀ · dB · V)) · Vᵀ`
///
/// where `Φ_{ij} = (log λ_i − log λ_j) / (λ_i − λ_j)`
/// with the L'Hôpital limit `1/λ_i` when `i == j` or `|λ_i - λ_j| < ε`.
///
/// Falls back to the symmetric formula even for mildly non-symmetric inputs
/// (gradient symmetrizes automatically for real functions of symmetric inputs).
fn logm_daleckii_krein(a: &Array2<f64>, db: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let (eigenvalues, v) = jacobi_eigh(a);
    let vt = transpose_f64(&v);

    // Y = Vᵀ · dB · V
    let vtdb = matmul_f64(&vt, db);
    let y = matmul_f64(&vtdb, &v);

    // Loewner matrix Φ.
    let mut phi = Array2::<f64>::zeros((n, n));
    let eps = 1e-12;
    for i in 0..n {
        for j in 0..n {
            let li = eigenvalues[i].max(f64::MIN_POSITIVE);
            let lj = eigenvalues[j].max(f64::MIN_POSITIVE);
            let diff = li - lj;
            if diff.abs() < eps * (li.abs() + lj.abs() + 1.0) {
                phi[[i, j]] = 1.0 / li;
            } else {
                phi[[i, j]] = (li.ln() - lj.ln()) / diff;
            }
        }
    }

    // Hadamard: Φ ⊙ Y.
    let mut y_phi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            y_phi[[i, j]] = phi[[i, j]] * y[[i, j]];
        }
    }

    // dA = V · (Φ ⊙ Y) · Vᵀ.
    let vy = matmul_f64(&v, &y_phi);
    matmul_f64(&vy, &vt)
}

fn matmul_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let p = b.ncols();
    debug_assert_eq!(k, b.nrows());
    let mut c = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

fn transpose_f64(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut t = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Spectral (Daleckii-Krein) backward for `f(A) = A^p` on a **symmetric** input.
///
/// For `A = V Λ Vᵀ` the reverse-mode gradient with upstream cotangent `dB` is
///
///   `dA = V · (Φ ⊙ (Vᵀ · dB · V)) · Vᵀ`
///
/// where the Loewner (divided-difference) matrix is
///
///   `Φ_{ij} = (λ_i^p − λ_j^p) / (λ_i − λ_j)`,  with the L'Hôpital limit
///   `Φ_{ii} = p · λ_i^{p−1}` when `λ_i ≈ λ_j`.
///
/// This is the exact VJP of the matrix power for symmetric matrices and
/// reduces to `p·A^{p-1}` in the scalar / commuting-perturbation limit.
fn powm_spectral_backward(a: &Array2<f64>, db: &Array2<f64>, power: f64) -> Array2<f64> {
    let n = a.nrows();
    let (eigenvalues, v) = jacobi_eigh(a);
    let vt = transpose_f64(&v);

    // Y = Vᵀ · dB · V
    let vtdb = matmul_f64(&vt, db);
    let y = matmul_f64(&vtdb, &v);

    // Loewner matrix of the divided differences of λ ↦ λ^p.
    let mut phi = Array2::<f64>::zeros((n, n));
    let eps = 1e-12;
    let pow_safe = |l: f64| -> f64 {
        if l.abs() > 1e-300 {
            l.powf(power)
        } else if power > 0.0 {
            0.0
        } else {
            // λ = 0 with non-positive power: undefined; clamp derivative to 0.
            0.0
        }
    };
    let dpow_safe = |l: f64| -> f64 {
        // d/dλ λ^p = p λ^{p-1}
        if l.abs() > 1e-300 {
            power * l.powf(power - 1.0)
        } else {
            0.0
        }
    };
    for i in 0..n {
        for j in 0..n {
            let li = eigenvalues[i];
            let lj = eigenvalues[j];
            let diff = li - lj;
            if diff.abs() < eps * (li.abs() + lj.abs() + 1.0) {
                phi[[i, j]] = dpow_safe(li);
            } else {
                phi[[i, j]] = (pow_safe(li) - pow_safe(lj)) / diff;
            }
        }
    }

    // Hadamard product then back-rotate: dA = V · (Φ ⊙ Y) · Vᵀ.
    let mut y_phi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            y_phi[[i, j]] = phi[[i, j]] * y[[i, j]];
        }
    }
    let vy = matmul_f64(&v, &y_phi);
    matmul_f64(&vy, &vt)
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward ops dispatched by gradient.rs::compute_grad_for_input.
//
// gradient.rs reaches these by op name ("MatrixSqrt" / "MatrixLog" / "MatrixPow").
// Each takes (original_input_A, upstream_gradient) and recomputes the forward
// internally so the exact VJP can be applied.  (The string-dispatch gradient
// path in gradient.rs does NOT invoke `Op::grad`, so the real backward MUST be
// delivered through these ops; otherwise the gradient would silently be zero.)
//
// These ops are bounded by `crate::Float` ONLY (not `FromPrimitive` /
// `ScalarOperand`) because gradient.rs::compute_grad_for_input is generic over
// `F: Float`.  All numerics run in f64 internally via NumCast (`to_f64` /
// `num_cast_from_f64`), which `num_traits::Float` provides.
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an `F` view to an owned `f64` matrix (NumCast-based; `Float`-only).
fn view_to_f64<F: Float>(m: &scirs2_core::ndarray::ArrayView2<F>) -> Array2<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Array2::<f64>::zeros((r, c));
    for i in 0..r {
        for j in 0..c {
            out[[i, j]] = m[[i, j]].to_f64().unwrap_or(0.0);
        }
    }
    out
}

/// Convert an `f64` matrix back to `F` (NumCast-based; `Float`-only).
fn f64_to_owned<F: Float>(m: &Array2<f64>) -> Array2<F> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Array2::<F>::zeros((r, c));
    for i in 0..r {
        for j in 0..c {
            out[[i, j]] = F::from(m[[i, j]]).unwrap_or_else(F::zero);
        }
    }
    out
}

/// Symmetry test on an `f64` matrix: `‖A − Aᵀ‖_F / ‖A‖_F < 1e-8`.
fn is_symmetric_f64(a: &Array2<f64>) -> bool {
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }
    let mut frob_sq = 0.0_f64;
    let mut asym_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            frob_sq += a[[i, j]] * a[[i, j]];
        }
        for j in (i + 1)..n {
            let d = a[[i, j]] - a[[j, i]];
            asym_sq += d * d;
        }
    }
    let frob = frob_sq.sqrt();
    let asym = (2.0 * asym_sq).sqrt();
    if frob > 1e-300 {
        asym / frob < 1e-8
    } else {
        asym < 1e-8
    }
}

/// `√A` for symmetric positive-definite `A` (f64), via Jacobi eigendecomposition.
/// Returns an error if `A` is not symmetric or has a non-positive eigenvalue.
fn sqrtm_spd_f64(a: &Array2<f64>) -> Result<Array2<f64>, OpError> {
    if !is_symmetric_f64(a) {
        return Err(OpError::Other(
            "sqrtm backward: input matrix is not symmetric".into(),
        ));
    }
    let n = a.nrows();
    let (eigenvalues, v) = jacobi_eigh(a);
    for &lambda in &eigenvalues {
        if lambda <= 0.0 {
            return Err(OpError::Other(format!(
                "sqrtm backward: non-positive eigenvalue {lambda:.6e}; input must be SPD"
            )));
        }
    }
    // √A = V · diag(√λ) · Vᵀ.
    let mut temp = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            temp[[i, j]] = v[[i, j]] * eigenvalues[j].sqrt();
        }
    }
    Ok(matmul_f64(&temp, &transpose_f64(&v)))
}

/// Backward op for `sqrtm` (SPD).  Solves the Sylvester equation
/// `S·X + X·S = dS` for `dA = X`, where `S = √A`.
pub(crate) struct MatrixSqrtBackwardOp;

impl<F: Float> Op<F> for MatrixSqrtBackwardOp {
    fn name(&self) -> &'static str {
        "MatrixSqrtBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let ds_input = ctx.input(1);

        let a = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixSqrtBackward: A must be 2D".into()))?;
        let ds = ds_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixSqrtBackward: dS must be 2D".into()))?;

        let a_f64 = view_to_f64(&a);
        let ds_f64 = view_to_f64(&ds);

        // Recompute S = √A (SPD path), then solve S·X + X·S = dS.
        let s_f64 = sqrtm_spd_f64(&a_f64)?;
        let da_f64 = solve_sylvester_local(&s_f64, &s_f64, &ds_f64)?;
        let da = f64_to_owned::<F>(&da_f64);
        ctx.append_output(da.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order gradient unsupported.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Backward op for `logm` via the Daleckii-Krein spectral formula.
pub(crate) struct MatrixLogBackwardOp;

impl<F: Float> Op<F> for MatrixLogBackwardOp {
    fn name(&self) -> &'static str {
        "MatrixLogBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let db_input = ctx.input(1);

        let a = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixLogBackward: A must be 2D".into()))?;
        let db = db_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixLogBackward: dB must be 2D".into()))?;

        let a_f64 = view_to_f64(&a);
        let db_f64 = view_to_f64(&db);

        if !is_symmetric_f64(&a_f64) {
            return Err(OpError::Other(
                "logm backward: spectral VJP requires a symmetric input; \
                 non-symmetric matrix-log gradient is not implemented"
                    .into(),
            ));
        }

        let da_f64 = logm_daleckii_krein(&a_f64, &db_f64);
        let da = f64_to_owned::<F>(&da_f64);
        ctx.append_output(da.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Backward op for `powm` (symmetric input) via the spectral divided-difference.
pub(crate) struct MatrixPowBackwardOp {
    pub(crate) power: f64,
}

impl<F: Float> Op<F> for MatrixPowBackwardOp {
    fn name(&self) -> &'static str {
        "MatrixPowBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let db_input = ctx.input(1);

        let a = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixPowBackward: A must be 2D".into()))?;
        let db = db_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("MatrixPowBackward: dB must be 2D".into()))?;

        let a_f64 = view_to_f64(&a);
        let db_f64 = view_to_f64(&db);

        if !is_symmetric_f64(&a_f64) {
            return Err(OpError::Other(
                "powm backward: spectral VJP requires a symmetric input; \
                 non-symmetric matrix-power gradient is not implemented"
                    .into(),
            ));
        }

        let da_f64 = powm_spectral_backward(&a_f64, &db_f64, self.power);
        let da = f64_to_owned::<F>(&da_f64);
        ctx.append_output(da.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions (keep existing fallback helpers for MatrixPowOp)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute matrix square root using eigendecomposition (symmetric PSD) or Denman-Beavers.
#[allow(dead_code)]
fn compute_matrix_sqrt<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    if is_symmetric_matrix(matrix) && is_positive_semidefinite(matrix)? {
        compute_matrix_sqrt_spd(matrix)
    } else {
        compute_matrix_sqrt_denman_beavers(matrix)
    }
}

/// Compute matrix logarithm using eigendecomposition
#[allow(dead_code)]
fn compute_matrix_log<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Check if matrix is symmetric
    if is_symmetric_matrix(matrix) {
        // Use Jacobi eigh for correct symmetric eigendecomposition.
        let (eigenvalues, v_f64) = jacobi_eigh_f64_from(matrix);

        // Check all eigenvalues are positive
        for &lambda in &eigenvalues {
            if lambda <= 0.0 {
                return Err(OpError::Other(
                    "Matrix has non-positive eigenvalues, cannot compute real logarithm".into(),
                ));
            }
        }

        // log(λ) for each eigenvalue.
        let log_ev: Vec<f64> = eigenvalues.iter().map(|&l| l.ln()).collect();

        // Reconstruct: log(A) = V · diag(log(λ)) · Vᵀ  (in f64).
        let mut temp_f64 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp_f64[[i, j]] = v_f64[[i, j]] * log_ev[j];
            }
        }
        let result_f64 = matmul_f64(&temp_f64, &transpose_f64(&v_f64));
        Ok(from_f64_mat(&result_f64))
    } else {
        // For general matrices, use inverse scaling and squaring method
        compute_matrix_log_inverse_scaling(matrix)
    }
}

/// Compute matrix power using eigendecomposition or repeated squaring
#[allow(dead_code)]
fn compute_matrix_pow<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    power: f64,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let p = F::from(power).ok_or(OpError::Other("Invalid power value".into()))?;

    // Special cases
    if power == 0.0 {
        return Ok(Array2::<F>::eye(n));
    } else if power == 1.0 {
        return Ok(matrix.to_owned());
    } else if power == -1.0 {
        return compute_matrix_inverse(matrix);
    }

    // For integer powers, use repeated squaring
    if power.fract() == 0.0 && power.abs() < 100.0 {
        let int_power = power as i32;
        return compute_matrix_pow_integer(matrix, int_power);
    }

    // For symmetric matrices, use eigendecomposition
    if is_symmetric_matrix(matrix) {
        let (eigenvalues, v_f64) = jacobi_eigh_f64_from(matrix);

        // Check for negative eigenvalues if power is not integer
        if power.fract() != 0.0 {
            for &lambda in &eigenvalues {
                if lambda < -1e-10 {
                    return Err(OpError::Other(
                        "Matrix has negative eigenvalues, cannot compute real fractional power"
                            .into(),
                    ));
                }
            }
        }

        // Compute power of eigenvalues.
        let pow_ev: Vec<f64> = eigenvalues
            .iter()
            .map(|&l| if l.abs() > 1e-300 { l.powf(power) } else { 0.0 })
            .collect();

        let mut temp_f64 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp_f64[[i, j]] = v_f64[[i, j]] * pow_ev[j];
            }
        }
        let result_f64 = matmul_f64(&temp_f64, &transpose_f64(&v_f64));
        Ok(from_f64_mat(&result_f64))
    } else {
        // For general matrices with fractional powers, use exp(p * log(A))
        let log_a = compute_matrix_log_inverse_scaling(matrix)?;
        let p_log_a = log_a.mapv(|x| x * p);
        compute_matrix_exp_pade(&p_log_a.view())
    }
}

/// Denman-Beavers iteration for matrix square root
#[allow(dead_code)]
fn compute_matrix_sqrt_denman_beavers<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut y = matrix.to_owned();
    let mut z = Array2::<F>::eye(n);

    let max_iter = 50;
    let tol = F::epsilon() * F::from(100.0).unwrap_or_else(|| F::one());

    for _ in 0..max_iter {
        let y_old = y.clone();

        // Compute inverses
        let y_inv = compute_matrix_inverse(&y.view())?;
        let z_inv = compute_matrix_inverse(&z.view())?;

        // Update Y and Z
        y = (&y + &z_inv) / F::from(2.0).unwrap_or_else(|| F::one());
        z = (&z + &y_inv) / F::from(2.0).unwrap_or_else(|| F::one());

        // Check convergence
        let diff = (&y - &y_old).mapv(|x| x.abs()).sum();
        if diff < tol {
            break;
        }
    }

    Ok(y)
}

/// Inverse scaling and squaring method for matrix logarithm
#[allow(dead_code)]
fn compute_matrix_log_inverse_scaling<
    F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive,
>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let i = Array2::<F>::eye(n);

    // Find s such that ||A^(1/2^s) - I|| < 0.5
    let mut s = 0;
    loop {
        let nn_f = F::from(n * n).unwrap_or_else(|| F::one());
        let norm = (&a - &i).mapv(|x| x.abs()).sum() / nn_f;
        if norm < F::from(0.5).unwrap_or_else(|| F::one()) {
            break;
        }
        // Take square root
        a = compute_matrix_sqrt_denman_beavers(&a.view())?;
        s += 1;
        if s > 20 {
            return Err(OpError::Other("Matrix logarithm failed to converge".into()));
        }
    }

    // Compute log using Padé approximation for log(I + X) where X = A - I
    let x = &a - &i;
    let mut log_a = compute_log_pade(&x)?;

    // Scale back
    let scale_factor = F::from(2.0_f64.powi(s)).unwrap_or_else(|| F::one());
    log_a *= scale_factor;

    Ok(log_a)
}

/// Padé approximation for log(I + X)
#[allow(dead_code)]
fn compute_log_pade<F: Float + scirs2_core::ndarray::ScalarOperand>(
    x: &Array2<F>,
) -> Result<Array2<F>, OpError> {
    let n = x.shape()[0];

    // Use Padé [3/3] approximation
    // log(I + X) ≈ X * (I + X/2 + X²/10) / (I + X/2 + 3X²/10)
    let x2 = x.dot(x);
    let half = F::from(0.5).unwrap_or_else(|| F::one());
    let tenth = F::from(0.1).unwrap_or_else(|| F::one());
    let three_tenths = F::from(0.3).unwrap_or_else(|| F::one());

    let i = Array2::<F>::eye(n);
    let numerator = &i + &(x * half) + &(&x2 * tenth);
    let denominator = &i + &(x * half) + &(&x2 * three_tenths);

    // Solve denominator * result = x * numerator
    let rhs = x.dot(&numerator);
    solve_matrix_equation(&denominator.view(), &rhs.view())
}

/// Integer matrix power using repeated squaring
#[allow(dead_code)]
fn compute_matrix_pow_integer<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    power: i32,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    if power == 0 {
        return Ok(Array2::<F>::eye(n));
    }

    let abs_power = power.unsigned_abs();
    let mut result = Array2::<F>::eye(n);
    let mut base = if power > 0 {
        matrix.to_owned()
    } else {
        compute_matrix_inverse(matrix)?
    };

    let mut p = abs_power;
    while p > 0 {
        if p & 1 == 1 {
            result = result.dot(&base);
        }
        base = base.dot(&base);
        p >>= 1;
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn is_symmetric_matrix<F: Float>(matrix: &scirs2_core::ndarray::ArrayView2<F>) -> bool {
    let n = matrix.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs()
                > F::epsilon() * F::from(10.0).unwrap_or_else(|| F::one())
            {
                return false;
            }
        }
    }
    true
}

#[allow(dead_code)]
fn is_positive_semidefinite<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<bool, OpError> {
    if !is_symmetric_matrix(matrix) {
        return Ok(false);
    }

    // Check eigenvalues via Jacobi eigh (correct for all n).
    let (eigenvalues, _) = jacobi_eigh_f64_from(matrix);
    for lambda in eigenvalues {
        if lambda < -1e-8 {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(dead_code)]
fn compute_matrix_inverse<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut inv = Array2::<F>::eye(n);

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < F::epsilon() {
            return Err(OpError::IncompatibleShape("Matrix is singular".into()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                a.swap((i, j), (max_row, j));
                inv.swap((i, j), (max_row, j));
            }
        }

        // Scale pivot row
        let pivot = a[[i, i]];
        for j in 0..n {
            a[[i, j]] /= pivot;
            inv[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = a[[k, i]];
                for j in 0..n {
                    let a_ij = a[[i, j]];
                    let inv_ij = inv[[i, j]];
                    a[[k, j]] -= factor * a_ij;
                    inv[[k, j]] -= factor * inv_ij;
                }
            }
        }
    }

    Ok(inv)
}

#[allow(dead_code)]
fn solve_matrix_equation<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // Solve AX = B using LU decomposition or direct inversion
    let a_inv = compute_matrix_inverse(a)?;
    Ok(a_inv.dot(b))
}

#[allow(dead_code)]
fn compute_matrix_exp_pade<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Compute norm of matrix
    let mut norm = F::zero();
    for i in 0..n {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum += matrix[[i, j]].abs();
        }
        if row_sum > norm {
            norm = row_sum;
        }
    }

    // Scaling parameter
    let s = if norm > F::one() {
        (norm.ln() / F::from(2.0).unwrap_or_else(|| F::one()).ln()).ceil()
    } else {
        F::zero()
    };

    let scale = F::from(2.0).unwrap_or_else(|| F::one()).powf(s);
    let scaled_matrix = matrix.mapv(|x| x / scale);

    // Padé approximation coefficients (order 6)
    let c0 = F::from(1.0).unwrap_or_else(|| F::one());
    let c1 = F::from(0.5).unwrap_or_else(|| F::one());
    let c2 = F::from(12.0).unwrap_or_else(|| F::one()).recip();
    let c3 = F::from(120.0).unwrap_or_else(|| F::one()).recip();
    let c4 = F::from(3360.0).unwrap_or_else(|| F::one()).recip();
    let c5 = F::from(30240.0).unwrap_or_else(|| F::one()).recip();
    let c6 = F::from(1209600.0).unwrap_or_else(|| F::one()).recip();

    // Compute powers of matrix
    let i = Array2::<F>::eye(n);
    let a2 = scaled_matrix.dot(&scaled_matrix);
    let a4 = a2.dot(&a2);
    let a6 = a4.dot(&a2);

    // Compute U and V for Padé approximation
    let u = &scaled_matrix * c1 + &a2 * c3 + &a4 * c5;
    let u = scaled_matrix.dot(&u);

    let v = &i * c0 + &a2 * c2 + &a4 * c4 + &a6 * c6;

    // Solve (V - U) * R = (V + U)
    let v_minus_u = &v - &u;
    let v_plus_u = &v + &u;

    // Use Gaussian elimination to solve
    let mut result = solve_matrix_equation(&v_minus_u.view(), &v_plus_u.view())?;

    // Square the result s times
    for _ in 0..s.to_usize().unwrap_or(0) {
        result = result.dot(&result);
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute matrix square root (SPD-restricted).
///
/// Applies to **symmetric positive-definite** inputs only.  Errors on non-SPD
/// matrices — see `MatrixSqrtOp` for the restriction rationale.
///
/// Alias: `sqrtm_pd`.
#[allow(dead_code)]
pub fn matrix_sqrt<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixSqrtOp)
}

/// Compute matrix square root (SPD-restricted) — explicit alias of [`matrix_sqrt`].
///
/// Errors if the input is not symmetric positive-definite.  The backward pass
/// solves the Sylvester equation `S · X + X · S = dS` for `dA = X`.
#[allow(dead_code)]
pub fn sqrtm_pd<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    matrix_sqrt(matrix)
}

/// Compute matrix logarithm.
///
/// Uses the Daleckii-Krein spectral backward for symmetric positive-definite
/// inputs.  For non-symmetric or nearly-indefinite inputs falls back to
/// inverse-scaling-and-squaring.
#[allow(dead_code)]
pub fn matrix_log<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixLogOp)
}

/// Compute matrix power
#[allow(dead_code)]
pub fn matrix_power<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    power: f64,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixPowOp { power })
}

// Aliases
pub use self::matrix_log as logm;
pub use self::matrix_power as powm;
pub use self::matrix_sqrt as sqrtm;

#[cfg(test)]
mod grad_tests {
    //! End-to-end finite-difference gradient checks for the matrix functions.
    //!
    //! These verify the *live* gradient path (gradient.rs string dispatch →
    //! MatrixSqrt/Log/Pow backward ops), guarding against the prior silent-zero
    //! regression.
    use crate::tensor_ops as T;
    use scirs2_core::ndarray::{array, Array2};

    /// A fixed 3×3 symmetric positive-definite matrix with distinct eigenvalues.
    fn spd_3x3() -> Array2<f64> {
        array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]
    }

    /// Symmetric central finite difference of `f(A) = sum_all(op(A))`.
    ///
    /// The matrix functions here are defined on **symmetric** inputs, so a valid
    /// directional derivative must keep `A` symmetric: off-diagonal entries are
    /// perturbed in the symmetric pair `(i,j)` & `(j,i)` together.  The resulting
    /// gradient is the symmetric gradient, which is what the spectral backward
    /// (Daleckii-Krein / Sylvester) produces.
    fn fd_loss<FwOp>(a: &Array2<f64>, forward: FwOp) -> Array2<f64>
    where
        FwOp: Fn(&Array2<f64>) -> f64,
    {
        let n = a.nrows();
        assert_eq!(n, a.ncols());
        let h = 1e-6_f64;
        let mut grad = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut ap = a.clone();
                let mut am = a.clone();
                if i == j {
                    ap[[i, i]] += h;
                    am[[i, i]] -= h;
                    grad[[i, i]] = (forward(&ap) - forward(&am)) / (2.0 * h);
                } else {
                    ap[[i, j]] += h;
                    ap[[j, i]] += h;
                    am[[i, j]] -= h;
                    am[[j, i]] -= h;
                    let d = (forward(&ap) - forward(&am)) / (2.0 * h);
                    // Split across the symmetric pair (matches symmetric grad).
                    grad[[i, j]] = 0.5 * d;
                    grad[[j, i]] = 0.5 * d;
                }
            }
        }
        grad
    }

    /// Symmetrise a gradient matrix: `(G + Gᵀ)/2`.  The analytic backward may
    /// return an unsymmetrised gradient; comparing the symmetric parts is the
    /// meaningful check for functions restricted to symmetric inputs.
    fn symmetrize(g: &Array2<f64>) -> Array2<f64> {
        let n = g.nrows();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                out[[i, j]] = 0.5 * (g[[i, j]] + g[[j, i]]);
            }
        }
        out
    }

    /// Forward `sum_all(op(A))` evaluated in the autograd graph.
    fn fwd_sum<BuildOp>(a: &Array2<f64>, build: BuildOp) -> f64
    where
        BuildOp: for<'g> Fn(&crate::tensor::Tensor<'g, f64>) -> crate::tensor::Tensor<'g, f64>,
    {
        crate::run(|g| {
            let av = T::variable(a.clone(), g);
            let y = build(&av);
            let loss = T::sum_all(y);
            loss.eval(g).expect("forward eval").iter().copied().sum()
        })
    }

    /// Analytic gradient `d sum_all(op(A)) / dA` via the autograd graph.
    fn analytic_grad<BuildOp>(a: &Array2<f64>, build: BuildOp) -> Array2<f64>
    where
        BuildOp: for<'g> Fn(&crate::tensor::Tensor<'g, f64>) -> crate::tensor::Tensor<'g, f64>,
    {
        crate::run(|g| {
            let av = T::variable(a.clone(), g);
            let y = build(&av);
            let loss = T::sum_all(y);
            let grads = T::grad(&[&loss], &[&av]);
            let ga = grads[0].eval(g).expect("grad eval");
            ga.into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("grad 2D")
                .to_owned()
        })
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| (x - y).abs().max(m))
    }

    #[test]
    fn matrix_sqrt_gradient_matches_fd() {
        let a = spd_3x3();
        let analytic = symmetrize(&analytic_grad(&a, T::matrix_sqrt));
        let numeric = fd_loss(&a, |ap| fwd_sum(ap, T::matrix_sqrt));
        // Gradient must NOT be all-zero (the regression we are fixing).
        let max_g = analytic.iter().fold(0.0_f64, |m, &x| x.abs().max(m));
        assert!(max_g > 1e-6, "sqrt gradient is all-zero (regression!)");
        let err = max_abs_diff(&analytic, &numeric);
        assert!(err < 1e-4, "matrix_sqrt grad fd mismatch: err = {err}");
    }

    #[test]
    fn matrix_log_gradient_matches_fd() {
        let a = spd_3x3();
        let analytic = symmetrize(&analytic_grad(&a, T::matrix_log));
        let numeric = fd_loss(&a, |ap| fwd_sum(ap, T::matrix_log));
        let max_g = analytic.iter().fold(0.0_f64, |m, &x| x.abs().max(m));
        assert!(max_g > 1e-6, "log gradient is all-zero (regression!)");
        let err = max_abs_diff(&analytic, &numeric);
        assert!(err < 1e-4, "matrix_log grad fd mismatch: err = {err}");
    }

    #[test]
    fn matrix_power_gradient_matches_fd() {
        let a = spd_3x3();
        // Use a fractional power so the spectral divided-difference path runs.
        let p = 1.5_f64;
        let analytic = symmetrize(&analytic_grad(&a, move |t| T::matrix_power(t, p)));
        let numeric = fd_loss(&a, |ap| fwd_sum(ap, move |t| T::matrix_power(t, p)));
        let max_g = analytic.iter().fold(0.0_f64, |m, &x| x.abs().max(m));
        assert!(max_g > 1e-6, "pow gradient is all-zero (regression!)");
        let err = max_abs_diff(&analytic, &numeric);
        assert!(err < 1e-3, "matrix_power grad fd mismatch: err = {err}");
    }
}
