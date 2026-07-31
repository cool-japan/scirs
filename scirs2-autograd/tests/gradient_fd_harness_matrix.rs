//! Finite-difference gradient harness for the **matrix-valued** operations.
//!
//! Companion to `gradient_fd_harness.rs`, which covers the element-wise and reduction
//! ops. Everything here differentiates a function of a whole matrix — matrix
//! exponential/logarithm/power, the matrix trigonometric functions, iterative linear
//! solvers, einsum, and the control-flow ops — where the wrong answer is not "off by a
//! constant" but "a completely different linear operator".
//!
//! The same two rules as the element-wise harness are load-bearing:
//!
//! 1. **The cotangent is NON-UNIFORM.** With an all-ones cotangent, `sum(f(A))` is
//!    symmetric in ways that hide a transposed or element-wise-instead-of-Fréchet VJP.
//! 2. **Inputs are built with `T::variable`.** `T::convert_to_tensor` marks the node
//!    non-differentiable, so a broken VJP would silently read as a zero gradient.
//!
//! Every check below fails on the code that was in the tree before this harness existed:
//! the matrix functions returned the cotangent unchanged (or multiplied it element-wise
//! by `cos(A)`), the iterative solvers returned it to *both* `A` and `b`, `einsum`
//! returned a mis-shaped copy that made the backward pass panic, `reduce_variance`
//! dropped the `2(x - mean)/N` factor, `conditional` fed both branches, and
//! `custom_unary_op` ignored the user's backward closure.

use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::{ArrayD, IxDyn};

type Ctx<'g> = ag::Context<'g, f64>;
type Tsr<'g> = ag::Tensor<'g, f64>;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Buf {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Buf {
    fn new(shape: &[usize], data: &[f64]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Buf: shape {shape:?} does not match {} values",
            data.len()
        );
        Buf {
            data: data.to_vec(),
            shape: shape.to_vec(),
        }
    }

    fn to_array(&self) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&self.shape), self.data.clone())
            .expect("Buf: shape/data mismatch")
    }
}

/// Deterministic, strictly non-uniform ramp.
fn ramp(shape: &[usize], start: f64, step: f64) -> Buf {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();
    Buf::new(shape, &data)
}

fn forward_loss<B>(xs: &[Buf], cotangent: &Buf, build: &B) -> f64
where
    B: for<'g> Fn(&[Tsr<'g>], &'g Ctx<'g>) -> Tsr<'g>,
{
    ag::run(|g| {
        let g: &Ctx = g;
        let ts: Vec<Tsr> = xs
            .iter()
            .map(|b| T::convert_to_tensor(b.to_array(), g))
            .collect();
        let y = build(&ts, g);
        let cot = T::convert_to_tensor(cotangent.to_array(), g);
        let loss = T::sum_all(T::mul(y, cot));
        let arr = loss.eval(g).expect("forward loss eval failed");
        arr.iter().copied().sum::<f64>()
    })
}

fn analytic_grads<B>(name: &str, xs: &[Buf], cotangent: &Buf, build: &B) -> Vec<ArrayD<f64>>
where
    B: for<'g> Fn(&[Tsr<'g>], &'g Ctx<'g>) -> Tsr<'g>,
{
    ag::run(|g| {
        let g: &Ctx = g;
        let ts: Vec<Tsr> = xs.iter().map(|b| T::variable(b.to_array(), g)).collect();
        let y = build(&ts, g);

        let y_arr = y.eval(g).expect("forward eval failed");
        assert_eq!(
            y_arr.shape(),
            cotangent.shape.as_slice(),
            "{name}: cotangent shape {:?} does not match output shape {:?}",
            cotangent.shape,
            y_arr.shape()
        );

        let cot = T::convert_to_tensor(cotangent.to_array(), g);
        let loss = T::sum_all(T::mul(y, cot));
        let grads = T::grad(&[loss], &ts);
        grads
            .iter()
            .map(|gt| gt.eval(g).expect("analytic gradient eval failed"))
            .collect()
    })
}

/// Central-difference check of every input of `build`.
fn check_grads<B>(name: &str, xs: &[Buf], cotangent: &Buf, build: B, tol: f64)
where
    B: for<'g> Fn(&[Tsr<'g>], &'g Ctx<'g>) -> Tsr<'g>,
{
    let analytic = analytic_grads(name, xs, cotangent, &build);
    assert_eq!(
        analytic.len(),
        xs.len(),
        "{name}: expected one gradient per input"
    );

    for (k, xk) in xs.iter().enumerate() {
        assert_eq!(
            analytic[k].shape(),
            xk.shape.as_slice(),
            "{name}: gradient {k} has shape {:?}, input has shape {:?}",
            analytic[k].shape(),
            xk.shape
        );
        let analytic_flat: Vec<f64> = analytic[k].iter().copied().collect();

        for (i, &got) in analytic_flat.iter().enumerate() {
            let h = 1e-5 * (1.0 + xk.data[i].abs());

            let mut plus = xs.to_vec();
            plus[k].data[i] += h;
            let mut minus = xs.to_vec();
            minus[k].data[i] -= h;

            let f_plus = forward_loss(&plus, cotangent, &build);
            let f_minus = forward_loss(&minus, cotangent, &build);
            let numeric = (f_plus - f_minus) / (2.0 * h);

            assert!(
                got.is_finite(),
                "{name}: analytic gradient[{k}][{i}] is not finite ({got})"
            );
            assert!(
                (got - numeric).abs() <= tol * (1.0 + numeric.abs()),
                "{name}: d/dx[{k}][{i}] analytic={got} finite-difference={numeric} \
                 (tol={tol}, x={:?})",
                xk.data
            );
        }
    }
}

fn check_matrix_fn<B>(name: &str, a: Buf, build: B, tol: f64)
where
    B: for<'g> Fn(Tsr<'g>) -> Tsr<'g>,
{
    let cot = ramp(&a.shape, 0.37, -0.19);
    check_grads(name, &[a], &cot, move |ts, _g| build(ts[0]), tol);
}

const LOOSE: f64 = 1e-4;

/// A modest-norm, non-symmetric, non-diagonal 3x3 matrix.
///
/// The norm is kept small so that `logm` stays on the principal branch and the finite
/// differences are well conditioned; the asymmetry is what distinguishes a genuine
/// Fréchet derivative from an element-wise or transposed rule.
fn general_3x3() -> Buf {
    Buf::new(
        &[3, 3],
        &[0.30, 0.24, -0.17, 0.11, -0.28, 0.19, -0.21, 0.13, 0.35],
    )
}

/// A symmetric positive definite 3x3 matrix (diagonally dominant).
fn spd_3x3() -> Buf {
    Buf::new(
        &[3, 3],
        &[2.40, 0.35, -0.20, 0.35, 1.90, 0.28, -0.20, 0.28, 1.55],
    )
}

// ---------------------------------------------------------------------------
// 1. Matrix exponential family
// ---------------------------------------------------------------------------

#[test]
fn fd_matrix_exponential_variants() {
    // All three entry points compute exp(A) by different forward algorithms and must
    // share the same Fréchet-derivative VJP.
    check_matrix_fn("matrix_exp", general_3x3(), |t| T::matrix_exp(&t), LOOSE);
    check_matrix_fn("expm2", general_3x3(), |t| T::expm2(&t), LOOSE);
    check_matrix_fn("expm3", general_3x3(), |t| T::expm3(&t), LOOSE);
}

/// `expm3` takes the symmetric branch (real eigendecomposition), which used to return the
/// diagonal of the matrix as its "eigenvalues" and the identity as its "eigenvectors" for
/// every matrix larger than 2x2.
#[test]
fn fd_matrix_exponential_symmetric_branch() {
    let b = ramp(&[3, 3], 0.13, 0.07);
    let cot = ramp(&[3, 3], 0.41, -0.23);
    check_grads(
        "expm3_symmetric",
        &[b],
        &cot,
        |ts, _g| {
            // A = B + Bᵀ is exactly symmetric even in floating point, so perturbing B
            // keeps the argument inside the symmetric branch.
            let sym = T::add(ts[0], T::transpose(ts[0], &[1, 0]));
            T::expm3(&sym)
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 2. Matrix trigonometric / hyperbolic / sign functions
// ---------------------------------------------------------------------------

#[test]
fn fd_matrix_trigonometric_functions() {
    // Before the fix these used `cos(A) * gy` **element-wise**, i.e. the chain rule for
    // `mapv(sin)` rather than the Fréchet derivative of the matrix sine.
    check_matrix_fn("sinm", general_3x3(), |t| T::sinm(&t), LOOSE);
    check_matrix_fn("cosm", general_3x3(), |t| T::cosm(&t), LOOSE);
    check_matrix_fn("sinhm", general_3x3(), |t| T::sinhm(&t), LOOSE);
    check_matrix_fn("coshm", general_3x3(), |t| T::coshm(&t), LOOSE);
}

/// The matrix sign function is *not* a constant function of `A`: its gradient is the
/// solution of `S L + L S = E - S E S` and is generically non-zero off the diagonal.
#[test]
fn fd_matrix_sign_function() {
    // Eigenvalues well away from the imaginary axis so `sign` is locally smooth.
    let a = Buf::new(
        &[3, 3],
        &[1.60, 0.30, -0.20, 0.25, 2.10, 0.18, -0.15, 0.22, 1.30],
    );
    check_matrix_fn("signm", a, |t| T::signm(&t), 1e-3);
}

/// `funm` applies a scalar function through the spectrum; its VJP is the
/// Daleckii-Krein divided-difference formula.
#[test]
fn fd_matrix_function_through_the_spectrum() {
    fn cube(x: f64) -> f64 {
        x * x * x
    }
    let b = ramp(&[3, 3], 0.21, 0.09);
    let cot = ramp(&[3, 3], 0.33, -0.17);
    check_grads(
        "funm_cube",
        &[b],
        &cot,
        |ts, _g| {
            let sym = T::add(ts[0], T::transpose(ts[0], &[1, 0]));
            T::funm(&sym, cube, "cube")
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 3. Pseudo-inverse
// ---------------------------------------------------------------------------

/// For a square invertible matrix the Moore-Penrose pseudo-inverse equals the inverse, so
/// its VJP must equal the `matrix_inverse` rule `-A^-ᵀ G A^-ᵀ`.
#[test]
fn fd_pseudo_inverse_square() {
    check_matrix_fn("pinv", spd_3x3(), |t| T::pinv(&t), LOOSE);
}

// ---------------------------------------------------------------------------
// 4. Iterative linear solvers
// ---------------------------------------------------------------------------

fn solver_system() -> (Buf, Buf, Buf) {
    let a = spd_3x3();
    let b = Buf::new(&[3], &[0.70, -0.45, 1.10]);
    let cot = Buf::new(&[3], &[0.29, -0.83, 0.51]);
    (a, b, cot)
}

#[test]
fn fd_conjugate_gradient_solver() {
    let (a, b, cot) = solver_system();
    check_grads(
        "conjugate_gradient_solve",
        &[a, b],
        &cot,
        |ts, _g| T::conjugate_gradient_solve(&ts[0], &ts[1], 200, Some(1e-15)),
        LOOSE,
    );
}

#[test]
fn fd_gmres_solver() {
    let (a, b, cot) = solver_system();
    check_grads(
        "gmres_solve",
        &[a, b],
        &cot,
        |ts, _g| T::gmres_solve(&ts[0], &ts[1], 200, 3, Some(1e-15)),
        LOOSE,
    );
}

#[test]
fn fd_bicgstab_solver() {
    let (a, b, cot) = solver_system();
    check_grads(
        "bicgstab_solve",
        &[a, b],
        &cot,
        |ts, _g| T::bicgstab_solve(&ts[0], &ts[1], 200, Some(1e-15)),
        LOOSE,
    );
}

#[test]
fn fd_pcg_solver() {
    let (a, b, cot) = solver_system();
    check_grads(
        "pcg_solve",
        &[a, b],
        &cot,
        |ts, _g| {
            T::pcg_solve(
                &ts[0],
                &ts[1],
                200,
                Some(1e-15),
                T::PreconditionerType::Jacobi,
            )
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 5. Einsum
// ---------------------------------------------------------------------------

#[test]
fn fd_einsum_matmul() {
    let a = ramp(&[2, 3], 0.4, 0.13);
    let b = ramp(&[3, 2], -0.6, 0.21);
    let cot = ramp(&[2, 2], 0.31, -0.17);
    check_grads(
        "einsum_ij_jk_ik",
        &[a, b],
        &cot,
        |ts, _g| T::einsum("ij,jk->ik", &[&ts[0], &ts[1]]),
        LOOSE,
    );
}

#[test]
fn fd_einsum_contraction_and_broadcast() {
    // `ij,j->i`: the gradient w.r.t. the vector sums over `i`, and the gradient w.r.t.
    // the matrix is an outer product — neither is the raw cotangent.
    let a = ramp(&[3, 4], 0.25, 0.11);
    let v = ramp(&[4], -0.7, 0.23);
    let cot = ramp(&[3], 0.37, -0.29);
    check_grads(
        "einsum_ij_j_i",
        &[a, v],
        &cot,
        |ts, _g| T::einsum("ij,j->i", &[&ts[0], &ts[1]]),
        LOOSE,
    );
}

#[test]
fn fd_einsum_full_contraction_to_scalar() {
    let a = ramp(&[3, 3], 0.31, 0.07);
    let b = ramp(&[3, 3], -0.44, 0.19);
    let cot = Buf::new(&[], &[0.73]);
    check_grads(
        "einsum_ij_ij_scalar",
        &[a, b],
        &cot,
        |ts, _g| T::einsum("ij,ij->", &[&ts[0], &ts[1]]),
        LOOSE,
    );
}

#[test]
fn fd_einsum_outer_product() {
    // Every label of each operand survives into the output, so the gradient of one
    // operand contracts the cotangent against the *other* operand.
    let u = ramp(&[3], 0.4, 0.17);
    let v = ramp(&[2], -0.5, 0.31);
    let cot = ramp(&[3, 2], 0.23, -0.13);
    check_grads(
        "einsum_i_j_ij",
        &[u, v],
        &cot,
        |ts, _g| T::einsum("i,j->ij", &[&ts[0], &ts[1]]),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 6. Variance reduction
// ---------------------------------------------------------------------------

#[test]
fn fd_reduce_variance_along_axis() {
    let x = ramp(&[3, 4], 0.35, 0.27);
    let cot = ramp(&[4], 0.41, -0.19);
    check_grads(
        "reduce_variance_axis0",
        &[x],
        &cot,
        |ts, _g| T::reduce_variance(ts[0], &[0], false),
        LOOSE,
    );
}

#[test]
fn fd_reduce_variance_keep_dims_and_all_axes() {
    let x = ramp(&[2, 3], -0.6, 0.31);
    let cot = ramp(&[1, 3], 0.23, 0.44);
    check_grads(
        "reduce_variance_keepdims",
        std::slice::from_ref(&x),
        &cot,
        |ts, _g| T::reduce_variance(ts[0], &[0], true),
        LOOSE,
    );

    let cot_all = Buf::new(&[], &[0.61]);
    check_grads(
        "reduce_variance_all_axes",
        &[x],
        &cot_all,
        |ts, _g| T::reduce_variance(ts[0], &[0, 1], false),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 7. Control flow: conditional and checkpoint
// ---------------------------------------------------------------------------

/// The taken branch must receive the whole cotangent and the other branch exactly zero.
///
/// Both branches are non-linear functions of *different* variables, so a rule that feeds
/// both (the previous behaviour) shows up as a non-zero gradient on the untaken input.
#[test]
fn fd_conditional_routes_to_the_taken_branch_only() {
    let x = ramp(&[4], 0.4, 0.21);
    let y = ramp(&[4], -0.7, 0.33);
    let cot = ramp(&[4], 0.29, -0.17);

    // Predicate true: the `true` branch (square(x)) is selected.
    check_grads(
        "conditional_true_branch",
        &[x.clone(), y.clone()],
        &cot,
        |ts, g| {
            let cond = T::convert_to_tensor(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).expect("cond"),
                g,
            );
            T::conditional(
                &cond,
                &T::square(ts[0]),
                &T::square(ts[1]),
                T::PredicateType::GreaterThanZero,
            )
        },
        LOOSE,
    );

    // Predicate false: the `false` branch is selected and `x` must get exactly zero.
    check_grads(
        "conditional_false_branch",
        &[x, y],
        &cot,
        |ts, g| {
            let cond = T::convert_to_tensor(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).expect("cond"),
                g,
            );
            T::conditional(
                &cond,
                &T::square(ts[0]),
                &T::square(ts[1]),
                T::PredicateType::GreaterThanZero,
            )
        },
        LOOSE,
    );
}

#[test]
fn fd_smart_checkpoint_is_transparent() {
    let x = ramp(&[5], 0.3, 0.19);
    let cot = ramp(&[5], 0.41, -0.23);
    check_grads(
        "smart_checkpoint",
        &[x],
        &cot,
        |ts, _g| {
            // The checkpoint boundary must not change the gradient of the wrapped
            // computation.
            let inner = T::sin(T::square(ts[0]));
            T::exp(T::smart_checkpoint(&inner, 1024))
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 8. Custom gradient closures
// ---------------------------------------------------------------------------

/// `custom_unary_op` must actually call the backward closure the caller registered.
#[test]
fn fd_custom_unary_op_uses_its_backward_closure() {
    let x = ramp(&[4], 0.5, 0.23);
    let cot = ramp(&[4], 0.31, -0.19);
    check_grads(
        "custom_unary_op_cube",
        &[x],
        &cot,
        |ts, g| {
            // Forward: x^3.  Backward: 3 x^2 * gy, registered as a closure.
            ag::custom_unary_op(
                "cube",
                |x: &scirs2_core::ndarray::ArrayViewD<f64>| x.mapv(|v| v * v * v),
                |gy: &Tsr, x: &Tsr, _y: &Tsr| Some(T::mul(*gy, T::scalar_mul(T::square(*x), 3.0))),
                ts[0],
                g,
            )
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 9. Tensor solve
// ---------------------------------------------------------------------------

#[test]
fn fd_tensor_solve_square() {
    let a = spd_3x3();
    let b = Buf::new(&[3], &[0.70, -0.45, 1.10]);
    let cot = Buf::new(&[3], &[0.29, -0.83, 0.51]);
    check_grads(
        "tensor_solve",
        &[a, b],
        &cot,
        |ts, _g| T::tensor_solve(&ts[0], &ts[1], None),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 10. Decompositions that used to be unevaluable
// ---------------------------------------------------------------------------

/// `svd_jacobi`, `randomized_svd`, `generalized_eigen` and `qr_pivot` each returned
/// tensors whose op unconditionally answered "should be handled by parent op": the whole
/// public API could not be evaluated at all.
#[test]
fn decompositions_are_evaluable() {
    ag::run(|g| {
        let g: &Ctx = g;
        let m = T::convert_to_tensor(spd_3x3().to_array(), g);

        let (u, s, vt) = T::svd_jacobi(&m, false);
        let u_a = u.eval(g).expect("svd_jacobi U must evaluate");
        let s_a = s.eval(g).expect("svd_jacobi S must evaluate");
        let vt_a = vt.eval(g).expect("svd_jacobi Vt must evaluate");
        assert_eq!(s_a.len(), 3);
        // Singular values are descending and positive.
        assert!(
            s_a[0] >= s_a[1] && s_a[1] >= s_a[2] && s_a[2] > 0.0,
            "{s_a:?}"
        );
        // U diag(s) Vt reconstructs the matrix.
        let mut recon = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += u_a[[i, k]] * s_a[k] * vt_a[[k, j]];
                }
                recon[i * 3 + j] = acc;
            }
        }
        for (got, want) in recon.iter().zip(spd_3x3().data.iter()) {
            assert!(
                (got - want).abs() < 1e-8,
                "SVD reconstruction {got} vs {want}"
            );
        }

        let (q, r, p) = T::qr_pivot(&m);
        q.eval(g).expect("qr_pivot Q must evaluate");
        r.eval(g).expect("qr_pivot R must evaluate");
        p.eval(g).expect("qr_pivot P must evaluate");

        let (ru, rs, rvt) = T::randomized_svd(&m, 3, 2, 2);
        ru.eval(g).expect("randomized_svd U must evaluate");
        rs.eval(g).expect("randomized_svd S must evaluate");
        rvt.eval(g).expect("randomized_svd Vt must evaluate");

        let identity = T::convert_to_tensor(
            ArrayD::from_shape_vec(
                IxDyn(&[3, 3]),
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            )
            .expect("identity"),
            g,
        );
        let (vals, vecs) = T::generalized_eigen(&m, &identity);
        let vals_a = vals
            .eval(g)
            .expect("generalized_eigen values must evaluate");
        vecs.eval(g)
            .expect("generalized_eigen vectors must evaluate");
        // With B = I the generalized problem is the ordinary symmetric one, so the trace
        // of the spectrum must equal the trace of the matrix.
        let trace: f64 = spd_3x3().data[0] + spd_3x3().data[4] + spd_3x3().data[8];
        assert!((vals_a.iter().sum::<f64>() - trace).abs() < 1e-8);
    });
}

/// Reduced-SVD singular values have an exact VJP.
#[test]
fn fd_svd_jacobi_singular_values() {
    // Well-separated singular values: the SVD VJP has `1/(sigma_i^2 - sigma_j^2)` terms,
    // so a nearly rank-deficient matrix (a plain ramp is almost rank 1) would be testing
    // conditioning rather than the rule.
    let a = Buf::new(
        &[3, 3],
        &[2.00, 0.30, -0.10, 0.20, 1.50, 0.40, -0.30, 0.10, 1.10],
    );
    let cot = Buf::new(&[3], &[0.41, -0.27, 0.63]);
    check_grads(
        "svd_jacobi_singular_values",
        &[a],
        &cot,
        |ts, _g| {
            let (_u, s, _vt) = T::svd_jacobi(&ts[0], false);
            s
        },
        1e-3,
    );
}

/// Generalized symmetric-definite eigenvalues: `Ā = V diag(ḡ) Vᵀ`, `B̄ = -V diag(ḡ λ) Vᵀ`.
#[test]
fn fd_generalized_eigen_values() {
    let a_seed = ramp(&[3, 3], 0.35, 0.13);
    let b_seed = ramp(&[3, 3], 0.11, 0.05);
    let cot = Buf::new(&[3], &[0.41, -0.27, 0.63]);
    check_grads(
        "generalized_eigen_values",
        &[a_seed, b_seed],
        &cot,
        |ts, g| {
            // A symmetric; B symmetric positive definite (diagonally shifted).
            let a = T::add(ts[0], T::transpose(ts[0], &[1, 0]));
            let bb = T::add(ts[1], T::transpose(ts[1], &[1, 0]));
            let shift = T::convert_to_tensor(
                ArrayD::from_shape_vec(
                    IxDyn(&[3, 3]),
                    vec![4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
                )
                .expect("shift"),
                g,
            );
            let b = T::add(bb, shift);
            let (vals, _vecs) = T::generalized_eigen(&a, &b);
            vals
        },
        1e-3,
    );
}

/// Symmetric-matrix eigenvalues: `d(sum(cotangent · λ))/dA = V · diag(cotangent) · Vᵀ`
/// (Hellmann-Feynman). Both `EigenvaluesOp::grad` and `EigenExtractOp{component: 0}::grad`
/// used to (a) eagerly `.eval()` a tensor that fails whenever it traces back to a
/// `Variable` (rather than a `T::variable`/`T::convert_to_tensor` embedded constant), and
/// (b), even when that eval happened to succeed, `EigenvaluesOp` placed the cotangent
/// directly on the diagonal of a zero matrix (correct only when V = I) while
/// `EigenExtractOp` fabricated an outright zero regardless.
///
/// As in `fd_generalized_eigen_values` above, the loss symmetrizes a raw (unconstrained)
/// seed via `A + Aᵀ` before calling into eigen -- `eigendecomposition_gradient`'s shared
/// formula (`V diag(g) Vᵀ`) assumes an *orthogonal* eigenvector matrix, which only holds
/// once the matrix handed to `compute()` is exactly symmetric. Perturbing entries of an
/// already-symmetric buffer independently (what `check_grads` does) would momentarily
/// break that symmetry and divert the re-derivation in `EigenExtractGradOp::compute` onto
/// `compute_general_eigen`'s non-orthogonal path instead -- a real analytic-vs-FD
/// mismatch, but of the *test's* making, not the fix's; routing every perturbation
/// through the symmetrizing `+ Aᵀ` keeps the composed function smooth in the raw,
/// unconstrained seed.
#[test]
fn fd_eigenvalues_symmetric() {
    // `raw + rawᵀ` reconstructs [[4,1,0.5],[1,3,0.2],[0.5,0.2,2]] -- well-separated
    // eigenvalues, no near-degeneracies.
    let raw = Buf::new(&[3, 3], &[2.0, 0.6, 0.3, 0.4, 1.5, 0.1, 0.2, 0.1, 1.0]);
    let cot = Buf::new(&[3], &[0.7, -1.3, 0.4]);

    check_grads(
        "eigenvalues (via T::eigenvalues)",
        std::slice::from_ref(&raw),
        &cot,
        |t, _| {
            let a = T::add(t[0], T::transpose(t[0], &[1, 0]));
            T::eigenvalues(a)
        },
        LOOSE,
    );
    check_grads(
        "eigenvalues (via T::eigen().0)",
        &[raw],
        &cot,
        |t, _| {
            let a = T::add(t[0], T::transpose(t[0], &[1, 0]));
            T::eigen(a).0
        },
        LOOSE,
    );
}

/// `T::eigen(A).1` (raw eigenvectors) has no correctly-implemented reverse-mode gradient:
/// `eigendecomposition_gradient`'s eigenvector-cotangent handling does not match finite
/// differences (found while testing the fix below) and has not been re-derived
/// correctly, so `EigenExtractOp{component: 1}::grad` must fail loudly instead of
/// returning a plausible-looking but wrong gradient. A loss that depends only on the
/// eigenvalues from the same `eigen()` call must still work.
#[test]
fn fd_eigen_vectors_gradient_is_honestly_unsupported() {
    ag::run(|g| {
        let g: &Ctx = g;
        let raw = T::variable(
            Buf::new(&[3, 3], &[2.0, 0.6, 0.3, 0.4, 1.5, 0.1, 0.2, 0.1, 1.0]).to_array(),
            g,
        );
        let a = T::add(raw, T::transpose(raw, &[1, 0]));
        let (vals, vecs) = T::eigen(a);

        // Eigenvalues-only loss: must still produce a correct, finite gradient.
        let vals_loss = T::sum_all(vals);
        let vals_grads = T::grad(&[vals_loss], &[&raw]);
        let vals_grad_arr = vals_grads[0]
            .eval(g)
            .expect("eigenvalues-only gradient must evaluate");
        for &v in vals_grad_arr.iter() {
            assert!(v.is_finite(), "eigenvalues gradient must be finite: {v}");
        }

        // Eigenvectors-feeding loss: must fail loudly, not silently.
        let vecs_loss = T::sum_all(vecs);
        let vecs_grads = T::grad(&[vecs_loss], &[&raw]);
        let err = vecs_grads[0].eval(g);
        assert!(
            err.is_err(),
            "eigenvector gradient has no verified formula; it must not evaluate to a number"
        );
    });
}

/// `eigendecomposition_gradient`'s eigenvalue formula (`V diag(g) Vᵀ`) assumes an
/// orthogonal `V`, true only when the input to `eigen()`/`eigenvalues()` is exactly
/// symmetric. For a genuinely asymmetric matrix that must fail loudly rather than
/// silently reuse the symmetric-only formula (which would produce a plausible-looking
/// but generally wrong value, since a general matrix's eigenvectors are not orthogonal).
#[test]
fn fd_eigenvalues_asymmetric_gradient_is_honestly_unsupported() {
    ag::run(|g| {
        let g: &Ctx = g;
        // Diagonally dominant but NOT symmetric (a[0][1] != a[1][0], etc.).
        let a = T::variable(
            Buf::new(&[3, 3], &[5.0, 1.0, 0.3, 0.2, 4.0, 0.6, 0.1, 0.4, 3.0]).to_array(),
            g,
        );
        let vals = T::eigenvalues(a);
        let loss = T::sum_all(vals);
        let grads = T::grad(&[loss], &[&a]);
        let err = grads[0].eval(g);
        assert!(
            err.is_err(),
            "eigenvalue gradient of an asymmetric matrix has no verified formula; it must \
             not evaluate to a number"
        );
    });
}

/// Real-world usage pattern: parameters held in a `VariableEnvironment` (not
/// `T::variable`'s embedded-constant flavor). `Op::grad` only ever has a bare `&Graph`
/// (`ctx.graph()`), never the `Context`/`VariableEnvironment` needed to resolve a
/// `Variable` node, so any backward implementation that eagerly `.eval()`s a tensor
/// tracing back to one fails to resolve it -- this must not silently degrade to a
/// fabricated zero gradient.
#[test]
fn eigenvalues_gradient_is_correct_through_variable_environment() {
    let mut env = ag::VariableEnvironment::<f64>::new();
    let vid = env.set(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
        )
        .expect("a"),
    );

    let grad_arr = env.run(|g| {
        let a = g.variable_by_id(vid);
        let vals = T::eigenvalues(a);
        let loss = T::sum_all(vals);
        let grads = T::grad(&[loss], &[&a]);
        grads[0]
            .eval(g)
            .expect("eigenvalues gradient must evaluate")
    });

    // `d(sum(eigenvalues(A)))/dA = d(trace(A))/dA = I` (sum of eigenvalues is the trace,
    // independent of the eigenvectors).
    assert_eq!(grad_arr.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let got = grad_arr[[i, j]];
            assert!(
                (got - expected).abs() < 1e-6,
                "d(trace)/dA[{i},{j}] = {got}, expected {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Ops whose gradient is honestly unavailable
// ---------------------------------------------------------------------------

/// An unimplemented gradient must *fail loudly*, not silently report zero.
#[test]
fn unavailable_gradients_report_an_error_instead_of_zero() {
    ag::run(|g| {
        let g: &Ctx = g;
        let x = T::variable(ramp(&[3], 0.4, 0.2).to_array(), g);
        let y = T::map(x, |v| v.mapv(|e| e * e));
        let loss = T::sum_all(y);
        let grads = T::grad(&[loss], &[x]);
        let err = grads[0].eval(g);
        assert!(
            err.is_err(),
            "map() has no derivative; its gradient must not evaluate to a number"
        );
        // `Graph::eval_tensors_in` used to discard the *actual* `Op::compute` error and
        // report a generic "Failed to compute tensor N" for any tensor that never made it
        // into `computed_values`, regardless of why. It now remembers the first genuine
        // compute error seen during the pass and surfaces that instead, so the caller sees
        // `UnsupportedGradOp::message` (built in `MapOp::grad` via
        // `append_unsupported_grad`) rather than a bare node id.
        let message = format!("{:?}", err.expect_err("error"));
        assert!(!message.is_empty());
        assert!(
            message.contains("map(") && message.contains("no known derivative"),
            "expected the real `MapOp::grad` explanation to propagate, got: {message}"
        );
    });
}

// ---------------------------------------------------------------------------
// 12. Parallel reduction
// ---------------------------------------------------------------------------

/// `parallel_sum` used to append no gradient at all, which the backward pass turns into a
/// silent zero.
#[test]
fn fd_parallel_sum_axis0() {
    let x = ramp(&[3, 4], 0.4, 0.17);
    let cot = ramp(&[4], 0.29, -0.13);
    check_grads(
        "parallel_sum_axis0",
        &[x],
        &cot,
        |ts, _g| T::parallel_sum(&ts[0], &[0], false),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 13. Kronecker product and the cached-op wrapper
// ---------------------------------------------------------------------------

#[test]
fn fd_kronecker_product() {
    let a = ramp(&[2, 2], 0.4, 0.21);
    let b = ramp(&[2, 2], -0.6, 0.19);
    let cot = ramp(&[4, 4], 0.13, 0.07);
    check_grads(
        "kron",
        &[a, b],
        &cot,
        |ts, _g| T::kron_tensor(&ts[0], &ts[1]),
        LOOSE,
    );
}

#[test]
fn fd_cached_op_applies_its_own_chain_rule() {
    let x = ramp(&[4], 0.4, 0.23);
    let cot = ramp(&[4], 0.31, -0.17);
    check_grads(
        "cached_op_square",
        std::slice::from_ref(&x),
        &cot,
        |ts, _g| T::cached_op(&ts[0], "square"),
        LOOSE,
    );
    check_grads(
        "cached_op_sqrt",
        &[x],
        &cot,
        |ts, _g| T::cached_op(&ts[0], "sqrt"),
        LOOSE,
    );
}
