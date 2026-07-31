//! Permanent finite-difference gradient harness.
//!
//! This is the regression gate for the backward pass. Every check compares the
//! reverse-mode gradient produced by [`ag::tensor_ops::grad`] against a central
//! finite difference of the same scalar loss.
//!
//! Two properties of the harness are load-bearing and must not be "simplified":
//!
//! 1. **The cotangent is NON-UNIFORM.** An all-ones cotangent silently passes
//!    `transpose`, `gather`, `reduce_sum(axis)` and `symmetrize` even when their VJP
//!    is wrong, because permuting/broadcasting a constant vector is a no-op. Every
//!    loss below is `sum(cotangent * f(x))` with a cotangent whose entries are all
//!    distinct.
//! 2. **Inputs are built with `T::variable`.** `T::convert_to_tensor` marks the node
//!    non-differentiable, so gradients through it silently collapse to zero and a
//!    broken VJP would go unnoticed.
//!
//! The loss is scalar, so `sum(cotangent * f(x))` has the exact directional derivative
//! `<cotangent, J·dx>`; the reverse-mode result must equal `Jᵀ·cotangent`.

use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::{ArrayD, IxDyn};

type Ctx<'g> = ag::Context<'g, f64>;
type Tsr<'g> = ag::Tensor<'g, f64>;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A flat, shape-carrying input buffer so finite differences can perturb one scalar
/// at a time without depending on the memory layout of `ArrayD`.
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

/// A deterministic, strictly non-uniform ramp: `start + i * step`.
fn ramp(shape: &[usize], start: f64, step: f64) -> Buf {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();
    Buf::new(shape, &data)
}

/// A deterministic non-uniform ramp with alternating signs, so that sign-dependent
/// VJPs (`abs`, `relu`, `maximum`, ...) are exercised on both branches.
fn signed_ramp(shape: &[usize], start: f64, step: f64) -> Buf {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let v = start + (i as f64) * step;
            if i % 2 == 0 {
                v
            } else {
                -v
            }
        })
        .collect();
    Buf::new(shape, &data)
}

/// Evaluates the scalar loss `sum(cotangent * build(xs))` on plain (non-differentiable)
/// constants. Only the forward value is needed here.
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

/// Reverse-mode gradients of `sum(cotangent * build(xs))` w.r.t. every entry of `xs`.
fn analytic_grads<B>(name: &str, xs: &[Buf], cotangent: &Buf, build: &B) -> Vec<ArrayD<f64>>
where
    B: for<'g> Fn(&[Tsr<'g>], &'g Ctx<'g>) -> Tsr<'g>,
{
    ag::run(|g| {
        let g: &Ctx = g;
        let ts: Vec<Tsr> = xs.iter().map(|b| T::variable(b.to_array(), g)).collect();
        let y = build(&ts, g);

        // Self-check: the cotangent must have exactly the output's shape. Otherwise
        // `y * cot` broadcasts and the test would silently measure a different loss
        // than the one it claims to.
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
///
/// `tol` is a *relative* tolerance; the comparison is
/// `|analytic - numeric| <= tol * (1 + |numeric|)`.
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
            // Step scaled to the magnitude of the coordinate: cbrt(eps) ~= 6e-6 is the
            // optimal central-difference step for f64.
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

/// Convenience wrapper for single-input, single-output element-wise functions.
fn check_unary<B>(name: &str, x: Buf, build: B, tol: f64)
where
    B: for<'g> Fn(Tsr<'g>) -> Tsr<'g>,
{
    let cot = ramp(&x.shape, 0.37, 0.61);
    check_grads(name, &[x], &cot, move |ts, _g| build(ts[0]), tol);
}

const TOL: f64 = 1e-6;
const LOOSE: f64 = 1e-4;

// ---------------------------------------------------------------------------
// 1. Elementary math functions
//
// Every one of these had an identity gradient before backprop was routed through
// `Op::grad` (sqrt(4) returned 1.0 instead of 0.25, and so on).
// ---------------------------------------------------------------------------

#[test]
fn fd_elementary_math_positive_domain() {
    let x = ramp(&[2, 3], 0.4, 0.35); // strictly positive, non-uniform
    check_unary("sqrt", x.clone(), |t| T::sqrt(t), TOL);
    check_unary("exp", x.clone(), |t| T::exp(t), TOL);
    check_unary("ln", x.clone(), |t| T::ln(t), TOL);
    check_unary("log2", x.clone(), |t| T::log2(t), TOL);
    check_unary("log10", x.clone(), |t| T::log10(t), TOL);
    check_unary("exp2", x.clone(), |t| T::exp2(t), TOL);
    check_unary("exp10", x.clone(), |t| T::exp10(t), TOL);
    check_unary("inv", x.clone(), |t| T::inv(t), TOL);
    check_unary("inv_sqrt", x.clone(), |t| T::inv_sqrt(t), TOL);
    check_unary("square", x.clone(), |t| T::square(t), TOL);
    check_unary("sinh", x.clone(), |t| T::sinh(t), TOL);
    check_unary("cosh", x.clone(), |t| T::cosh(t), TOL);
    check_unary("asinh", x.clone(), |t| T::asinh(t), TOL);
    check_unary("pow_2.5", x, |t| T::pow(t, 2.5), TOL);
}

/// `lgamma`/`digamma` used to be identity-gradient DEFAULTs (`Op::grad` for `Digamma`
/// explicitly returned `None`, which the backward pass reads back as an implicit zero --
/// silently wrong, since trigamma(x) is never zero on this domain).
#[test]
fn fd_gamma_family() {
    // Away from the poles at non-positive integers, where lgamma/digamma/trigamma are all
    // smooth.
    let x = ramp(&[2, 3], 0.8, 0.31);
    check_unary("lgamma_f64", x.clone(), |t| T::lgamma_f64(t), TOL);
    check_unary("digamma_f64", x, |t| T::digamma_f64(t), TOL);
}

/// `trigamma`'s own derivative (the order-2 polygamma function, "tetragamma") is not
/// implemented. Differentiating `digamma` a *second* time must fail loudly instead of
/// silently reporting zero -- the same contract as `unavailable_gradients_report_an_error_instead_of_zero`
/// in `gradient_fd_harness_matrix.rs`, exercised through the gamma family specifically.
#[test]
fn digamma_second_derivative_is_honestly_unsupported() {
    ag::run(|g| {
        let g: &Ctx = g;
        let x = T::variable(ramp(&[3], 0.8, 0.31).to_array(), g);
        let y = T::digamma_f64(x);
        let loss = T::sum_all(y);
        let first_grads = T::grad(&[loss], &[x]);

        // First derivative (trigamma) is real and must evaluate to finite numbers.
        let gx = first_grads[0]
            .eval(g)
            .expect("digamma's gradient (trigamma) must evaluate");
        for &v in gx.iter() {
            assert!(v.is_finite(), "trigamma(x) must be finite here, got {v}");
        }

        // Second derivative (tetragamma) is unimplemented and must error.
        let first_loss = T::sum_all(first_grads[0]);
        let second_grads = T::grad(&[first_loss], &[x]);
        let err = second_grads[0].eval(g);
        assert!(
            err.is_err(),
            "trigamma() has no known derivative of its own; the second derivative of \
             digamma must not silently evaluate to a number"
        );
    });
}

#[test]
fn fd_elementary_math_trigonometry() {
    let x = ramp(&[2, 3], -0.9, 0.31); // inside (-1, 1) for asin/acos/atanh
    check_unary("sin", x.clone(), |t| T::sin(t), TOL);
    check_unary("cos", x.clone(), |t| T::cos(t), TOL);
    check_unary("tan", x.clone(), |t| T::tan(t), TOL);
    check_unary("tanh", x.clone(), |t| T::tanh(t), TOL);
    check_unary("asin", x.clone(), |t| T::asin(t), TOL);
    check_unary("acos", x.clone(), |t| T::acos(t), TOL);
    check_unary("atan", x.clone(), |t| T::atan(t), TOL);
    check_unary("atanh", x, |t| T::atanh(t), TOL);

    let above_one = ramp(&[4], 1.3, 0.45);
    check_unary("acosh", above_one, |t| T::acosh(t), TOL);
}

#[test]
fn fd_elementary_math_sign_sensitive() {
    // Alternating signs: `abs` had an inverted gradient (abs(-2) -> +1.0).
    let x = signed_ramp(&[6], 0.7, 0.4);
    check_unary("abs", x.clone(), |t| T::abs(t), TOL);
    check_unary("neg", x.clone(), |t| T::neg(t), TOL);
    check_unary("relu", x.clone(), |t| T::relu(t), TOL);
    check_unary("leaky_relu", x, |t| T::leaky_relu(t, 0.1), TOL);
}

#[test]
fn fd_binary_arithmetic_same_shape() {
    let a = ramp(&[2, 3], 0.6, 0.27);
    // Deliberately chosen so that no entry of `b` equals the matching entry of `a`:
    // `maximum`/`minimum` have no well-defined derivative at a tie (the analytic
    // sub-gradient sends the cotangent to *both* operands while a central difference
    // sees half of it), and a tie would test the tie-breaking rule, not the VJP.
    let b = ramp(&[2, 3], 1.45, -0.13);
    let cot = ramp(&[2, 3], 0.29, 0.44);

    check_grads(
        "add",
        &[a.clone(), b.clone()],
        &cot,
        |t, _| t[0] + t[1],
        TOL,
    );
    check_grads(
        "sub",
        &[a.clone(), b.clone()],
        &cot,
        |t, _| t[0] - t[1],
        TOL,
    );
    check_grads(
        "mul",
        &[a.clone(), b.clone()],
        &cot,
        |t, _| t[0] * t[1],
        TOL,
    );
    check_grads(
        "div",
        &[a.clone(), b.clone()],
        &cot,
        |t, _| t[0] / t[1],
        TOL,
    );
    check_grads(
        "maximum",
        &[a.clone(), b.clone()],
        &cot,
        |t, _| T::maximum(t[0], t[1]),
        TOL,
    );
    check_grads("minimum", &[a, b], &cot, |t, _| T::minimum(t[0], t[1]), TOL);
}

/// Broadcasting: the gradient w.r.t. the smaller operand must be the *sum* over the
/// broadcast axis, not a copy of the upstream cotangent.
#[test]
fn fd_binary_arithmetic_broadcast_reduces() {
    let x = ramp(&[3, 4], 0.5, 0.21);
    let bias = ramp(&[4], 0.9, -0.17);
    let cot = ramp(&[3, 4], 0.13, 0.29);

    check_grads(
        "broadcast_add",
        &[x.clone(), bias.clone()],
        &cot,
        |t, _| t[0] + t[1],
        TOL,
    );
    check_grads(
        "broadcast_mul",
        &[x.clone(), bias.clone()],
        &cot,
        |t, _| t[0] * t[1],
        TOL,
    );
    check_grads("broadcast_sub", &[x, bias], &cot, |t, _| t[0] - t[1], TOL);
}

/// `tensor_ops::broadcast_ops`'s public `broadcast_*` functions are a *separate*
/// implementation from the plain `+`/`-`/`*` operators exercised above
/// (`OptimizedBroadcastOp`, not `AddOp`/`SubOp`/`MulOp`). Its gradient used to ignore
/// broadcasting entirely: `broadcast_binary_op` builds the op's `BroadcastInfo` from a
/// hardcoded placeholder shape (`vec![1]` for both operands), so
/// `self.info.{left,right}_needs_broadcast` was always `false` and the smaller operand
/// received an un-reduced, mis-shaped gradient -- which surfaced as a hard `AddN` shape
/// mismatch the moment that gradient had to be accumulated with another contribution.
/// Separately, `broadcast_pow`/`broadcast_maximum`/`broadcast_minimum`'s forward pass used
/// to zip the two operands' *flat* iterators directly, silently truncating to the shorter
/// one and then panicking when reshaping the result back to the full output shape.
#[test]
fn fd_broadcast_ops_module_reduces() {
    let x = ramp(&[3, 4], 0.5, 0.21); // in [0.5, 3.11], always positive
    let cot = ramp(&[3, 4], 0.13, 0.29);

    let bias = ramp(&[4], 0.9, -0.17);
    check_grads(
        "broadcast_ops::broadcast_add",
        &[x.clone(), bias.clone()],
        &cot,
        |t, _| T::broadcast_add(&t[0], &t[1]),
        TOL,
    );
    check_grads(
        "broadcast_ops::broadcast_sub",
        &[x.clone(), bias.clone()],
        &cot,
        |t, _| T::broadcast_sub(&t[0], &t[1]),
        TOL,
    );
    check_grads(
        "broadcast_ops::broadcast_mul",
        &[x.clone(), bias.clone()],
        &cot,
        |t, _| T::broadcast_mul(&t[0], &t[1]),
        TOL,
    );

    let divisor = ramp(&[4], 1.3, 0.4); // away from zero
    check_grads(
        "broadcast_ops::broadcast_div",
        &[x.clone(), divisor],
        &cot,
        |t, _| T::broadcast_div(&t[0], &t[1]),
        TOL,
    );

    // `hi` is greater than every entry of `x` (x tops out at 3.11): the max/argmax mask
    // is 100% on the broadcast operand, so its reduced gradient must be non-trivial.
    let hi = ramp(&[4], 5.0, 0.4);
    check_grads(
        "broadcast_ops::broadcast_maximum",
        &[x.clone(), hi],
        &cot,
        |t, _| T::broadcast_maximum(&t[0], &t[1]),
        TOL,
    );
    // `higher` is greater than every entry of `x` too, so this time `x` (the non-broadcast
    // operand) wins the min and the broadcast operand's contribution is trivially zero --
    // the complementary case to `broadcast_maximum` above.
    let higher = ramp(&[4], 10.0, 0.4);
    check_grads(
        "broadcast_ops::broadcast_minimum",
        &[x.clone(), higher],
        &cot,
        |t, _| T::broadcast_minimum(&t[0], &t[1]),
        TOL,
    );

    // Power: base strictly positive (needed for the ln(x) term in d/dy), modest exponent.
    let exponent = ramp(&[4], 1.2, 0.3);
    check_grads(
        "broadcast_ops::broadcast_pow",
        &[x, exponent],
        &cot,
        |t, _| T::broadcast_pow(&t[0], &t[1]),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 2. Activations
// ---------------------------------------------------------------------------

#[test]
fn fd_activations() {
    let x = signed_ramp(&[8], 0.35, 0.29);
    check_unary("sigmoid", x.clone(), |t| T::sigmoid(t), TOL);
    check_unary("softplus", x.clone(), |t| T::softplus(t), TOL);
    check_unary("elu", x.clone(), |t| T::elu(t, 1.0), TOL);
    check_unary("swish", x.clone(), |t| T::swish(t), LOOSE);
    check_unary("gelu", x.clone(), |t| T::gelu(t), LOOSE);
    check_unary("mish", x, |t| T::mish(t), LOOSE);
}

// ---------------------------------------------------------------------------
// 3. Reductions — axis and all-reduce forms, non-uniform cotangent
//
// `reduce_sum` along an axis used to broadcast a single value back over the reduced
// axis, which is only correct for a uniform cotangent. `reduce_mean` lost the 1/N.
// `reduce_max`/`reduce_min` broadcast instead of masking.
// ---------------------------------------------------------------------------

#[test]
fn fd_reduce_sum_axis() {
    let x = ramp(&[3, 4], 0.2, 0.37);
    // Output shape [4]; a non-uniform cotangent is what distinguishes a correct
    // axis-aware backward from a plain broadcast.
    let cot = Buf::new(&[4], &[2.0, 5.0, -3.0, 0.5]);
    check_grads(
        "reduce_sum(axis=0)",
        std::slice::from_ref(&x),
        &cot,
        |t, _| T::reduce_sum(t[0], &[0], false),
        TOL,
    );

    let cot_rows = Buf::new(&[3], &[1.5, -2.5, 4.0]);
    check_grads(
        "reduce_sum(axis=1)",
        std::slice::from_ref(&x),
        &cot_rows,
        |t, _| T::reduce_sum(t[0], &[1], false),
        TOL,
    );

    let cot_keep = Buf::new(&[3, 1], &[1.5, -2.5, 4.0]);
    check_grads(
        "reduce_sum(axis=1, keep_dims)",
        &[x],
        &cot_keep,
        |t, _| T::reduce_sum(t[0], &[1], true),
        TOL,
    );
}

#[test]
fn fd_reduce_mean_axis() {
    let x = ramp(&[3, 4], -0.6, 0.29);
    let cot = Buf::new(&[4], &[2.0, 5.0, -3.0, 0.5]);
    check_grads(
        "reduce_mean(axis=0)",
        std::slice::from_ref(&x),
        &cot,
        |t, _| T::reduce_mean(t[0], &[0], false),
        TOL,
    );

    let scalar_cot = Buf::new(&[], &[1.0]);
    check_grads("mean_all", &[x], &scalar_cot, |t, _| T::mean_all(t[0]), TOL);
}

#[test]
fn fd_reduce_all_forms() {
    let x = ramp(&[2, 3], 0.4, 0.33);
    let scalar_cot = Buf::new(&[], &[1.0]);

    check_grads(
        "sum_all",
        std::slice::from_ref(&x),
        &scalar_cot,
        |t, _| T::sum_all(t[0]),
        TOL,
    );
    check_grads(
        "reduce_sum(all axes)",
        std::slice::from_ref(&x),
        &scalar_cot,
        |t, _| T::reduce_sum(t[0], &[0, 1], false),
        TOL,
    );
    check_grads(
        "reduce_logsumexp",
        std::slice::from_ref(&x),
        &Buf::new(&[3], &[1.7, -0.8, 2.3]),
        |t, _| T::reduce_logsumexp(t[0], 0, false),
        LOOSE,
    );
    check_grads(
        "reduce_prod(axis=0)",
        &[x],
        &Buf::new(&[3], &[1.7, -0.8, 2.3]),
        |t, _| T::reduce_prod(t[0], &[0], false),
        TOL,
    );
}

#[test]
fn fd_reduce_max_min_masks() {
    // Values are all distinct, so the max/min sub-gradient is unambiguous.
    let x = Buf::new(&[2, 3], &[0.3, 1.7, -0.4, 2.2, -1.1, 0.9]);
    let cot = Buf::new(&[3], &[1.5, -2.5, 4.0]);

    check_grads(
        "reduce_max(axis=0)",
        std::slice::from_ref(&x),
        &cot,
        |t, _| T::reduce_max(t[0], &[0], false),
        TOL,
    );
    check_grads(
        "reduce_min(axis=0)",
        &[x],
        &cot,
        |t, _| T::reduce_min(t[0], &[0], false),
        TOL,
    );
}

#[test]
fn fd_norm_reductions() {
    let x = signed_ramp(&[5], 0.6, 0.4);
    let scalar_cot = Buf::new(&[], &[1.0]);
    check_grads(
        "l2_norm",
        std::slice::from_ref(&x),
        &scalar_cot,
        |t, _| T::l2_norm(t[0], &[0], false),
        LOOSE,
    );
    check_grads(
        "l1_norm",
        &[x],
        &scalar_cot,
        |t, _| T::l1_norm(t[0], &[0], false),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 4. Matrix products
// ---------------------------------------------------------------------------

#[test]
fn fd_matmul() {
    let a = ramp(&[2, 3], 0.4, 0.21);
    let b = ramp(&[3, 4], -0.7, 0.19);
    let cot = ramp(&[2, 4], 0.31, 0.47);
    check_grads("matmul", &[a, b], &cot, |t, _| T::matmul(t[0], t[1]), TOL);
}

#[test]
fn fd_batch_matmul() {
    // 3-D batches: the old `ends_with("MatMul")` arm applied a 2-D transpose here.
    let a = ramp(&[2, 2, 3], 0.4, 0.11);
    let b = ramp(&[2, 3, 2], -0.6, 0.13);
    let cot = ramp(&[2, 2, 2], 0.23, 0.41);
    check_grads(
        "batch_matmul",
        &[a, b],
        &cot,
        |t, _| T::batch_matmul(t[0], t[1]),
        TOL,
    );
}

// ---------------------------------------------------------------------------
// 5. Shape manipulation — the class of ops a uniform cotangent cannot test at all
// ---------------------------------------------------------------------------

#[test]
fn fd_transpose() {
    let x = ramp(&[2, 3], 0.5, 0.37);
    // Cotangent shape is the transposed shape and every entry differs, so a VJP that
    // forgets to transpose back is caught.
    let cot = ramp(&[3, 2], 0.19, 0.53);
    check_grads(
        "transpose",
        &[x],
        &cot,
        |t, _| T::transpose(t[0], &[1, 0]),
        TOL,
    );
}

#[test]
fn fd_reshape_and_squeeze() {
    let x = ramp(&[2, 3], 0.5, 0.37);
    let cot_flat = ramp(&[6], 0.19, 0.53);
    check_grads(
        "reshape",
        std::slice::from_ref(&x),
        &cot_flat,
        |t, _| T::reshape(t[0], &[6]),
        TOL,
    );

    let y = ramp(&[1, 4], 0.3, 0.29);
    let cot4 = ramp(&[4], 0.21, 0.62);
    check_grads("squeeze", &[y], &cot4, |t, _| T::squeeze(t[0], &[0]), TOL);

    let z = ramp(&[4], 0.3, 0.29);
    let cot14 = ramp(&[1, 4], 0.21, 0.62);
    check_grads(
        "expand_dims",
        &[z],
        &cot14,
        |t, _| T::expand_dims(t[0], &[0]),
        TOL,
    );
}

#[test]
fn fd_slice() {
    let x = ramp(&[6], 0.5, 0.37);
    let cot = ramp(&[3], 0.19, 0.53);
    check_grads(
        "slice",
        &[x],
        &cot,
        |t, _| T::slice(t[0], [1_isize], [4_isize]),
        TOL,
    );
}

#[test]
fn fd_concat() {
    let a = ramp(&[2, 2], 0.5, 0.37);
    let b = ramp(&[2, 3], -0.4, 0.23);
    let cot = ramp(&[2, 5], 0.11, 0.29);
    check_grads(
        "concat(axis=1)",
        &[a, b],
        &cot,
        |t, _| T::concat(&[t[0], t[1]], 1),
        TOL,
    );
}

#[test]
fn fd_gather_counts_index_multiplicity() {
    let x = ramp(&[4, 2], 0.5, 0.37);
    // Index 1 appears twice: its gradient must be the SUM of both rows' cotangents.
    let cot = ramp(&[3, 2], 0.17, 0.41);
    check_grads(
        "gather",
        &[x],
        &cot,
        |t, g| {
            let idx = T::convert_to_tensor(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f64, 3.0, 1.0]).expect("index array"),
                g,
            );
            T::gather(t[0], idx, 0)
        },
        TOL,
    );
}

#[test]
fn fd_split() {
    let x = ramp(&[5], 0.5, 0.37);
    let cot = ramp(&[2], 0.19, 0.53);
    check_grads(
        "split[0]",
        std::slice::from_ref(&x),
        &cot,
        |t, _| T::split(t[0], &[2, 3], 0)[0],
        TOL,
    );
    let cot1 = ramp(&[3], 0.23, 0.61);
    check_grads(
        "split[1]",
        &[x],
        &cot1,
        |t, _| T::split(t[0], &[2, 3], 0)[1],
        TOL,
    );
}

// ---------------------------------------------------------------------------
// 6. Softmax family and cross-entropies
// ---------------------------------------------------------------------------

#[test]
fn fd_softmax_family() {
    let x = ramp(&[2, 4], -0.8, 0.31);
    let cot = ramp(&[2, 4], 0.17, 0.43);
    check_grads(
        "softmax",
        std::slice::from_ref(&x),
        &cot,
        |t, _| T::softmax(t[0], -1),
        LOOSE,
    );
    check_grads(
        "log_softmax",
        &[x],
        &cot,
        |t, _| T::log_softmax(t[0], -1),
        LOOSE,
    );
}

#[test]
fn fd_softmax_cross_entropy() {
    let logits = ramp(&[2, 3], -0.7, 0.29);
    // One-hot-ish but non-uniform targets; gradient is taken w.r.t. the logits only.
    let targets = Buf::new(&[2, 3], &[0.7, 0.2, 0.1, 0.1, 0.3, 0.6]);
    // `softmax_cross_entropy` reduces the class axis away, so the loss is `[batch]`.
    let cot = Buf::new(&[2], &[1.3, -0.7]);
    check_grads(
        "softmax_cross_entropy",
        &[logits],
        &cot,
        move |t, g| {
            let tgt = T::convert_to_tensor(targets.to_array(), g);
            T::softmax_cross_entropy(t[0], tgt)
        },
        LOOSE,
    );
}

#[test]
fn fd_sigmoid_cross_entropy() {
    let logits = signed_ramp(&[2, 3], 0.4, 0.27);
    let targets = Buf::new(&[2, 3], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let cot = ramp(&[2, 3], 0.31, 0.19);
    check_grads(
        "sigmoid_cross_entropy",
        &[logits],
        &cot,
        move |t, g| {
            let tgt = T::convert_to_tensor(targets.to_array(), g);
            T::sigmoid_cross_entropy(t[0], tgt)
        },
        LOOSE,
    );
}

#[test]
fn fd_sparse_softmax_cross_entropy() {
    let logits = ramp(&[3, 4], -0.5, 0.23);
    let labels = Buf::new(&[3], &[2.0, 0.0, 3.0]);
    let cot = Buf::new(&[3, 1], &[1.1, -0.6, 2.3]);
    check_grads(
        "sparse_softmax_cross_entropy",
        &[logits],
        &cot,
        move |t, g| {
            let lbl = T::convert_to_tensor(labels.to_array(), g);
            T::sparse_softmax_cross_entropy(t[0], lbl)
        },
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 7. Convolution / pooling
// ---------------------------------------------------------------------------

#[test]
fn fd_conv2d() {
    // [batch=1, in_ch=1, 4, 4] * [out_ch=2, in_ch=1, 2, 2] -> [1, 2, 3, 3]
    let x = ramp(&[1, 1, 4, 4], 0.2, 0.13);
    let w = signed_ramp(&[2, 1, 2, 2], 0.4, 0.11);
    let cot = ramp(&[1, 2, 3, 3], 0.09, 0.17);
    check_grads(
        "conv2d",
        &[x, w],
        &cot,
        |t, _| T::conv2d(t[0], t[1], 0, 1),
        LOOSE,
    );
}

#[test]
fn fd_max_pool2d() {
    // Distinct values so the arg-max is unambiguous and the sub-gradient is exact.
    let x = ramp(&[1, 1, 4, 4], 0.2, 0.37);
    let cot = ramp(&[1, 1, 2, 2], 0.11, 0.53);
    check_grads(
        "max_pool2d",
        &[x],
        &cot,
        |t, _| T::max_pool2d(t[0], 2, 0, 2),
        LOOSE,
    );
}

// ---------------------------------------------------------------------------
// 8. Custom-gradient API
//
// The whole public custom-gradient surface used to be a silent no-op: the user's
// backward was never called, `scale_gradient` did not scale and `detach` did not
// detach. These are exact-value checks, not finite differences, because the custom
// backward deliberately does NOT equal the true derivative.
// ---------------------------------------------------------------------------

mod custom_gradient_api {
    use super::*;
    use scirs2_autograd::custom_gradient::{custom_op, detach, scale_gradient, CustomGradientOp};
    use scirs2_core::ndarray::{ArrayD, ArrayViewD};
    use std::sync::Arc;

    /// Forward = identity, backward = 3 * upstream. If the user's backward is
    /// ignored, the gradient of `sum(x)` comes out as 1 instead of 3.
    struct TripleGrad;

    impl CustomGradientOp<f64> for TripleGrad {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, ag::error::OpError> {
            Ok(inputs[0].to_owned())
        }

        fn backward<'g>(
            &self,
            output_grad: &ag::Tensor<'g, f64>,
            _saved: &[ag::Tensor<'g, f64>],
            _ctx: &'g ag::Graph<f64>,
        ) -> Vec<Option<ag::Tensor<'g, f64>>> {
            vec![Some(*output_grad * 3.0)]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "TripleGrad"
        }
    }

    #[test]
    fn custom_op_backward_is_actually_applied() {
        ag::run(|g| {
            let g: &Ctx = g;
            let x = T::variable(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("x"),
                g,
            );
            let y = custom_op(Arc::new(TripleGrad), &[x], g);
            let loss = T::sum_all(y);
            let grads = T::grad(&[loss], &[x]);
            let gx = grads[0].eval(g).expect("custom_op gradient eval");
            for (i, v) in gx.iter().enumerate() {
                assert!(
                    (v - 3.0).abs() < 1e-12,
                    "custom backward must be applied: gx[{i}] = {v}, expected 3.0"
                );
            }
        });
    }

    #[test]
    fn scale_gradient_scales() {
        ag::run(|g| {
            let g: &Ctx = g;
            let x = T::variable(
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, 1.5, -2.0, 3.25]).expect("x"),
                g,
            );
            // f = sum(2.5 * x^2) so the true gradient is 5*x; the reversal layer must
            // multiply it by -2.
            let y = scale_gradient(T::square(x), -2.0, g);
            let loss = T::sum_all(T::scalar_mul(y, 2.5));
            let grads = T::grad(&[loss], &[x]);
            let gx = grads[0].eval(g).expect("scale_gradient eval");
            let expected = [0.5, 1.5, -2.0, 3.25].map(|v: f64| -2.0 * 5.0 * v);
            for (i, exp) in expected.iter().enumerate() {
                assert!(
                    (gx[[i]] - exp).abs() < 1e-9,
                    "scale_gradient(-2.0): gx[{i}] = {}, expected {exp}",
                    gx[[i]]
                );
            }
        });
    }

    #[test]
    fn detach_blocks_the_gradient() {
        ag::run(|g| {
            let g: &Ctx = g;
            let x = T::variable(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("x"),
                g,
            );
            // loss = sum(detach(x^2)) + sum(4*x): only the second term may contribute.
            let blocked = detach(T::square(x), g);
            let loss = T::sum_all(blocked) + T::sum_all(T::scalar_mul(x, 4.0));
            let grads = T::grad(&[loss], &[x]);
            let gx = grads[0].eval(g).expect("detach eval");
            for (i, v) in gx.iter().enumerate() {
                assert!(
                    (v - 4.0).abs() < 1e-12,
                    "detach must block the x^2 branch: gx[{i}] = {v}, expected 4.0"
                );
            }
        });
    }

    #[test]
    fn stop_gradient_blocks_the_gradient() {
        ag::run(|g| {
            let g: &Ctx = g;
            let x = T::variable(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("x"),
                g,
            );
            let blocked = T::stop_gradient(T::square(x));
            let loss = T::sum_all(blocked) + T::sum_all(T::scalar_mul(x, 4.0));
            let grads = T::grad(&[loss], &[x]);
            let gx = grads[0].eval(g).expect("stop_gradient eval");
            for (i, v) in gx.iter().enumerate() {
                assert!(
                    (v - 4.0).abs() < 1e-12,
                    "stop_gradient must block the x^2 branch: gx[{i}] = {v}, expected 4.0"
                );
            }
        });
    }
}

// ---------------------------------------------------------------------------
// 9. Regression guard: the silent-identity default is gone
// ---------------------------------------------------------------------------

/// `pow` was the op that exposed the defect: `d/dx x^2` at x = 3 returned 1 (the
/// upstream cotangent passed straight through) instead of 6.
#[test]
fn pow_gradient_is_not_the_identity() {
    ag::run(|g| {
        let g: &Ctx = g;
        let x = T::variable(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).expect("x"),
            g,
        );
        let y = T::pow(x, 2.0);
        let grads = T::grad(&[y], &[x]);
        let gx = grads[0].eval(g).expect("pow gradient eval");
        assert!(
            (gx[[0]] - 6.0).abs() < 1e-9,
            "d/dx x^2 at x=3 must be 6, got {}",
            gx[[0]]
        );
    });
}

/// Composition check: a chain that mixes reduction, transpose and an elementary
/// function. Each stage used to be individually wrong in a way that a uniform
/// cotangent hides.
#[test]
fn fd_composite_chain() {
    let x = ramp(&[3, 4], 0.35, 0.17);
    let cot = Buf::new(&[4], &[1.3, -0.8, 2.1, 0.4]);
    check_grads(
        "sum(axis=0) . transpose . sqrt",
        &[x],
        &cot,
        |t, _| {
            let s = T::sqrt(t[0]);
            let tr = T::transpose(s, &[1, 0]);
            T::reduce_sum(tr, &[1], false)
        },
        TOL,
    );
}

// ---------------------------------------------------------------------------
// 10. Gradient-checking / clipping helpers that used to report success blindly
// ---------------------------------------------------------------------------

mod helper_apis {
    use scirs2_autograd as ag;
    use scirs2_autograd::gradient_clipping::{
        AdaptiveClipByNorm, ClipByGlobalNorm, ClipByNorm, GradientClipper,
    };
    use scirs2_autograd::tensor_ops as T;
    use scirs2_autograd::test_helper::gradient_check;
    use scirs2_autograd::variable::GetVariableTensor;

    /// `gradient_check` must actually verify the gradient it is handed. It used to
    /// discard every argument and return `true` unconditionally.
    #[test]
    fn gradient_check_verifies_a_real_variable() {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("w")
            .set(scirs2_core::ndarray::Array1::from(vec![0.7_f64, -1.3, 2.1]).into_dyn());

        env.run(|ctx| {
            let w = ctx.variable("w");
            // f(w) = sum(w^3): analytically 3w^2, which the FD probe must reproduce.
            let f = T::sum_all(T::mul(T::mul(w, w), w));
            assert!(
                gradient_check(ctx, &f, &[w], 1e-4, 1e-4),
                "gradient_check must accept the correct gradient of sum(w^3)"
            );
        });
    }

    /// A non-variable parameter cannot be perturbed, so nothing can be verified: the
    /// honest answer is `false`, not the old unconditional `true`.
    #[test]
    fn gradient_check_rejects_unverifiable_params() {
        ag::run(|ctx: &mut ag::Context<f64>| {
            let x = T::variable(
                scirs2_core::ndarray::Array1::from(vec![1.0_f64, 2.0]).into_dyn(),
                ctx,
            );
            let f = T::sum_all(T::square(x));
            // `x` here is a graph constant, not a VariableEnvironment variable.
            assert!(
                !gradient_check(ctx, &f, &[x], 1e-4, 1e-4),
                "gradient_check must not claim success for a parameter it cannot perturb"
            );
        });
    }

    /// `gradient_check` must not just accept a *plausible-looking* verdict: feed it a
    /// deliberately wrong "analytical" gradient and confirm it is rejected.
    /// `stop_gradient` severs only the *symbolic* backward edge -- `T::grad` reports an
    /// all-zero gradient for `w` -- while forward evaluation still reads straight
    /// through to `w`'s current value, so perturbing `w` demonstrably still changes the
    /// loss. That is a genuine mismatch between the reported and true gradient, not a
    /// contrived one, and `gradient_check` must catch it.
    #[test]
    fn gradient_check_fails_on_a_deliberately_wrong_gradient() {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("w")
            .set(scirs2_core::ndarray::Array1::from(vec![0.7_f64, -1.3, 2.1]).into_dyn());

        env.run(|ctx| {
            let w = ctx.variable("w");
            // sum(stop_gradient(w)^2): forward value still depends on w (so perturbing
            // it changes the loss), but `T::grad` reports a zero gradient for `w`
            // because `stop_gradient` cuts the backward edge.
            let f = T::sum_all(T::square(T::stop_gradient(w)));
            assert!(
                !gradient_check(ctx, &f, &[w], 1e-4, 1e-4),
                "gradient_check must reject the zeroed gradient behind a stop_gradient \
                 boundary, which disagrees with the numerical probe"
            );
        });
    }

    #[test]
    fn clip_by_norm_reports_whether_it_clipped() {
        ag::run(|ctx: &mut ag::Context<f64>| {
            // Frobenius norm of [3, 4] is exactly 5.
            let big = T::convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![3.0_f64, 4.0]).into_dyn(),
                ctx,
            );
            let mut clipper = ClipByNorm::new(1.0_f64);
            let _ = clipper.clip_gradients(&[big]);
            assert!(
                clipper.was_clipped(),
                "a gradient of norm 5 clipped at 1.0 must report was_clipped()"
            );
            let stats = clipper.get_clipping_stats();
            assert_eq!(stats.num_clipped, 1);
            assert!(
                stats
                    .original_norm
                    .map(|n: f64| (n - 5.0).abs() < 1e-9)
                    .unwrap_or(false),
                "measured original norm should be 5, got {:?}",
                stats.original_norm
            );

            let small = T::convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![0.3_f64, 0.4]).into_dyn(),
                ctx,
            );
            let mut clipper2 = ClipByNorm::new(1.0_f64);
            let _ = clipper2.clip_gradients(&[small]);
            assert!(
                !clipper2.was_clipped(),
                "a gradient of norm 0.5 clipped at 1.0 must NOT report was_clipped()"
            );
        });
    }

    #[test]
    fn clip_by_global_norm_reports_whether_it_clipped() {
        ag::run(|ctx: &mut ag::Context<f64>| {
            let g1 = T::convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![3.0_f64, 4.0]).into_dyn(),
                ctx,
            );
            let g2 = T::convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![12.0_f64]).into_dyn(),
                ctx,
            );
            // global norm = sqrt(25 + 144) = 13
            let mut clipper = ClipByGlobalNorm::new(2.0_f64);
            let _ = clipper.clip_gradients(&[g1, g2]);
            assert!(clipper.was_clipped());
            let stats = clipper.get_clipping_stats();
            assert!(
                stats
                    .original_norm
                    .map(|n: f64| (n - 13.0).abs() < 1e-9)
                    .unwrap_or(false),
                "measured global norm should be 13, got {:?}",
                stats.original_norm
            );
        });
    }

    /// The adaptive clipper must move its own threshold; it used to change only when the
    /// caller invoked `set_threshold` by hand.
    #[test]
    fn adaptive_clipper_actually_adapts() {
        ag::run(|ctx: &mut ag::Context<f64>| {
            let big = T::convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![6.0_f64, 8.0]).into_dyn(),
                ctx,
            );
            let mut clipper = AdaptiveClipByNorm::new(1.0_f64, 0.5_f64);
            assert_eq!(clipper.current_threshold(), 1.0);
            let _ = clipper.clip_gradients(std::slice::from_ref(&big));
            let after_one = clipper.current_threshold();
            assert!(
                after_one > 1.0,
                "threshold must move towards the observed norm (10), got {after_one}"
            );
            let _ = clipper.clip_gradients(std::slice::from_ref(&big));
            assert!(
                clipper.current_threshold() > after_one,
                "threshold must keep converging towards the observed norm"
            );
            assert!(
                clipper
                    .observed_norm_ema()
                    .map(|n: f64| n > 0.0)
                    .unwrap_or(false),
                "the running norm average must be populated"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// 11. `custom_unary_op`: the caller-supplied backward closure
//
// `custom_unary_op` stores a user closure and must *call* it during backprop. Its
// `ClosureOp::grad` used to discard the closure entirely and push the incoming
// cotangent through unchanged ("lifetime issues"), so every op built with this
// function silently reported `d f(x) / dx = 1` no matter what rule the caller
// registered. The closure bound is now higher-ranked over the graph lifetime
// (`for<'graph> Fn(&Tensor<'graph, F>, ..) -> Option<Tensor<'graph, F>>`), which is
// what lets a `'static` `Op` hold it and invoke it with the backward pass's own
// lifetime.
// ---------------------------------------------------------------------------

/// The registered rule is used *verbatim*.
///
/// The forward is `x^2`, so three answers are distinguishable in one shot:
/// `1` (the old pass-through), `2x` (the mathematically "expected" derivative, which
/// would mean the closure was ignored in favour of something inferred), and `5`
/// (the rule actually registered). Only the last one is correct here: a custom
/// gradient means *the caller's* rule, right or wrong.
#[test]
fn custom_unary_op_backward_closure_is_applied_verbatim() {
    let x_buf = ramp(&[2, 3], 0.4, 0.35);
    let (shape, values) = ag::run(|g| {
        let g: &Ctx = g;
        let x = T::variable(x_buf.to_array(), g);
        let y = ag::custom_unary_op(
            "square_with_five_times_backward",
            |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| e * e),
            |gy: &Tsr, _x: &Tsr, _y: &Tsr| Some(T::scalar_mul(*gy, 5.0)),
            x,
            g,
        );
        let loss = T::sum_all(y);
        let gx = T::grad(&[loss], &[x])[0];
        let arr = gx.eval(g).expect("custom_unary_op gradient must evaluate");
        (
            arr.shape().to_vec(),
            arr.iter().copied().collect::<Vec<f64>>(),
        )
    });

    assert_eq!(
        shape, x_buf.shape,
        "the gradient must have the input's shape, got {shape:?}"
    );
    for (i, &v) in values.iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1e-12,
            "custom_unary_op backward `5 * gy` must yield exactly 5 at index {i}, got {v} \
             (1 means the closure was ignored, {} means it was replaced by d(x^2)/dx)",
            2.0 * x_buf.data[i]
        );
    }
}

/// A backward closure that reads the *output* tensor (`y`) it was handed.
/// `d tanh(x) / dx = 1 - tanh(x)^2`, expressed through `y`, must survive finite
/// differences. Under the old pass-through this reported a constant 1.
#[test]
fn fd_custom_unary_op_backward_reads_the_output() {
    let x = ramp(&[2, 3], -0.6, 0.31);
    let cot = ramp(&[2, 3], 0.37, 0.61);
    check_grads(
        "custom_unary_op(tanh)",
        &[x],
        &cot,
        |ts, g| {
            ag::custom_unary_op(
                "tanh_via_closure",
                |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| e.tanh()),
                // 1 - y^2, built purely from the tensors the closure was handed: a
                // closure that satisfies the higher-ranked bound cannot capture a
                // graph handle from the enclosing scope.
                |gy: &Tsr, _x: &Tsr, y: &Tsr| Some(*gy * (T::square(*y) * -1.0 + 1.0)),
                ts[0],
                g,
            )
        },
        TOL,
    );
}

/// A backward closure that reads the *input* tensor (`x`) it was handed:
/// `d/dx 1/(1 + x^2) = -2x / (1 + x^2)^2`.
#[test]
fn fd_custom_unary_op_backward_reads_the_input() {
    let x = signed_ramp(&[5], 0.35, 0.27);
    let cot = ramp(&[5], 0.41, -0.23);
    check_grads(
        "custom_unary_op(cauchy)",
        &[x],
        &cot,
        |ts, g| {
            ag::custom_unary_op(
                "cauchy_via_closure",
                |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| 1.0 / (1.0 + e * e)),
                |gy: &Tsr, x: &Tsr, _y: &Tsr| {
                    let denom = T::square(T::square(*x) + 1.0);
                    Some(*gy * ((*x * -2.0) / denom))
                },
                ts[0],
                g,
            )
        },
        TOL,
    );
}

/// Returning `None` from the closure must block the gradient, not fall back to the
/// pass-through.
#[test]
fn custom_unary_op_backward_none_blocks_the_gradient() {
    let x_buf = ramp(&[4], 0.5, 0.23);
    let values = ag::run(|g| {
        let g: &Ctx = g;
        let x = T::variable(x_buf.to_array(), g);
        let y = ag::custom_unary_op(
            "square_with_no_backward",
            |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| e * e),
            |_gy: &Tsr, _x: &Tsr, _y: &Tsr| None,
            x,
            g,
        );
        let loss = T::sum_all(y);
        let gx = T::grad(&[loss], &[x])[0];
        let arr = gx.eval(g).expect("blocked gradient must still evaluate");
        arr.iter().copied().collect::<Vec<f64>>()
    });
    for (i, &v) in values.iter().enumerate() {
        assert!(
            v.abs() < 1e-12,
            "a `None` backward must block the gradient, got {v} at index {i}"
        );
    }
}

/// Negative test: a custom op whose registered backward is *wrong* must be caught by
/// `gradient_check`.
///
/// `custom_unary_op` deliberately applies the caller's rule without questioning it, so
/// the only thing standing between a mistyped custom gradient and a silently wrong
/// model is a numerical check. This pins that `gradient_check` performs one. The
/// companion positive control below is what rules out "rejects everything".
#[test]
fn custom_unary_op_with_a_wrong_backward_is_caught_by_gradient_check() {
    use scirs2_autograd::test_helper::gradient_check;
    use scirs2_autograd::variable::GetVariableTensor;

    let mut env = ag::VariableEnvironment::<f64>::new();
    env.name("w")
        .set(scirs2_core::ndarray::Array1::from(vec![0.7_f64, -1.3, 2.1]).into_dyn());

    env.run(|ctx| {
        let w = ctx.variable("w");
        // Forward x^2, backward `7 * gy` -- the true rule is `2x * gy`, and 2x is
        // [1.4, -2.6, 4.2], nowhere near 7.
        let wrong = ag::custom_unary_op(
            "square_with_wrong_backward",
            |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| e * e),
            |gy: &Tsr, _x: &Tsr, _y: &Tsr| Some(*gy * 7.0),
            w,
            ctx,
        );
        assert!(
            !gradient_check(ctx, &wrong, &[w], 1e-4, 1e-4),
            "gradient_check must reject a custom_unary_op whose registered backward \
             disagrees with the numerical derivative of its own forward"
        );
    });
}

/// Positive control for the test above: the *same* forward with the *correct*
/// registered backward must pass `gradient_check`. Without this control the negative
/// test could be passing for the wrong reason (e.g. `gradient_check` rejecting every
/// custom op on principle).
#[test]
fn custom_unary_op_with_the_right_backward_passes_gradient_check() {
    use scirs2_autograd::test_helper::gradient_check;
    use scirs2_autograd::variable::GetVariableTensor;

    let mut env = ag::VariableEnvironment::<f64>::new();
    env.name("w")
        .set(scirs2_core::ndarray::Array1::from(vec![0.7_f64, -1.3, 2.1]).into_dyn());

    env.run(|ctx| {
        let w = ctx.variable("w");
        let right = ag::custom_unary_op(
            "square_with_right_backward",
            |v: &scirs2_core::ndarray::ArrayViewD<f64>| v.mapv(|e| e * e),
            |gy: &Tsr, x: &Tsr, _y: &Tsr| Some(T::mul(*gy, T::scalar_mul(*x, 2.0))),
            w,
            ctx,
        );
        assert!(
            gradient_check(ctx, &right, &[w], 1e-4, 1e-4),
            "gradient_check must accept a custom_unary_op whose registered backward is \
             the true derivative of its forward"
        );
    });
}

// ---------------------------------------------------------------------------
// 12. `norm_frobenius`: the standalone Frobenius-norm op
//
// `tensor_ops::norm_frobenius` is `norm_ops::FrobeniusNormOp`, a monolithic op
// distinct from the composite `frobenius_norm` (square -> sum_all -> sqrt). Its
// backward used to `.eval()` the input and the incoming cotangent *while the backward
// graph was being built* and splice the result back in as a `convert_to_tensor`
// constant -- returning no gradient at all (read back as zero) whenever either eval
// could not be satisfied that early.
//
// `fd_norm_frobenius` alone does NOT catch that: `check_grads` feeds plain graph
// constants, which *are* evaluable at build time, so the snapshot happens to be right
// there. The two tests after it are the ones that separate a real gradient node from a
// spliced-in number: with the old implementation both measured 0 where the exact
// answer is x / ||x||_F.
// ---------------------------------------------------------------------------

#[test]
fn fd_norm_frobenius() {
    let x = signed_ramp(&[3, 2], 0.4, 0.35);
    let cot = Buf::new(&[], &[0.73]);
    check_grads(
        "norm_frobenius",
        &[x],
        &cot,
        |ts, _g| T::norm_frobenius(&ts[0]),
        TOL,
    );
}

/// A placeholder has nothing fed to it while the backward graph is being built, so the
/// old eval-inside-`grad` implementation gave up and reported no gradient at all --
/// which the backward pass reads back as a silent zero. Measured on the old code:
/// `gradient[0] = 0` where the exact answer is 0.6.
#[test]
fn norm_frobenius_gradient_flows_through_a_placeholder() {
    ag::run(|ctx: &mut ag::Context<f64>| {
        let x = ctx.placeholder("x", &[2, 2]);
        let n = T::norm_frobenius(&x);
        let gx = T::grad(&[n], &[x])[0];

        // ||[[3, 4], [0, 0]]||_F = 5, so the gradient is [[0.6, 0.8], [0, 0]].
        let x_val = scirs2_core::ndarray::array![[3.0_f64, 4.0], [0.0, 0.0]];
        let result = ctx
            .evaluator()
            .push(&gx)
            .feed(x, x_val.view().into_dyn())
            .run();
        let arr = result[0]
            .as_ref()
            .expect("norm_frobenius gradient must evaluate for a fed placeholder");
        assert_eq!(arr.shape(), &[2, 2]);
        let expected = [0.6, 0.8, 0.0, 0.0];
        for (i, (&got, &want)) in arr.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "norm_frobenius placeholder gradient[{i}] = {got}, expected {want}"
            );
        }
    });
}

/// The gradient must be a live graph node, not a constant snapshot: re-evaluating it
/// after the variable changes must produce the gradient at the *new* value. An
/// optimizer loop is exactly this pattern, and a frozen (or, as the old code actually
/// did here, an all-zero) gradient makes training silently stop learning.
#[test]
fn norm_frobenius_gradient_tracks_the_current_variable_value() {
    use scirs2_autograd::variable::GetVariableTensor;

    let mut env = ag::VariableEnvironment::<f64>::new();
    env.name("w")
        .set(scirs2_core::ndarray::array![[3.0_f64, 4.0], [0.0, 0.0]].into_dyn());

    env.run(|ctx| {
        let w = ctx.variable("w");
        let gw = T::grad(&[T::norm_frobenius(&w)], &[w])[0];

        let first = gw.eval(ctx).expect("first gradient eval failed");
        let first_vals: Vec<f64> = first.iter().copied().collect();
        for (i, (&got, &want)) in first_vals
            .iter()
            .zip([0.6, 0.8, 0.0, 0.0].iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-9,
                "gradient at w = [[3,4],[0,0]]: [{i}] = {got}, expected {want}"
            );
        }

        // Move the variable to [[0, 0], [6, 8]] (norm 10) in place. The environment
        // holds exactly one variable, so the first entry is `w`.
        let (_, cell) = ctx
            .env()
            .iter()
            .next()
            .expect("the environment must contain the variable w");
        {
            let mut view = cell.borrow_mut();
            for (slot, value) in view.iter_mut().zip([0.0, 0.0, 6.0, 8.0]) {
                *slot = value;
            }
        }

        let second = gw.eval(ctx).expect("second gradient eval failed");
        let second_vals: Vec<f64> = second.iter().copied().collect();
        for (i, (&got, &want)) in second_vals
            .iter()
            .zip([0.0, 0.0, 0.6, 0.8].iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-9,
                "gradient after moving w to [[0,0],[6,8]]: [{i}] = {got}, expected {want} \
                 (a stale value here means the gradient was baked in as a constant)"
            );
        }
    });
}
