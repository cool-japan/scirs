//! EML Symbolic Gradient vs Float-Tape Gradient — Criterion Benchmarks
//!
//! Measures wall time of computing gradients via the EML symbolic path
//! (`eml_scalar_op` + `T::grad`) against the standard float-tape path
//! (`T::sin` / `T::exp` / `T::pow` / operator overloads + `T::grad`).
//!
//! Each benchmark group runs 100 gradient evaluations at evenly-spaced x
//! values to amortise graph-construction overhead and produce a stable
//! throughput envelope.
//!
//! ## Groups
//!
//! | Group               | EML op                         | Tape op            |
//! |---------------------|--------------------------------|--------------------|
//! | `x_squared_grad`    | `Pow(Var(0), Const(2.0))`     | `T::pow(x, 2.0)`  |
//! | `sin_grad`          | `Sin(Var(0))`                  | `T::sin(x)`        |
//! | `exp_grad`          | `Exp(Var(0))`                  | `T::exp(x)`        |
//! | `composition_grad`  | `eml(x²) * 2.0` (mixed)       | `T::pow(x,2)*2.0`  |
//! | `multi_input_grad`  | `Mul(Var(0), Var(1))` / ∂x    | `x * y` / ∂x      |

#![cfg(feature = "symbolic")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::arr0;
use scirs2_symbolic::eml::LoweredOp;
use std::hint::black_box;
use std::sync::Arc;

const ITERS: usize = 100;

// ---------------------------------------------------------------------------
// Benchmark 1: x² gradient — EML vs tape
// ---------------------------------------------------------------------------

fn bench_x_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("x_squared_grad");
    group.throughput(Throughput::Elements(ITERS as u64));

    let op = Arc::new(LoweredOp::Pow(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Const(2.0)),
    ));

    // EML path: build LoweredOp once, evaluate gradient 100 times
    group.bench_function(BenchmarkId::new("eml", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    // Tape path: T::pow(x, 2.0)
    group.bench_function(BenchmarkId::new("tape", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = T::pow(x, 2.0f64);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: sin(x) gradient — EML vs tape
// ---------------------------------------------------------------------------

fn bench_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin_grad");
    group.throughput(Throughput::Elements(ITERS as u64));

    let op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));

    // EML path
    group.bench_function(BenchmarkId::new("eml", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    // Tape path: T::sin(x)
    group.bench_function(BenchmarkId::new("tape", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = T::sin(x);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: exp(x) gradient — EML vs tape
// ---------------------------------------------------------------------------

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_grad");
    group.throughput(Throughput::Elements(ITERS as u64));

    let op = Arc::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0))));

    // EML path
    group.bench_function(BenchmarkId::new("eml", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    // Tape path: T::exp(x)
    group.bench_function(BenchmarkId::new("tape", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y = T::exp(x);
                    let dy_dx = &T::grad(&[y], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dy_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: composition — 2 * x² gradient — EML+mul vs tape
//
// EML path: eml(x²) composed with regular autograd `* 2.0`.
// Tape path: T::pow(x, 2.0) * 2.0.
// Expected gradient: 4*x.
// ---------------------------------------------------------------------------

fn bench_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("composition_grad");
    group.throughput(Throughput::Elements(ITERS as u64));

    // x² EML op — the 2.0 multiplier is applied via regular autograd
    let sq_op = Arc::new(LoweredOp::Pow(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Const(2.0)),
    ));

    // EML path: eml(x²) * 2.0, then T::grad
    group.bench_function(BenchmarkId::new("eml_then_mul", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let x_sq = ag::eml_scalar_op(Arc::clone(&sq_op), &[x], g);
                    // Compose with regular autograd multiplication
                    let h = x_sq * 2.0f64;
                    let dh_dx = &T::grad(&[h], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dh_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    // Tape path: T::pow(x, 2.0) * 2.0
    group.bench_function(BenchmarkId::new("tape_pow_mul", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let h = T::pow(x, 2.0f64) * 2.0f64;
                    let dh_dx = &T::grad(&[h], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let out = g.evaluator().push(dh_dx).feed(x, xv.view()).run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: multi-input — f(x, y) = x * y, gradient w.r.t. x
//
// EML path: Mul(Var(0), Var(1)), differentiate w.r.t. Var(0).
// Tape path: x * y, differentiate w.r.t. x.
// Expected gradient: y.
// ---------------------------------------------------------------------------

fn bench_multi_input(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_input_grad");
    group.throughput(Throughput::Elements(ITERS as u64));

    let mul_op = Arc::new(LoweredOp::Mul(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Var(1)),
    ));

    // EML path
    group.bench_function(BenchmarkId::new("eml", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                let y_val = 0.5 + i as f64 * 0.01;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y_ph = g.placeholder("y", &[]);
                    let result = ag::eml_scalar_op(Arc::clone(&mul_op), &[x, y_ph], g);
                    let d_dx = &T::grad(&[result], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let yv = arr0(y_val).into_dyn();
                    let out = g
                        .evaluator()
                        .push(d_dx)
                        .feed(x, xv.view())
                        .feed(y_ph, yv.view())
                        .run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    // Tape path: x * y_ph, differentiate w.r.t. x
    group.bench_function(BenchmarkId::new("tape", ITERS), |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..ITERS {
                let x_val = 0.1 + i as f64 * 0.02;
                let y_val = 0.5 + i as f64 * 0.01;
                ag::run(|g: &mut ag::Context<f64>| {
                    let x = g.placeholder("x", &[]);
                    let y_ph = g.placeholder("y", &[]);
                    let result = x * y_ph;
                    let d_dx = &T::grad(&[result], &[x])[0];

                    let xv = arr0(x_val).into_dyn();
                    let yv = arr0(y_val).into_dyn();
                    let out = g
                        .evaluator()
                        .push(d_dx)
                        .feed(x, xv.view())
                        .feed(y_ph, yv.view())
                        .run();
                    let grad_val = out[0]
                        .as_ref()
                        .ok()
                        .and_then(|a| a.iter().next().copied())
                        .unwrap_or(0.0);
                    sum += grad_val;
                });
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_x_squared,
    bench_sin,
    bench_exp,
    bench_composition,
    bench_multi_input
);
criterion_main!(benches);
