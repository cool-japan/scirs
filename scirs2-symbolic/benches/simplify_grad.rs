//! Benchmarks for simplify and grad.

use criterion::{criterion_group, criterion_main, Criterion};
use scirs2_symbolic::eml::{grad, simplify_op, LoweredOp};
use std::hint::black_box;

fn build_redundant() -> LoweredOp {
    // f = (x*1 + 0) * exp(ln(x)) — simplifies to x*x = x²
    let x = LoweredOp::Var(0);
    let x_id = LoweredOp::Add(
        Box::new(LoweredOp::Mul(
            Box::new(x.clone()),
            Box::new(LoweredOp::Const(1.0)),
        )),
        Box::new(LoweredOp::Const(0.0)),
    );
    let exp_ln_x = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(x))));
    LoweredOp::Mul(Box::new(x_id), Box::new(exp_ln_x))
}

fn build_complicated() -> LoweredOp {
    // f = (x + y)² - 2*x*y, simplifies to x² + y²
    let x = LoweredOp::Var(0);
    let y = LoweredOp::Var(1);
    let sum = LoweredOp::Add(Box::new(x.clone()), Box::new(y.clone()));
    let sum_sq = LoweredOp::Mul(Box::new(sum.clone()), Box::new(sum));
    let xy = LoweredOp::Mul(Box::new(x), Box::new(y));
    let two_xy = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(xy));
    LoweredOp::Sub(Box::new(sum_sq), Box::new(two_xy))
}

fn bench_simplify_redundant(c: &mut Criterion) {
    let op = build_redundant();
    c.bench_function("simplify_op redundant chain", |b| {
        b.iter(|| simplify_op(black_box(&op)))
    });
}

fn bench_simplify_complicated(c: &mut Criterion) {
    let op = build_complicated();
    c.bench_function("simplify_op (x+y)^2 - 2xy", |b| {
        b.iter(|| simplify_op(black_box(&op)))
    });
}

fn bench_grad_polynomial(c: &mut Criterion) {
    // f = x^5 + 3x^4 - 2x^3 + x^2 - x + 7
    let x = LoweredOp::Var(0);
    let x2 = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    let x3 = LoweredOp::Mul(Box::new(x2.clone()), Box::new(x.clone()));
    let x4 = LoweredOp::Mul(Box::new(x2.clone()), Box::new(x2.clone()));
    let x5 = LoweredOp::Mul(Box::new(x4.clone()), Box::new(x.clone()));
    let f = LoweredOp::Add(
        Box::new(x5),
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Const(3.0)),
            Box::new(x4),
        )),
    );
    let f = LoweredOp::Sub(
        Box::new(f),
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Const(2.0)),
            Box::new(x3),
        )),
    );
    let f = LoweredOp::Add(Box::new(f), Box::new(x2));
    let f = LoweredOp::Sub(Box::new(f), Box::new(x));
    let f = LoweredOp::Add(Box::new(f), Box::new(LoweredOp::Const(7.0)));

    c.bench_function("grad polynomial degree 5", |b| {
        b.iter(|| grad(black_box(&f), 0))
    });
}

fn bench_grad_chain(c: &mut Criterion) {
    // f = sin(exp(x²))  → chain rule depth 3
    let x = LoweredOp::Var(0);
    let x_sq = LoweredOp::Mul(Box::new(x.clone()), Box::new(x));
    let exp_xsq = LoweredOp::Exp(Box::new(x_sq));
    let f = LoweredOp::Sin(Box::new(exp_xsq));
    c.bench_function("grad sin(exp(x^2)) chain rule", |b| {
        b.iter(|| grad(black_box(&f), 0))
    });
}

criterion_group!(
    simplify_grad_benches,
    bench_simplify_redundant,
    bench_simplify_complicated,
    bench_grad_polynomial,
    bench_grad_chain,
);
criterion_main!(simplify_grad_benches);
