//! Benchmarks for `eml::eval` (real and complex interpreters).

use criterion::{criterion_group, criterion_main, Criterion};
use num_complex::Complex64;
use scirs2_symbolic::eml::eval::{eval_complex, eval_real, EvalCtx};
use scirs2_symbolic::eml::{lower, Canonical, EmlTree, LoweredOp};
use std::hint::black_box;

fn build_simple_polynomial() -> LoweredOp {
    // f(x) = x^4 + 3x^3 - 2x^2 + x - 5
    let x = LoweredOp::Var(0);
    let x2 = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    let x3 = LoweredOp::Mul(Box::new(x2.clone()), Box::new(x.clone()));
    let x4 = LoweredOp::Mul(Box::new(x2.clone()), Box::new(x2.clone()));

    LoweredOp::Sub(
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Sub(
                Box::new(LoweredOp::Add(
                    Box::new(x4),
                    Box::new(LoweredOp::Mul(
                        Box::new(LoweredOp::Const(3.0)),
                        Box::new(x3),
                    )),
                )),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Const(2.0)),
                    Box::new(x2),
                )),
            )),
            Box::new(x.clone()),
        )),
        Box::new(LoweredOp::Const(5.0)),
    )
}

fn build_transcendental() -> LoweredOp {
    // f(x) = sin(x) + cos(x²) - exp(x/2)
    let x = LoweredOp::Var(0);
    let x_sq = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    LoweredOp::Sub(
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Sin(Box::new(x.clone()))),
            Box::new(LoweredOp::Cos(Box::new(x_sq))),
        )),
        Box::new(LoweredOp::Exp(Box::new(LoweredOp::Div(
            Box::new(x),
            Box::new(LoweredOp::Const(2.0)),
        )))),
    )
}

fn bench_eval_polynomial(c: &mut Criterion) {
    let op = build_simple_polynomial();
    let ctx = EvalCtx::new(&[1.7]);
    c.bench_function("eval_real polynomial degree 4", |b| {
        b.iter(|| eval_real(black_box(&op), black_box(&ctx)))
    });
}

fn bench_eval_transcendental(c: &mut Criterion) {
    let op = build_transcendental();
    let ctx = EvalCtx::new(&[1.7]);
    c.bench_function("eval_real transcendental", |b| {
        b.iter(|| eval_real(black_box(&op), black_box(&ctx)))
    });
}

fn bench_eval_complex_path(c: &mut Criterion) {
    let op = build_transcendental();
    let vars = [Complex64::new(1.7, 0.3)];
    c.bench_function("eval_complex transcendental", |b| {
        b.iter(|| eval_complex(black_box(&op), black_box(&vars)))
    });
}

fn bench_eval_canonical_sin(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let formula = Canonical::sin(&x);
    let lowered = lower(&formula);
    let vars = [Complex64::new(0.5, 0.0)];
    c.bench_function("eval_complex Canonical::sin (543-deep tree)", |b| {
        b.iter(|| eval_complex(black_box(&lowered), black_box(&vars)))
    });
}

fn bench_batch_1000(c: &mut Criterion) {
    let op = build_transcendental();
    let xs: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();
    c.bench_function("eval_real batch 1000", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for x in &xs {
                let ctx = EvalCtx::new(std::slice::from_ref(x));
                if let Ok(v) = eval_real(black_box(&op), &ctx) {
                    sum += v;
                }
            }
            sum
        })
    });
}

criterion_group!(
    eval_benches,
    bench_eval_polynomial,
    bench_eval_transcendental,
    bench_eval_complex_path,
    bench_eval_canonical_sin,
    bench_batch_1000,
);
criterion_main!(eval_benches);
