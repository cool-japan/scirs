//! JIT vs interpreter performance comparison (the headline performance bench).

#![cfg(feature = "jit")]

use criterion::{criterion_group, criterion_main, Criterion};
use scirs2_symbolic::compile::to_jit;
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::LoweredOp;
use std::hint::black_box;

fn build_workload() -> LoweredOp {
    // f(x, y) = sin(x²) + cos(x*y) - exp(y/2)
    let x = LoweredOp::Var(0);
    let y = LoweredOp::Var(1);
    let x_sq = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    let xy = LoweredOp::Mul(Box::new(x), Box::new(y.clone()));
    let y_half = LoweredOp::Div(Box::new(y), Box::new(LoweredOp::Const(2.0)));

    LoweredOp::Sub(
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Sin(Box::new(x_sq))),
            Box::new(LoweredOp::Cos(Box::new(xy))),
        )),
        Box::new(LoweredOp::Exp(Box::new(y_half))),
    )
}

fn bench_interpreter_single(c: &mut Criterion) {
    let op = build_workload();
    let ctx = EvalCtx::new(&[1.5, 0.7]);
    c.bench_function("eval_real (interpreter) single eval", |b| {
        b.iter(|| eval_real(black_box(&op), black_box(&ctx)))
    });
}

fn bench_jit_compile(c: &mut Criterion) {
    let op = build_workload();
    c.bench_function("compile::to_jit one-time cost", |b| {
        b.iter(|| {
            let _f = to_jit(black_box(&op)).expect("compile");
        })
    });
}

fn bench_jit_eval_after_compile(c: &mut Criterion) {
    let op = build_workload();
    let func = to_jit(&op).expect("compile");
    c.bench_function("JIT eval (post-compile)", |b| {
        b.iter(|| black_box(func.eval(black_box(&[1.5, 0.7]))))
    });
}

fn bench_batch_interpreter(c: &mut Criterion) {
    let op = build_workload();
    let inputs: Vec<[f64; 2]> = (0..1000)
        .map(|i| [(i as f64) * 0.001, (i as f64) * 0.0007])
        .collect();
    c.bench_function("eval_real batch 1000 (interpreter)", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for v in &inputs {
                let ctx = EvalCtx::new(v);
                if let Ok(r) = eval_real(black_box(&op), &ctx) {
                    sum += r;
                }
            }
            sum
        })
    });
}

fn bench_batch_jit(c: &mut Criterion) {
    let op = build_workload();
    let func = to_jit(&op).expect("compile");
    let inputs: Vec<[f64; 2]> = (0..1000)
        .map(|i| [(i as f64) * 0.001, (i as f64) * 0.0007])
        .collect();
    c.bench_function("JIT batch 1000 (post-compile)", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for v in &inputs {
                sum += black_box(func.eval(black_box(v)));
            }
            sum
        })
    });
}

criterion_group!(
    jit_vs_interp_benches,
    bench_interpreter_single,
    bench_jit_compile,
    bench_jit_eval_after_compile,
    bench_batch_interpreter,
    bench_batch_jit,
);
criterion_main!(jit_vs_interp_benches);
