//! Criterion microbenchmarks for the `cas::canonicalize` + `cas::identity_db` +
//! `cas::e_graph` pipeline.
//!
//! These benchmarks characterise the performance of the EML rewriter and
//! enable regression detection in CI ("AXIOM-style" benchmark as described in
//! the Phase 2 plan).
//!
//! Five benchmark groups:
//! 1. `canonicalize` — fixed-point canonical-form pipeline on simple expressions
//! 2. `apply_standard_identity_db` — trig/hyperbolic identity rewriting
//! 3. `canonicalize_egraph` — equality-saturation canonicalization
//! 4. `cas::cse_dag` — CSE DAG `eval_all` vs 4× independent `eval_real`
//! 5. `cas::series` — Taylor polynomial and Padé rational approximant

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use scirs2_symbolic::{
    cas::{
        apply_standard_identity_db, canonicalize,
        cse_dag::CseDag,
        e_graph::{canonicalize_egraph, SaturationBudget},
        series::{pade, taylor},
    },
    eml::{eval_real, grad, EvalCtx, LoweredOp},
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helper constructors (mirror the style in eval.rs / simplify_grad.rs)
// ═══════════════════════════════════════════════════════════════════════════════

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Box::new(base), Box::new(exp))
}

fn exp_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Box::new(x))
}

fn ln_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Ln(Box::new(x))
}

fn sin_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Box::new(x))
}

fn cos_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Box::new(x))
}

fn sinh_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Sinh(Box::new(x))
}

fn cosh_op(x: LoweredOp) -> LoweredOp {
    LoweredOp::Cosh(Box::new(x))
}

fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1 — canonicalize (simple expressions)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_canonicalize(c_crit: &mut Criterion) {
    // ── 1a. x + 0 (trivial constant fold) ──────────────────────────────────
    let x_plus_0 = add(var(0), c(0.0));
    c_crit.bench_function("canonicalize x+0 trivial fold", |b| {
        b.iter(|| canonicalize(black_box(&x_plus_0)))
    });

    // ── 1b. exp(ln(x)) — inverse cancellation ───────────────────────────────
    let exp_ln_x = exp_op(ln_op(var(0)));
    c_crit.bench_function("canonicalize exp(ln(x)) inverse cancel", |b| {
        b.iter(|| canonicalize(black_box(&exp_ln_x)))
    });

    // ── 1c. ln(x*y) → ln(x) + ln(y) (log product rule) ────────────────────
    let ln_xy = ln_op(mul(var(0), var(1)));
    c_crit.bench_function("canonicalize ln(x*y) log-product rule", |b| {
        b.iter(|| canonicalize(black_box(&ln_xy)))
    });

    // ── 1d. (x^2)^3 → x^6 (power of power) ─────────────────────────────────
    let x_sq = pow(var(0), c(2.0));
    let x_sq_cubed = pow(x_sq, c(3.0));
    c_crit.bench_function("canonicalize (x^2)^3 power-of-power", |b| {
        b.iter(|| canonicalize(black_box(&x_sq_cubed)))
    });

    // ── 1e. 10-deep Add(Add(..., Const(0.0)), Const(0.0)) ───────────────────
    let mut deep_add = var(0);
    for _ in 0..10 {
        deep_add = add(deep_add, c(0.0));
    }
    c_crit.bench_function("canonicalize 10-deep Add(x,0) chain", |b| {
        b.iter(|| canonicalize(black_box(&deep_add)))
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2 — apply_standard_identity_db (trig identities)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_identity_db(c_crit: &mut Criterion) {
    // ── 2a. sin²(x) + cos²(x) — Pythagorean identity ────────────────────────
    let sin2_plus_cos2 = add(pow(sin_op(var(0)), c(2.0)), pow(cos_op(var(0)), c(2.0)));
    c_crit.bench_function("identity_db sin2+cos2 Pythagorean", |b| {
        b.iter(|| apply_standard_identity_db(black_box(&sin2_plus_cos2)))
    });

    // ── 2b. tan(x) → sin(x)/cos(x) expansion ───────────────────────────────
    let tan_x = LoweredOp::Tan(Box::new(var(0)));
    c_crit.bench_function("identity_db tan(x) expansion", |b| {
        b.iter(|| apply_standard_identity_db(black_box(&tan_x)))
    });

    // ── 2c. cosh²(x) - sinh²(x) → 1 ────────────────────────────────────────
    let cosh2_minus_sinh2 = sub(pow(cosh_op(var(0)), c(2.0)), pow(sinh_op(var(0)), c(2.0)));
    c_crit.bench_function("identity_db cosh2-sinh2 hyperbolic", |b| {
        b.iter(|| apply_standard_identity_db(black_box(&cosh2_minus_sinh2)))
    });

    // ── 2d. Deep nested trig expression ─────────────────────────────────────
    // sin(cos(sin(cos(x)))) — 4 levels of nesting
    let nested_trig = sin_op(cos_op(sin_op(cos_op(var(0)))));
    c_crit.bench_function("identity_db deep nested trig sin(cos(sin(cos(x))))", |b| {
        b.iter(|| apply_standard_identity_db(black_box(&nested_trig)))
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3 — canonicalize_egraph (equality saturation)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_egraph(c_crit: &mut Criterion) {
    // Use a modest budget for all e-graph benches so CI wall-time stays bounded.
    // The default (30 iters / 10,000 nodes) is fine for simple ops but can
    // blow out on distributive expressions; we cap at 15 iters / 5,000 nodes.
    let budget = SaturationBudget {
        max_iterations: 15,
        max_nodes: 5_000,
    };

    // ── 3a. x + 0 (trivial rule, should be fast) ────────────────────────────
    let x_plus_0 = add(var(0), c(0.0));
    let b1 = budget.clone();
    c_crit.bench_function("egraph x+0 trivial", |b| {
        b.iter(|| canonicalize_egraph(black_box(&x_plus_0), Some(b1.clone())))
    });

    // ── 3b. x * 1 + 0 (two rules) ───────────────────────────────────────────
    let x_times_1_plus_0 = add(mul(var(0), c(1.0)), c(0.0));
    let b2 = budget.clone();
    c_crit.bench_function("egraph x*1+0 two rules", |b| {
        b.iter(|| canonicalize_egraph(black_box(&x_times_1_plus_0), Some(b2.clone())))
    });

    // ── 3c. (x + y) * (a + b) with distribution rules ───────────────────────
    // Uses a smaller budget because distribution generates many intermediate classes.
    let foil_budget = SaturationBudget {
        max_iterations: 8,
        max_nodes: 2_000,
    };
    let foil_op = mul(add(var(0), var(1)), add(var(2), var(3)));
    c_crit.bench_function("egraph (x+y)*(a+b) FOIL distribution", |b| {
        b.iter(|| canonicalize_egraph(black_box(&foil_op), Some(foil_budget.clone())))
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4 — cas::cse_dag  (CSE eval_all vs 4× independent eval_real)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_cse_dag(c_crit: &mut Criterion) {
    // f(x, y) = x² + y²
    let f = add(pow(var(0), c(2.0)), pow(var(1), c(2.0)));
    let gx = grad(&f, 0); // 2x
    let gy = grad(&f, 1); // 2y
    let hxx = grad(&gx, 0); // 2

    let point = [1.5_f64, 2.5_f64];

    // ── Build CSE DAG once (setup outside hot loop) ──────────────────────────
    let mut dag = CseDag::new();
    let f_key = dag.add(&f);
    let gx_key = dag.add(&gx);
    let gy_key = dag.add(&gy);
    let hxx_key = dag.add(&hxx);

    // ── 4a. CSE eval_all — evaluates all 4 expressions at once, sharing work ─
    c_crit.bench_function("cse_dag eval_all f+grad+hessian", |b| {
        b.iter(|| {
            let vals = dag
                .eval_all(black_box(&point))
                .expect("cse_dag eval_all failed");
            // Consume all four results so the compiler cannot eliminate them.
            let _ = (vals[&f_key], vals[&gx_key], vals[&gy_key], vals[&hxx_key]);
        })
    });

    // ── 4b. 4× independent eval_real — naive baseline ───────────────────────
    let ctx = EvalCtx::new(&point);
    c_crit.bench_function("eval_real 4x independent f+grad+hessian", |b| {
        b.iter(|| {
            let v0 = eval_real(black_box(&f), &ctx).expect("eval f");
            let v1 = eval_real(black_box(&gx), &ctx).expect("eval gx");
            let v2 = eval_real(black_box(&gy), &ctx).expect("eval gy");
            let v3 = eval_real(black_box(&hxx), &ctx).expect("eval hxx");
            let _ = (v0, v1, v2, v3);
        })
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5 — cas::series (Taylor polynomial and Padé approximant)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_series(c_crit: &mut Criterion) {
    // ── 5a. taylor(exp(x), 0, 0.0, 4) — 4th order Taylor for exp ────────────
    let exp_x = exp_op(var(0));
    c_crit.bench_function("series taylor(exp(x), order=4)", |b| {
        b.iter(|| taylor(black_box(&exp_x), 0, 0.0, 4).expect("taylor exp failed"))
    });

    // ── 5b. pade(sin(x), 0, 0.0, 3, 2) — Padé [3/2] for sin ────────────────
    let sin_x = sin_op(var(0));
    c_crit.bench_function("series pade(sin(x), [3/2])", |b| {
        b.iter(|| pade(black_box(&sin_x), 0, 0.0, 3, 2).expect("pade sin failed"))
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wire-up
// ═══════════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_canonicalize,
    bench_identity_db,
    bench_egraph,
    bench_cse_dag,
    bench_series,
);
criterion_main!(benches);
