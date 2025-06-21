use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::array;
use scirs2_linalg::norm::matrix_norm;
use scirs2_linalg::{det, inv};

fn bench_det(c: &mut Criterion) {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    c.bench_function("determinant 2x2", |b| {
        b.iter(|| det(black_box(&a.view()), None))
    });
}

fn bench_inv(c: &mut Criterion) {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    c.bench_function("inverse 2x2", |b| {
        b.iter(|| inv(black_box(&a.view()), None))
    });
}

fn bench_norm(c: &mut Criterion) {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    c.bench_function("frobenius norm 2x2", |b| {
        b.iter(|| matrix_norm(black_box(&a.view()), black_box("frobenius")))
    });
}

criterion_group!(benches, bench_det, bench_inv, bench_norm);
criterion_main!(benches);
