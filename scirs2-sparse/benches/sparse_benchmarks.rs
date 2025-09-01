use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use rand::Rng;
use scirs2_sparse::*;
use std::hint::black_box;

#[allow(dead_code)]
fn generate_sparse_matrix(size: usize, density: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = rand::rng();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..size {
        for j in 0..size {
            if rng.random::<f64>() < density {
                rows.push(i);
                cols.push(j);
                data.push(rng.random::<f64>());
            }
        }
    }

    (rows, cols, data)
}

#[allow(dead_code)]
fn bench_sparse_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_construction");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("csr_from_triplets", size),
            size,
            |b, &size| {
                b.iter(|| {
                    CsrArray::from_triplets(
                        black_box(&rows),
                        black_box(&cols),
                        black_box(&data),
                        black_box(shape),
                        false,
                    )
                    .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("csc_from_triplets", size),
            size,
            |b, &size| {
                b.iter(|| {
                    CscArray::from_triplets(
                        black_box(&rows),
                        black_box(&cols),
                        black_box(&data),
                        black_box(shape),
                        false,
                    )
                    .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("coo_from_triplets", size),
            size,
            |b, &size| {
                b.iter(|| {
                    CooArray::from_triplets(
                        black_box(&rows),
                        black_box(&cols),
                        black_box(&data),
                        black_box(shape),
                        false,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_sparse_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_operations");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));

        group.throughput(Throughput::Elements(data.len() as u64));

        // Matrix-vector multiplication
        group.bench_with_input(BenchmarkId::new("csr_matvec", size), size, |b, &size| {
            b.iter(|| csr.dot_vector(&vector.view()).unwrap())
        });

        // Transpose operation
        group.bench_with_input(BenchmarkId::new("csr_transpose", size), size, |b, &size| {
            b.iter(|| csr.transpose().unwrap())
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_sparse_sparse_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_sparse_ops");

    for size in [100, 500].iter() {
        let (rows1, cols1, data1) = generate_sparse_matrix(*size, 0.02);
        let (rows2, cols2, data2) = generate_sparse_matrix(*size, 0.02);
        let shape = (*size, *size);

        let csr1 = CsrArray::from_triplets(&rows1, &cols1, &data1, shape, false).unwrap();
        let csr2 = CsrArray::from_triplets(&rows2, &cols2, &data2, shape, false).unwrap();

        group.throughput(Throughput::Elements((data1.len() + data2.len()) as u64));

        // Matrix-matrix multiplication
        group.bench_with_input(BenchmarkId::new("csr_matmul", size), size, |b, &size| {
            b.iter(|| black_box(&csr1).dot(black_box(&csr2)).unwrap())
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_format_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversions");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(BenchmarkId::new("coo_to_csr", size), size, |b, &size| {
            b.iter(|| black_box(&coo).to_csr().unwrap())
        });

        group.bench_with_input(BenchmarkId::new("csr_to_csc", size), size, |b, &size| {
            b.iter(|| black_box(&csr).to_csc().unwrap())
        });

        group.bench_with_input(BenchmarkId::new("csc_to_csr", size), size, |b, &size| {
            b.iter(|| black_box(&csc).to_csr().unwrap())
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_linear_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_solvers");

    for size in [50, 100].iter() {
        // Create a diagonally dominant matrix for guaranteed convergence
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        // Add strong diagonal
        for i in 0..*size {
            rows.push(i);
            cols.push(i);
            data.push(10.0);
        }

        // Add some off-diagonal elements
        let mut rng = rand::rng();
        for i in 0..*size {
            for j in (i + 1)..*size {
                if rng.random::<f64>() < 0.02 {
                    let val = rng.random::<f64>() * 0.5;
                    rows.push(i);
                    cols.push(j);
                    data.push(val);
                    rows.push(j);
                    cols.push(i);
                    data.push(val);
                }
            }
        }

        let shape = (*size, *size);
        let _matrix = CsrArray::from_triplets(&rows, &cols, &data, shape, true).unwrap();
        let _rhs = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("conjugate_gradient", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let _options = CGOptions {
                        max_iter: 100,
                        rtol: 1e-6,
                        atol: 1e-12,
                        x0: None,
                        preconditioner: None,
                    };
                    // Skip CG for now - needs LinearOperator trait implementation
                    black_box(0.0)
                })
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_symmetric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_operations");

    for size in [100, 500, 1000].iter() {
        // Create symmetric matrix data (lower triangular only)
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        let mut rng = rand::rng();

        for i in 0..*size {
            for j in 0..=i {
                if rng.random::<f64>() < 0.05 {
                    rows.push(i);
                    cols.push(j);
                    data.push(rng.random::<f64>());
                }
            }
        }

        let shape = (*size, *size);
        let csr_temp = CsrArray::from_triplets(&rows, &cols, &data, shape, true).unwrap();
        let sym_csr = SymCsrArray::from_csr_array(&csr_temp).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));

        group.throughput(Throughput::Elements(data.len() as u64));

        // Symmetric matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("sym_csr_matvec", size),
            size,
            |b, &size| b.iter(|| sym_csr.dot_vector(&vector.view()).unwrap()),
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_special_formats(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_formats");

    for size in [100, 500, 1000].iter() {
        // Create DIA format data
        let offsets = vec![-3, -1, 0, 1, 3];
        let mut diagonals = Vec::new();
        for _ in offsets.iter() {
            let diag = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));
            diagonals.push(diag);
        }

        let dia = DiaArray::new(diagonals, offsets.clone(), (*size, *size)).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));

        group.throughput(Throughput::Elements(*size as u64));

        // DIA matrix-vector multiplication
        group.bench_with_input(BenchmarkId::new("dia_matvec", size), size, |b, &size| {
            b.iter(|| dia.dot_vector(&vector.view()).unwrap())
        });

        // DIA to CSR conversion
        group.bench_with_input(BenchmarkId::new("dia_to_csr", size), size, |b, &size| {
            b.iter(|| black_box(&dia).to_csr().unwrap())
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_dok_lil_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dok_lil_operations");

    for size in [100, 500].iter() {
        let shape = (*size, *size);

        // DOK operations
        group.bench_with_input(
            BenchmarkId::new("dok_construction", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut dok = DokArray::new(shape);
                    for i in 0..size {
                        for j in 0..size {
                            if (i + j) % 20 == 0 {
                                dok.set(i, j, (i + j) as f64).unwrap();
                            }
                        }
                    }
                    black_box(dok)
                })
            },
        );

        // LIL operations
        group.bench_with_input(
            BenchmarkId::new("lil_construction", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut lil = LilArray::new(shape);
                    for i in 0..size {
                        for j in 0..size {
                            if (i + j) % 20 == 0 {
                                lil.set(i, j, (i + j) as f64).unwrap();
                            }
                        }
                    }
                    black_box(lil)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_construction,
    bench_sparse_operations,
    bench_sparse_sparse_ops,
    bench_format_conversions,
    bench_linear_solvers,
    bench_symmetric_operations,
    bench_special_formats,
    bench_dok_lil_operations
);
criterion_main!(benches);
