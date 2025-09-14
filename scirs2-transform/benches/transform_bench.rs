//! Benchmarks for scirs2-transform comparing with scikit-learn performance
//!
//! Run with: cargo bench --bench transform_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array2, ArrayBase, Axis, Data};
use ndarray_rand::rand::distributions::Uniform;
use ndarray_rand::RandomExt;
use scirs2_transform::*;
use std::hint::black_box;

const SAMPLE_SIZES: &[usize] = &[100, 1000, 10_000];
const FEATURE_SIZES: &[usize] = &[10, 50, 100];

/// Benchmark normalization operations
#[allow(dead_code)]
fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Normalization");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in FEATURE_SIZES {
            let data = Array2::random((n_samples, n_features), Uniform::new(-100.0, 100.0));

            // MinMax normalization
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new("MinMax", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            normalize_array(black_box(data), NormalizationMethod::MinMax, 0);
                    });
                },
            );

            // Z-score normalization
            group.bench_with_input(
                BenchmarkId::new("ZScore", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            normalize_array(black_box(data), NormalizationMethod::ZScore, 0);
                    });
                },
            );

            // L2 normalization
            group.bench_with_input(
                BenchmarkId::new("L2", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result = normalize_array(black_box(data), NormalizationMethod::L2, 1);
                    });
                },
            );

            // Robust normalization
            group.bench_with_input(
                BenchmarkId::new("Robust", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            normalize_array(black_box(data), NormalizationMethod::Robust, 0);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SIMD-accelerated normalization operations
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn bench_simd_normalization(c: &mut Criterion) {
    use scirs2_transform::normalize_simd::*;

    let mut group = c.benchmark_group("SIMD_Normalization");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in FEATURE_SIZES {
            let data = Array2::random((n_samples, n_features), Uniform::new(-100.0, 100.0));

            // SIMD MinMax normalization
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new("SIMD_MinMax", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            simd_normalize_array(black_box(data), NormalizationMethod::MinMax, 0);
                    });
                },
            );

            // SIMD Z-score normalization
            group.bench_with_input(
                BenchmarkId::new("SIMD_ZScore", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            simd_normalize_array(black_box(data), NormalizationMethod::ZScore, 0);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark scaling operations
#[allow(dead_code)]
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in FEATURE_SIZES {
            let data = Array2::random((n_samples, n_features), Uniform::new(-100.0, 100.0));

            // MaxAbsScaler
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new("MaxAbsScaler", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    let mut scaler = MaxAbsScaler::new();
                    scaler.fit(data).unwrap();

                    b.iter(|| {
                        let _result = scaler.transform(black_box(data));
                    });
                },
            );

            // QuantileTransformer
            group.bench_with_input(
                BenchmarkId::new(
                    "QuantileTransformer",
                    format!("{}x{}", n_samples, n_features),
                ),
                &data,
                |b, data| {
                    let mut transformer = QuantileTransformer::new(100, "uniform", false).unwrap();
                    transformer.fit(data).unwrap();

                    b.iter(|| {
                        let _result = transformer.transform(black_box(data));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark feature engineering operations
#[allow(dead_code)]
fn bench_feature_engineering(c: &mut Criterion) {
    let mut group = c.benchmark_group("Feature_Engineering");

    for &n_samples in &[100, 1000] {
        // Smaller sizes for polynomial features
        for &n_features in &[5, 10, 20] {
            // Fewer features due to polynomial expansion
            let data = Array2::random((n_samples, n_features), Uniform::new(-10.0, 10.0));

            // Polynomial features (degree 2)
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "PolynomialFeatures",
                    format!("{}x{}", n_samples, n_features),
                ),
                &data,
                |b, data| {
                    let poly = PolynomialFeatures::new(2, false, false);

                    b.iter(|| {
                        let _result = poly.transform(black_box(data));
                    });
                },
            );

            // Power transformation
            group.bench_with_input(
                BenchmarkId::new("PowerTransform", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    let mut pt = PowerTransformer::new("yeo-johnson", true).unwrap();
                    pt.fit(data).unwrap();

                    b.iter(|| {
                        let _result = pt.transform(black_box(data));
                    });
                },
            );

            // Binarization
            group.bench_with_input(
                BenchmarkId::new("Binarize", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result = binarize(black_box(data), 0.0);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark dimensionality reduction operations
#[allow(dead_code)]
fn bench_dimensionality_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dimensionality_Reduction");

    for &n_samples in &[100, 500] {
        // Smaller sizes for expensive operations
        for &n_features in &[20, 50] {
            let data = Array2::random((n_samples, n_features), Uniform::new(-10.0, 10.0));
            let n_components = n_features / 2;

            // PCA
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new("PCA", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    let mut pca = PCA::new(n_components, true, false);
                    pca.fit(data).unwrap();

                    b.iter(|| {
                        let _result = pca.transform(black_box(data));
                    });
                },
            );

            // TruncatedSVD
            group.bench_with_input(
                BenchmarkId::new("TruncatedSVD", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    let mut svd = TruncatedSVD::new(n_components);
                    svd.fit(data).unwrap();

                    b.iter(|| {
                        let _result = svd.transform(black_box(data));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark imputation operations
#[allow(dead_code)]
fn bench_imputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Imputation");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in &[10, 20] {
            // Smaller feature sizes for imputation
            // Create data with missing values
            let mut data = Array2::random((n_samples, n_features), Uniform::new(-10.0, 10.0));
            // Insert NaN values randomly (about 10%)
            for i in 0..n_samples {
                for j in 0..n_features {
                    if rand::random::<f64>() < 0.1 {
                        data[[i, j]] = f64::NAN;
                    }
                }
            }

            // SimpleImputer
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new("SimpleImputer", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean, f64::NAN);
                    imputer.fit(data).unwrap();

                    b.iter(|| {
                        let _result = imputer.transform(black_box(data));
                    });
                },
            );

            // KNNImputer (only for smaller datasets)
            if n_samples <= 1000 {
                group.bench_with_input(
                    BenchmarkId::new("KNNImputer", format!("{}x{}", n_samples, n_features)),
                    &data,
                    |b, data| {
                        let mut imputer = KNNImputer::new(
                            5,
                            DistanceMetric::Euclidean,
                            WeightingScheme::Distance,
                            f64::NAN,
                        );
                        imputer.fit(data).unwrap();

                        b.iter(|| {
                            let _result = imputer.transform(black_box(data));
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark pipeline operations
#[allow(dead_code)]
fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pipeline");

    for &n_samples in &[100, 1000] {
        for &n_features in &[10, 20] {
            let data = Array2::random((n_samples, n_features), Uniform::new(-10.0, 10.0));

            // Simple pipeline: StandardScaler -> PCA
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "StandardScaler_PCA",
                    format!("{}x{}", n_samples, n_features),
                ),
                &data,
                |b, data| {
                    // Create pipeline
                    let normalizer = Normalizer::new(NormalizationMethod::ZScore, 0);
                    let pca = PCA::new(n_features / 2, true, false);

                    // Note: Since we don't have the adapter implementations compiled,
                    // we'll benchmark the operations separately
                    b.iter(|| {
                        let mut norm = normalizer.clone();
                        let normalized = norm.fit_transform(black_box(data)).unwrap();

                        let mut pca_copy = pca.clone();
                        let _result = pca_copy.fit_transform(&normalized);
                    });
                },
            );
        }
    }

    group.finish();
}

// Configure criterion and add benchmark groups
criterion_group!(
    benches,
    bench_normalization,
    bench_scaling,
    bench_feature_engineering,
    bench_dimensionality_reduction,
    bench_imputation,
    bench_pipeline
);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_simd_normalization);

#[cfg(not(feature = "simd"))]
criterion_main!(benches);

#[cfg(feature = "simd")]
criterion_main!(benches, simd_benches);
