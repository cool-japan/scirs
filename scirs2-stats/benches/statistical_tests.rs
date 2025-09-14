//! Benchmarks for statistical tests
//!
//! This benchmark suite measures the performance of various statistical tests
//! in scirs2-stats across different sample sizes and data conditions.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use scirs2_stats::tests::ttest::Alternative;
use scirs2_stats::{
    anderson_darling,
    dagostino_k2,
    friedman,
    kendalltau,
    kruskal_wallis,
    // Non-parametric tests
    mann_whitney,
    // Correlation functions
    pearsonr,
    // Normality tests
    shapiro_wilk,
    spearmanr,
    // T-tests
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    wilcoxon,
};
use std::hint::black_box;

/// Generate random normal data
#[allow(dead_code)]
fn generate_normaldata(n: usize, mean: f64, std: f64) -> Array1<f64> {
    let mut rng = rand::rng();
    let normal = Normal::new(mean, std).unwrap();
    Array1::from_shape_fn(n, |_| normal.sample(&mut rng))
}

/// Benchmark t-tests
#[allow(dead_code)]
fn bench_t_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("t_tests");

    let samplesizes = vec![10, 50, 100, 500, 1000];

    for &n in &samplesizes {
        // One-sample t-test
        let data = generate_normaldata(n, 5.0, 1.0);
        group.bench_with_input(BenchmarkId::new("one_sample", n), &data, |b, data| {
            b.iter(|| {
                black_box(ttest_1samp(
                    &data.view(),
                    5.0,
                    Alternative::TwoSided,
                    "propagate",
                ));
            });
        });

        // Independent samples t-test
        let data1 = generate_normaldata(n, 5.0, 1.0);
        let data2 = generate_normaldata(n, 5.5, 1.0);
        group.bench_with_input(
            BenchmarkId::new("independent", n),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(ttest_ind(
                        &data1.view(),
                        &data2.view(),
                        true,
                        Alternative::TwoSided,
                        "propagate",
                    ));
                });
            },
        );

        // Welch's t-test (unequal variances)
        group.bench_with_input(
            BenchmarkId::new("welch", n),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(ttest_ind(
                        &data1.view(),
                        &data2.view(),
                        false,
                        Alternative::TwoSided,
                        "propagate",
                    ));
                });
            },
        );

        // Paired t-test
        group.bench_with_input(
            BenchmarkId::new("paired", n),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(ttest_rel(
                        &data1.view(),
                        &data2.view(),
                        Alternative::TwoSided,
                        "propagate",
                    ));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark non-parametric tests
#[allow(dead_code)]
fn bench_nonparametric_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonparametric_tests");

    let samplesizes = vec![10, 50, 100, 500];

    for &n in &samplesizes {
        let data1 = generate_normaldata(n, 5.0, 1.0);
        let data2 = generate_normaldata(n, 5.5, 1.0);

        // Mann-Whitney U test
        group.bench_with_input(
            BenchmarkId::new("mann_whitney", n),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(mann_whitney(
                        &data1.view(),
                        &data2.view(),
                        "two-sided",
                        false,
                    ));
                });
            },
        );

        // Wilcoxon signed-rank test
        group.bench_with_input(BenchmarkId::new("wilcoxon", n), &data1, |b, data| {
            b.iter(|| {
                black_box(wilcoxon(
                    &data.view(),
                    &Array1::zeros(data.len()).view(),
                    "wilcox",
                    false,
                ));
            });
        });
    }

    // Kruskal-Wallis test with multiple groups
    let groups = vec![
        generate_normaldata(30, 5.0, 1.0),
        generate_normaldata(30, 5.5, 1.0),
        generate_normaldata(30, 6.0, 1.0),
    ];

    group.bench_function("kruskal_wallis_3groups", |b| {
        b.iter(|| {
            black_box(kruskal_wallis(
                &groups.iter().map(|g| g.view()).collect::<Vec<_>>(),
            ));
        });
    });

    group.finish();
}

/// Benchmark normality tests
#[allow(dead_code)]
fn bench_normality_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("normality_tests");

    let samplesizes = vec![20, 50, 100, 200];

    for &n in &samplesizes {
        let normaldata = generate_normaldata(n, 0.0, 1.0);

        // Shapiro-Wilk test (limited to smaller samples)
        if n <= 50 {
            group.bench_with_input(
                BenchmarkId::new("shapiro_wilk", n),
                &normaldata,
                |b, data| {
                    b.iter(|| {
                        black_box(shapiro_wilk(&data.view()));
                    });
                },
            );
        }

        // Anderson-Darling test
        group.bench_with_input(
            BenchmarkId::new("anderson_darling", n),
            &normaldata,
            |b, data| {
                b.iter(|| {
                    black_box(anderson_darling(&data.view()));
                });
            },
        );

        // D'Agostino K² test
        group.bench_with_input(
            BenchmarkId::new("dagostino_k2", n),
            &normaldata,
            |b, data| {
                b.iter(|| {
                    black_box(dagostino_k2(&data.view()));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark correlation functions
#[allow(dead_code)]
fn bench_correlations(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlations");

    let samplesizes = vec![10, 50, 100, 500, 1000];

    for &n in &samplesizes {
        // Generate correlated data
        let mut rng = rand::rng();
        let x: Array1<f64> = Array1::from_shape_fn(n, |_| StandardNormal.sample(&mut rng));
        let noise: Array1<f64> = Array1::from_shape_fn(n, |_| StandardNormal.sample(&mut rng));
        let y = &x * 0.8 + &noise * 0.2; // Correlation ~ 0.8

        // Pearson correlation
        group.bench_with_input(
            BenchmarkId::new("pearson", n),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(pearsonr(&x.view(), &y.view(), "two-sided"));
                });
            },
        );

        // Spearman correlation
        group.bench_with_input(
            BenchmarkId::new("spearman", n),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(spearmanr(&x.view(), &y.view(), "two-sided"));
                });
            },
        );

        // Kendall tau (more expensive, limit size)
        if n <= 100 {
            group.bench_with_input(
                BenchmarkId::new("kendall", n),
                &(x.clone(), y.clone()),
                |b, (x, y)| {
                    b.iter(|| {
                        black_box(kendalltau(&x.view(), &y.view(), "b", "two-sided"));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark ANOVA tests
#[allow(dead_code)]
fn bench_anova(c: &mut Criterion) {
    let mut group = c.benchmark_group("anova");

    // Test different numbers of groups and sample sizes
    let configurations = vec![
        (3, 20),  // 3 groups, 20 samples each
        (3, 50),  // 3 groups, 50 samples each
        (5, 20),  // 5 groups, 20 samples each
        (5, 50),  // 5 groups, 50 samples each
        (10, 20), // 10 groups, 20 samples each
    ];

    use scirs2_stats::one_way_anova;

    for (num_groups, samples_per_group) in configurations {
        let groups: Vec<Array1<f64>> = (0..num_groups)
            .map(|i| generate_normaldata(samples_per_group, i as f64 * 0.5, 1.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("one_way", format!("{}x{}", num_groups, samples_per_group)),
            &groups,
            |b, groups| {
                b.iter(|| {
                    let group_views: Vec<_> = groups.iter().map(|g| g.view()).collect();
                    let group_refs: Vec<&_> = group_views.iter().collect();
                    black_box(one_way_anova(&group_refs));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_t_tests,
    bench_nonparametric_tests,
    bench_normality_tests,
    bench_correlations,
    bench_anova
);
criterion_main!(benches);
