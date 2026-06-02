//! SciRS2 v0.3.0 Time Series Performance Benchmark Suite
//!
//! Comprehensive benchmarks for time series analysis:
//! - ARIMA model fitting and forecasting
//! - SARIMA (Seasonal ARIMA) fitting
//! - Seasonal decomposition (STL, classical)
//! - Autocorrelation computation
//! - Differencing operations
//! - Rolling window operations
//!
//! Performance Targets (v0.3.0):
//! - ARIMA fitting: <1s for 1000 points
//! - SARIMA fitting: <5s for 1000 points
//! - STL decomposition: <500ms for 1000 points
//! - Forecasting: <100ms for 100 steps ahead
//! - Autocorrelation: <10ms for 1000 lags

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{ChaCha8Rng, Distribution, Normal, SeedableRng};
use scirs2_metrics::regression::{
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    root_mean_squared_error,
};
use scirs2_series::{
    arima_models::ArimaModel,
    decomposition::{decompose_seasonal, stl_decomposition, DecompositionModel, STLOptions},
    sarima_models::SarimaModel,
    utils::{autocorrelation, difference_series, moving_average, partial_autocorrelation},
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Duration;

/// Benchmark results for serialization and reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub category: String,
    pub operation: String,
    pub size: usize,
    pub mean_time_ns: f64,
    pub throughput_ops_per_sec: f64,
    pub std_dev_ns: f64,
}

/// Global benchmark results collector
static mut BENCHMARK_RESULTS: Option<Vec<BenchmarkResult>> = None;

/// Initialize results collector
fn init_results() {
    unsafe {
        BENCHMARK_RESULTS = Some(Vec::new());
    }
}

/// Save all benchmark results to JSON
fn save_results() {
    unsafe {
        if let Some(ref results) = BENCHMARK_RESULTS {
            let json = serde_json::to_string_pretty(results)
                .expect("Failed to serialize benchmark results");
            let output_path = std::env::temp_dir().join("scirs2_v030_series_results.json");
            let mut file = File::create(&output_path).expect("Failed to create results file");
            file.write_all(json.as_bytes())
                .expect("Failed to write results");
            println!(
                "\n Saved {} benchmark results to {}",
                results.len(),
                output_path.display()
            );
        }
    }
}

/// Generate synthetic time series data with trend and seasonality
fn generate_time_series(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("Failed to create normal distribution");

    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        let trend = 0.1 * (i as f64);
        let seasonal = 10.0 * ((i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin());
        let noise = normal.sample(&mut rng);
        series.push(trend + seasonal + noise);
    }

    Array1::from_vec(series)
}

/// Generate random walk time series
fn generate_random_walk(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("Failed to create normal distribution");

    let mut series = Vec::with_capacity(n);
    let mut value = 0.0;
    for _ in 0..n {
        value += normal.sample(&mut rng);
        series.push(value);
    }

    Array1::from_vec(series)
}

// =============================================================================
// ARIMA Model Benchmarks
// =============================================================================

/// Benchmark ARIMA model fitting
fn bench_arima_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("arima_fitting");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [100, 500, 1000, 2000];

    for &size in &sizes {
        let data = generate_random_walk(size, 42);

        group.throughput(Throughput::Elements(size as u64));

        // ARIMA(1,1,1)
        group.bench_with_input(BenchmarkId::new("arima_111", size), &size, |b, _| {
            b.iter(|| {
                let result = ArimaModel::<f64>::new(1, 1, 1).map(|mut model| model.fit(&data));
                black_box(result)
            })
        });

        // ARIMA(2,1,2)
        group.bench_with_input(BenchmarkId::new("arima_212", size), &size, |b, _| {
            b.iter(|| {
                let result = ArimaModel::<f64>::new(2, 1, 2).map(|mut model| model.fit(&data));
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark ARIMA forecasting
fn bench_arima_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("arima_forecasting");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let train_size = 1000;
    let forecast_steps = [10, 50, 100, 200];

    let data = generate_random_walk(train_size, 42);
    let mut model = ArimaModel::<f64>::new(2, 1, 2).expect("Failed to create ARIMA model");
    model.fit(&data).expect("Failed to fit ARIMA model");

    for &steps in &forecast_steps {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("forecast", steps), &steps, |b, &s| {
            b.iter(|| {
                let forecast = model.forecast(s, &data);
                black_box(forecast)
            })
        });
    }

    group.finish();
}

// =============================================================================
// SARIMA Model Benchmarks
// =============================================================================

/// Benchmark SARIMA model fitting
fn bench_sarima_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sarima_fitting");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));

    // Use only larger sizes to ensure sufficient data for seasonal models
    let sizes = [200, 500, 1000];

    for &size in &sizes {
        let data = generate_time_series(size, 42);

        group.throughput(Throughput::Elements(size as u64));

        // SARIMA(1,1,1)(1,1,1,12)
        group.bench_with_input(
            BenchmarkId::new("sarima_111_111_12", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = SarimaModel::<f64>::new(1, 1, 1, 1, 1, 1, 12)
                        .map(|mut model| model.fit(&data));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SARIMA forecasting
fn bench_sarima_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sarima_forecasting");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let train_size = 500;
    let forecast_steps = [12, 24, 36]; // 1, 2, 3 years for monthly data

    let data = generate_time_series(train_size, 42);
    let mut model =
        SarimaModel::<f64>::new(1, 1, 1, 1, 1, 1, 12).expect("Failed to create SARIMA model");
    model.fit(&data).expect("Failed to fit SARIMA model");

    for &steps in &forecast_steps {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("forecast", steps), &steps, |b, &s| {
            b.iter(|| {
                let forecast = model.predict(s, &data);
                black_box(forecast)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Seasonal Decomposition Benchmarks
// =============================================================================

/// Benchmark STL decomposition
fn bench_stl_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("stl_decomposition");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [100, 500, 1000, 2000];
    let period = 12;
    let options = STLOptions::default();

    for &size in &sizes {
        let data = generate_time_series(size, 42);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("stl", size), &size, |b, _| {
            b.iter(|| {
                let result = stl_decomposition(&data, period, &options);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark classical (additive/multiplicative) decomposition
fn bench_classical_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("classical_decomposition");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [100, 500, 1000, 2000, 5000];
    let period = 12;

    for &size in &sizes {
        let data = generate_time_series(size, 42);

        group.throughput(Throughput::Elements(size as u64));

        // Additive decomposition
        group.bench_with_input(BenchmarkId::new("additive", size), &size, |b, _| {
            b.iter(|| {
                let result = decompose_seasonal(&data, period, DecompositionModel::Additive);
                black_box(result)
            })
        });

        // Multiplicative decomposition
        group.bench_with_input(BenchmarkId::new("multiplicative", size), &size, |b, _| {
            b.iter(|| {
                let result = decompose_seasonal(&data, period, DecompositionModel::Multiplicative);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Autocorrelation Benchmarks
// =============================================================================

/// Benchmark autocorrelation function (ACF)
fn bench_acf_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("acf_computation");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [100, 500, 1000, 2000, 5000];

    for &size in &sizes {
        let data = generate_time_series(size, 42);
        let max_lag = (size / 4).min(100);

        group.throughput(Throughput::Elements((size * max_lag) as u64));

        group.bench_with_input(BenchmarkId::new("acf", size), &size, |b, _| {
            b.iter(|| {
                let result = autocorrelation(&data, Some(max_lag));
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark partial autocorrelation function (PACF)
fn bench_pacf_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pacf_computation");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [100, 500, 1000, 2000];

    for &size in &sizes {
        let data = generate_time_series(size, 42);
        let max_lag = (size / 4).min(100);

        group.throughput(Throughput::Elements((size * max_lag) as u64));

        group.bench_with_input(BenchmarkId::new("pacf", size), &size, |b, _| {
            b.iter(|| {
                let result = partial_autocorrelation(&data, Some(max_lag));
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Differencing Operations Benchmarks
// =============================================================================

/// Benchmark differencing operations
fn bench_differencing_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("differencing_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10_000, 100_000];

    for &size in &sizes {
        let data = generate_time_series(size, 42);

        group.throughput(Throughput::Elements(size as u64));

        // First-order differencing
        group.bench_with_input(BenchmarkId::new("difference_1", size), &size, |b, _| {
            b.iter(|| {
                let result = difference_series(&data, 1);
                black_box(result)
            })
        });

        // Second-order differencing
        group.bench_with_input(BenchmarkId::new("difference_2", size), &size, |b, _| {
            b.iter(|| {
                let result = difference_series(&data, 2);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Rolling Window Operations Benchmarks
// =============================================================================

/// Benchmark rolling window statistics (moving average)
fn bench_rolling_window_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_window_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [1000, 5000, 10_000, 50_000];
    let windows = [10, 30, 50];

    for &size in &sizes {
        let data = generate_time_series(size, 42);

        for &window in &windows {
            if window > size / 10 {
                continue;
            }

            group.throughput(Throughput::Elements(size as u64));

            // Rolling mean (moving average)
            group.bench_with_input(
                BenchmarkId::new("rolling_mean", format!("n{}_w{}", size, window)),
                &(size, window),
                |b, _| {
                    b.iter(|| {
                        let result = moving_average(&data, window);
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Forecasting Accuracy Benchmarks
// =============================================================================

/// Benchmark forecasting accuracy metrics
fn bench_forecasting_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecasting_accuracy_metrics");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 500, 1000, 5000];

    for &size in &sizes {
        let actual = generate_time_series(size, 42);
        let predicted = generate_time_series(size, 43); // Different seed

        group.throughput(Throughput::Elements(size as u64));

        // Mean Absolute Error
        group.bench_with_input(BenchmarkId::new("mae", size), &size, |b, _| {
            b.iter(|| {
                let result = mean_absolute_error(&actual, &predicted);
                black_box(result)
            })
        });

        // Mean Squared Error
        group.bench_with_input(BenchmarkId::new("mse", size), &size, |b, _| {
            b.iter(|| {
                let result = mean_squared_error(&actual, &predicted);
                black_box(result)
            })
        });

        // Root Mean Squared Error
        group.bench_with_input(BenchmarkId::new("rmse", size), &size, |b, _| {
            b.iter(|| {
                let result = root_mean_squared_error(&actual, &predicted);
                black_box(result)
            })
        });

        // Mean Absolute Percentage Error
        group.bench_with_input(BenchmarkId::new("mape", size), &size, |b, _| {
            b.iter(|| {
                let result = mean_absolute_percentage_error(&actual, &predicted);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Groups and Main
// =============================================================================

fn benchmark_startup(_: &mut Criterion) {
    init_results();
    println!("\n SciRS2 v0.3.0 Time Series Performance Benchmark Suite\n");
}

fn benchmark_teardown(_: &mut Criterion) {
    save_results();
    println!("\n Time Series Benchmarks Completed Successfully!\n");
}

criterion_group!(startup, benchmark_startup);

criterion_group!(
    arima_benchmarks,
    bench_arima_fitting,
    bench_arima_forecasting,
);

criterion_group!(
    sarima_benchmarks,
    bench_sarima_fitting,
    bench_sarima_forecasting,
);

criterion_group!(
    decomposition_benchmarks,
    bench_stl_decomposition,
    bench_classical_decomposition,
);

criterion_group!(
    autocorr_benchmarks,
    bench_acf_computation,
    bench_pacf_computation,
);

criterion_group!(differencing_benchmarks, bench_differencing_ops,);

criterion_group!(rolling_benchmarks, bench_rolling_window_ops,);

criterion_group!(accuracy_benchmarks, bench_forecasting_accuracy,);

criterion_group!(teardown, benchmark_teardown);

criterion_main!(
    startup,
    arima_benchmarks,
    sarima_benchmarks,
    decomposition_benchmarks,
    autocorr_benchmarks,
    differencing_benchmarks,
    rolling_benchmarks,
    accuracy_benchmarks,
    teardown,
);
