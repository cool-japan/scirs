# SciRS2 Datasets

[![crates.io](https://img.shields.io/crates/v/scirs2-datasets.svg)](https://crates.io/crates/scirs2-datasets)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-datasets)](https://docs.rs/scirs2-datasets)
[![Version](https://img.shields.io/badge/version-0.4.4-green)]()

A comprehensive dataset loading and generation library for the SciRS2 scientific computing ecosystem. Provides classic toy datasets, synthetic data generators, time series benchmarks, graph datasets, image datasets, anomaly detection benchmarks, financial data, medical imaging (synthetic), recommendation datasets, and more — all with a consistent, ergonomic API inspired by `scikit-learn.datasets`.

## Features

### Classic Toy Datasets
- **Iris**: 150 samples, 4 features, 3 classes (Fisher's classic)
- **Boston Housing**: 506 samples, 13 features, regression (housing prices)
- **Breast Cancer**: 569 samples, 30 features, binary classification
- **Wine**: 178 samples, 13 features, 3 classes
- **Digits**: 1797 samples, 64 features (8x8 pixel images), 10 classes
- **Diabetes**: 442 samples, 10 features, regression

### Synthetic Data Generators
- **Classification**: Linear and non-linear, configurable clusters, noise, redundant features
- **Regression**: Multi-output regression with configurable informative features
- **Clustering**: `make_blobs` (Gaussian), hierarchical cluster structures
- **Non-linear patterns**: `make_spirals`, `make_moons`, `make_circles`, `make_swiss_roll`
- **Time series**: AR/MA/ARIMA processes, seasonal, trend, noise-configurable generators
- **Imbalanced datasets**: Configurable class imbalance ratios

### Specialized Benchmark Datasets
- **Graph**: Cora (citation network), Citeseer, PROTEINS, benchmark graphs for GNN evaluation
- **Image**: MNIST-like, Fashion-MNIST-like, CIFAR-10 format (synthetic)
- **Text**: 20 Newsgroups (topics), IMDB (sentiment), NER datasets, QA datasets
- **Anomaly Detection**: KDD Cup, benchmark anomaly detection datasets
- **Financial Time Series**: Synthetic stock prices, volatility, portfolio data
- **Medical Imaging**: Synthetic MRI/CT-like volumes for algorithm testing
- **Recommendation Systems**: MovieLens-like interaction matrices, collaborative filtering benchmarks
- **Physics Simulations**: N-body dynamics, fluid simulation snapshots, wave equations
- **Knowledge Graphs**: Entity-relation triples for link prediction benchmarks

### Dataset Utilities
- **Cross-Validation**: K-fold, stratified K-fold, time series split, group K-fold
- **Train/Test Splitting**: Random and stratified splits
- **Sampling**: Random, stratified, bootstrap, importance sampling
- **Data Balancing**: Random oversampling, random undersampling, SMOTE-like
- **Feature Engineering**: Polynomial features, binning, statistical feature extraction
- **Scaling and Normalization**: Min-max, robust scaling, standard scaling, L1/L2 normalization
- **Caching**: Platform-specific disk caching with SHA256 integrity verification
- **Streaming Generators**: Infinite generators for online learning benchmarks

## Installation

```toml
[dependencies]
scirs2-datasets = "0.4.4"
```

With remote dataset download support:

```toml
[dependencies]
scirs2-datasets = { version = "0.4.4", features = ["download"] }
```

## Quick Start

### Classic Datasets

```rust
use scirs2_datasets::{load_iris, load_boston, load_digits, load_wine, load_breast_cancer, load_diabetes};

let iris    = load_iris()?;
let boston  = load_boston()?;
let digits  = load_digits()?;
let wine    = load_wine()?;
let cancer  = load_breast_cancer()?;
let diabetes = load_diabetes()?;

println!("Iris:   {} samples, {} features, {} classes",
         iris.n_samples(), iris.n_features(),
         iris.target_names.as_ref().map_or(0, |t| t.len()));
```

### Synthetic Data

```rust
use scirs2_datasets::{
    make_classification, make_regression,
    make_blobs, make_spirals, make_moons, make_circles, make_swiss_roll
};

// Classification dataset: 1000 samples, 10 features, 3 classes
let clf_data = make_classification(1000, 10, 3, 2, 4, Some(42))?;

// Regression dataset: 500 samples, 5 features, 3 informative
let reg_data = make_regression(500, 5, 3, 0.1, Some(42))?;

// Clustering: 300 samples, 4 Gaussian clusters
let blobs = make_blobs(300, 2, 4, 1.0, Some(42))?;

// Non-linear patterns
let spirals    = make_spirals(200, 2, 0.1, Some(42))?;
let moons      = make_moons(150, 0.05, Some(42))?;
let circles    = make_circles(200, 0.1, Some(42))?;
let swiss_roll = make_swiss_roll(500, 0.1, Some(42))?;
```

### Time Series

```rust
use scirs2_datasets::{make_time_series, make_arima_series, TimeSeriesBenchmark};

// Generic time series with trend and seasonality
let ts = make_time_series(1000, 24, 0.1, Some(42))?;

// Specific ARIMA process
let arima_ts = make_arima_series(500, &[0.7, -0.2], 1, &[0.4], 0.1, Some(42))?;

// Load standard benchmark datasets
let m4_daily = TimeSeriesBenchmark::load("m4_daily")?;
```

### Graph Datasets

```rust
use scirs2_datasets::{load_cora, load_citeseer, load_proteins};

let cora     = load_cora()?;        // 2708 nodes, 5429 edges, 7 classes
let citeseer = load_citeseer()?;    // 3327 nodes, 4732 edges, 6 classes
let proteins = load_proteins()?;    // 1113 graphs, graph classification

println!("Cora: {} nodes, {} edges", cora.num_nodes(), cora.num_edges());
```

### Anomaly Detection Benchmarks

```rust
use scirs2_datasets::{load_anomaly_benchmark, AnomalyBenchmark};

// KDD Cup 99 subset
let kdd = load_anomaly_benchmark(AnomalyBenchmark::KddCup)?;
println!("Anomaly ratio: {:.2}%",
         kdd.anomaly_fraction() * 100.0);

// Synthetic anomaly injection
use scirs2_datasets::make_anomaly_dataset;
let (data, labels) = make_anomaly_dataset(1000, 0.05, Some(42))?;
```

### Text Datasets

```rust
use scirs2_datasets::{load_newsgroups, load_imdb, load_ner_dataset};

let newsgroups = load_newsgroups(Some(&["sci.med", "sci.space", "comp.graphics"]))?;
let imdb       = load_imdb(Some(5000))?;   // 5000 reviews, balanced
let ner_data   = load_ner_dataset("conll2003")?;
```

### Financial Data

```rust
use scirs2_datasets::{make_financial_series, FinancialDataConfig};

let config = FinancialDataConfig {
    n_assets: 5,
    n_days: 252,      // 1 trading year
    volatility: 0.2,
    mean_return: 0.0005,
    correlation: 0.3,
    seed: Some(42),
};

let prices = make_financial_series(config)?;
println!("Shape: {:?}", prices.shape());
```

### Cross-Validation

```rust
use scirs2_datasets::{load_iris, k_fold_split, stratified_k_fold_split, train_test_split};

let iris = load_iris()?;

// Standard K-fold
let folds = k_fold_split(iris.n_samples(), 5, true, Some(42))?;
for (i, (train_idx, test_idx)) in folds.iter().enumerate() {
    println!("Fold {}: {} train, {} test", i, train_idx.len(), test_idx.len());
}

// Stratified split
if let Some(targets) = &iris.target {
    let strat_folds = stratified_k_fold_split(targets, 5, true, Some(42))?;
    let (train_idx, test_idx) = train_test_split(iris.n_samples(), 0.8, Some(42))?;
}

// Time series split (no data leakage)
let ts_folds = time_series_split(1000, 5, Some(10))?;
```

### Caching System

```rust
use scirs2_datasets::{CacheManager, DatasetCache};

let cache = CacheManager::new()?;
let stats = cache.get_statistics()?;
println!("Cache: {} datasets, {:.1} MB", stats.total_files, stats.total_size_mb);

// Clear specific dataset from cache
cache.remove("iris")?;

// Clear all cached datasets
cache.clear()?;
```

## Dataset API

All datasets implement the `Dataset<F>` trait:

```rust
pub trait Dataset<F> {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn data(&self) -> &Array2<F>;
    fn target(&self) -> Option<&Array1<F>>;
    fn feature_names(&self) -> Option<&[String]>;
    fn target_names(&self) -> Option<&[String]>;
    fn description(&self) -> &str;
}
```

## Module Map

| Module | Contents |
|--------|----------|
| `toy_datasets` | Iris, Boston, Digits, Wine, Breast Cancer, Diabetes |
| `generators` | `make_classification`, `make_regression`, `make_blobs`, non-linear patterns |
| `time_series_benchmarks` | ARIMA generators, M4/M5 format, seasonal decomposition datasets |
| `graph_datasets` | Cora, Citeseer, PROTEINS, synthetic graph generators |
| `image_datasets` | MNIST-like, CIFAR-10 format, synthetic images |
| `text_datasets` | 20 Newsgroups, IMDB, NER, QA |
| `anomaly_benchmarks` | KDD Cup, synthetic anomaly injection, detection benchmarks |
| `financial` | Synthetic asset prices, volatility, portfolio construction |
| `medical_datasets` | Synthetic MRI/CT volumes, segmentation masks |
| `recommendation_datasets` | MovieLens-like, collaborative filtering matrices |
| `graph_benchmarks` | GNN benchmark suites |
| `regression_benchmarks` | Regression performance benchmarks |
| `imbalanced` | Imbalanced classification datasets and SMOTE-like |
| `synthetic_signals` | Synthetic signal datasets for DSP algorithms |
| `physics` | N-body, fluid simulation, wave equation snapshots |
| `knowledge_graph_datasets` | Entity-relation triples for KG tasks |
| `benchmark` | Comprehensive ML algorithm benchmarks |
| `utils` | Cross-validation, train/test split, sampling, scaling |
| `cache` | Disk caching with SHA256 verification |

## Performance

- **Memory-efficient loading**: Lazy loading and memory-mapped access via `scirs2-io`
- **Fast generators**: Vectorized synthetic data generation using `scirs2-core` RNG
- **Integrity verified**: SHA256 checksums on all cached downloads
- **Cross-platform caching**: Platform-specific cache directories (XDG on Linux, Application Support on macOS, AppData on Windows)
- **Test coverage**: 584 lib tests, 100% public API coverage

## Integration

Works seamlessly with other SciRS2 crates:

```rust
use scirs2_datasets::load_iris;
use scirs2_stats::distributions::normal;
use scirs2_linalg::decomposition::pca;
use scirs2_metrics::classification::accuracy_score;

let iris = load_iris()?;
// Feed directly into scirs2-linalg, scirs2-stats, scirs2-metrics, etc.
```

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
