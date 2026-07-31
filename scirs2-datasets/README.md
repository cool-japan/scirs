# SciRS2 Datasets

[![crates.io](https://img.shields.io/crates/v/scirs2-datasets.svg)](https://crates.io/crates/scirs2-datasets)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-datasets)](https://docs.rs/scirs2-datasets)
[![Version](https://img.shields.io/badge/version-0.6.6-green)]()
[![Status](https://img.shields.io/badge/status-partial-yellow)]()

A dataset loading and generation library for the SciRS2 scientific computing ecosystem. Provides classic toy datasets, synthetic data generators, real-world benchmark datasets, domain-specific (astronomy/climate/genomics) loaders, HuggingFace-format-compatible readers, and more — all with a consistent, ergonomic API inspired by `scikit-learn.datasets`.

> **Status note**: a set of source files under `src/` (graph/text/image/anomaly/financial/medical/recommendation/knowledge-graph/physics/synthetic-signal/regression/time-series "benchmark" modules — roughly 17,900 lines total) exist on disk but are **not currently wired into the crate's module tree**: no `mod`/`pub mod` declaration anywhere in the crate reaches them, so none of their public items compile into the published crate or are reachable via `scirs2_datasets::`. See `TODO.md` for the full list. Everything documented below has been verified directly against `src/` on the 0.6.1 branch (2026-07-15).

## Features

### Classic Toy Datasets
- **Iris**: 150 samples, 4 features, 3 classes (Fisher's classic)
- **Boston Housing**: 506 samples, 13 features, regression (housing prices) — kept for API compatibility; deprecated upstream (see Known Issues)
- **Breast Cancer**: 569 samples, 30 features, binary classification
- **Wine**: 178 samples, 13 features, 3 classes
- **Digits**: 1797 samples, 64 features (8x8 pixel images), 10 classes
- **Diabetes**: 442 samples, 10 features, regression

### Synthetic Data Generators
- **Classification**: `make_classification` — linear/non-linear, configurable classes, clusters-per-class, informative features
- **Regression**: `make_regression` — configurable informative features and noise
- **Clustering**: `make_blobs` (Gaussian), `make_hierarchical_clusters` (nested main/sub-cluster structure)
- **Non-linear / manifold patterns**: `make_spirals`, `make_moons`, `make_circles`, `make_swiss_roll`, `make_s_curve`, `make_helix`, `make_torus`, `make_twin_peaks`, `make_severed_sphere`
- **Time series**: `make_time_series`, `make_ar_process`, `make_random_walk`, `make_seasonal`, `make_sine_wave` (trend/seasonality/noise all configurable)
- **Graphs**: `make_karate_club`, `make_random_graph`, `make_barabasi_albert`, `make_watts_strogatz`
- **Advanced generators**: `make_anomaly_dataset`, `make_adversarial_examples`, `make_continual_learning_dataset`, `make_domain_adaptation_dataset`, `make_few_shot_dataset`, `make_multitask_dataset`
- **Imbalanced data helpers**: `random_oversample`, `random_undersample`, `create_balanced_dataset` (configurable class-balance ratios)
- **Reproducible**: seed parameter (`Option<u64>`) threaded through every generator

### Real-World & Domain-Specific Datasets
- **Real-world benchmarks** (`RealWorldDatasets`): Adult, Titanic, Bank Marketing, German Credit, California Housing, Red/White Wine Quality, Energy Efficiency, Heart Disease, Diabetes Readmission, Credit Approval, Mushroom, Spam, Auto MPG, Concrete Strength, Air Passengers, Electricity Load, Stock Prices, Bitcoin Prices, CIFAR-10 subset, Fashion-MNIST subset, IMDB Reviews, News Articles, Credit Card Fraud, Loan Default
- **Domain-specific** (`domain_specific`): astronomy (stellar classification), climate, and genomics (gene expression) convenience loaders
- **Synthetic large-benchmark-format datasets**: M5 competition retail forecasting (`m5_dataset`), Penn Treebank / WikiText-103 language modelling, Criteo click-through-rate, ImageNet-100-class synthetic images
- **HuggingFace compatibility**: Arrow-backed `ArrowDataset` reader, `HfDatasetCard` metadata parsing/writing (`huggingface`, `arrow_dataset`, `hub_metadata`)
- **Quantum-inspired & neuromorphic generators**: `make_quantum_blobs`, `make_quantum_classification`, `make_quantum_regression`, `NeuromorphicProcessor`

### Dataset Utilities
- **Cross-Validation**: `k_fold_split`, `stratified_k_fold_split`, `time_series_split`, `train_test_split`
- **Sampling**: `random_sample`, `stratified_sample`, `importance_sample` (bootstrap sampling is also available via the `utils::sampling` module path)
- **Data Balancing**: `random_oversample`, `random_undersample`, `create_balanced_dataset`
- **Feature Engineering**: `polynomial_features`, `create_binned_features`, `statistical_features`
- **Scaling and Normalization**: `min_max_scale`, `robust_scale`, `normalize`
- **Caching**: `CacheManager` / `DatasetCache` — platform-specific disk caching with SHA256 integrity verification
- **Streaming & Sharding**: streaming iterators and `DataLoader`-style batching (`streaming`, `streaming_csv`), dataset sharding for distributed training (`sharding`)
- **Distributed primitives**: `par_map_rows`, `par_fold_rows`, `core_par_map_chunks`, `core_map_reduce_chunks`, `par_feature_stats` (backed by `scirs2-core`'s distributed thread-pool/parallel-iterator primitives)

### GPU Acceleration (optional)
- **`wgpu` feature**: real, threshold-gated `wgpu`/`GpuNdarray` dispatch inside `make_classification`/`make_regression`/`make_blobs` for large workloads (`GPU_DATASET_THRESHOLD` = 4096 output elements), with a silent, correctness-preserving fallback to the CPU path below the threshold or when no adapter is present
- **`AdvancedGpuOptimizer` benchmarking** (`gpu_optimization` module): genuinely measures CPU vs. GPU execution time rather than simulating it. `BenchmarkResult::gpu_time_ms` and `::speedup` are `Option<f64>` — `None` (never a fabricated number) whenever no real GPU dispatch executed, e.g. on a CPU-only backend
- **`cuda` feature**: optional NVIDIA-only acceleration via the pure-Rust `oxicuda-*` stack (`gpu_cuda` module), additive and separate from the `wgpu` path

## Installation

```toml
[dependencies]
scirs2-datasets = "0.6.6"
```

With remote dataset download support:

```toml
[dependencies]
scirs2-datasets = { version = "0.6.6", features = ["download"] }
```

## Quick Start

### Classic Datasets

```rust
use scirs2_datasets::{load_iris, load_boston, load_digits, load_wine, load_breast_cancer, load_diabetes};

let iris     = load_iris()?;
let boston   = load_boston()?;
let digits   = load_digits()?;
let wine     = load_wine()?;
let cancer   = load_breast_cancer()?;
let diabetes = load_diabetes()?;

println!("Iris:   {} samples, {} features, {} classes",
         iris.n_samples(), iris.n_features(),
         iris.targetnames().map_or(0, |t| t.len()));
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
let circles    = make_circles(200, 0.5, 0.1, Some(42))?;  // (n_samples, factor, noise, seed)
let swiss_roll = make_swiss_roll(500, 0.1, Some(42))?;
```

### Time Series

```rust
use scirs2_datasets::{make_time_series, make_ar_process, make_seasonal};

// Generic time series: n_samples, n_features, trend, seasonality, noise, seed
let ts = make_time_series(1000, 24, true, true, 0.1, Some(42))?;

// AR(2) process: n_samples, AR coefficients, noise_std, seed
let ar_ts = make_ar_process(500, &[0.7, -0.2], 0.1, Some(42))?;

// Seasonal series: n_samples, period, amplitude, trend, noise, seed
let seasonal_ts = make_seasonal(500, 12.0, 1.0, 0.05, 0.1, Some(42))?;
```

### Graph Datasets

```rust
use scirs2_datasets::{make_karate_club, make_random_graph, make_watts_strogatz, make_barabasi_albert};

let karate      = make_karate_club()?;                        // Zachary's karate club
let random_g    = make_random_graph(50, 0.3, Some(42))?;      // Erdos-Renyi, n_nodes=50
let small_world = make_watts_strogatz(100, 4, 0.3, Some(42))?; // n_nodes, k, rewiring prob, seed
let scale_free  = make_barabasi_albert(100, 3, Some(42))?;     // n_nodes, edges-per-new-node, seed

println!("Karate club: {} nodes", karate.n_samples());
```

### Anomaly Detection

```rust
use scirs2_datasets::{make_anomaly_dataset, AnomalyConfig};

let config = AnomalyConfig {
    anomaly_fraction: 0.05,
    random_state: Some(42),
    ..Default::default()
};
let anomaly_data = make_anomaly_dataset(1000, 10, config)?;
println!("Anomaly dataset: {} samples, {} features",
         anomaly_data.n_samples(), anomaly_data.n_features());
```

### Real-World Datasets (text, financial, tabular)

```rust
use scirs2_datasets::{RealWorldDatasets, RealWorldConfig};

let mut real_world = RealWorldDatasets::new(RealWorldConfig::default())?;

let imdb  = real_world.load_imdb_reviews()?;   // sentiment-style text dataset
let news  = real_world.load_news_articles()?;  // news-classification-style text dataset
let btc   = real_world.load_bitcoin_prices()?;  // synthetic financial time series
let stock = real_world.load_stock_prices()?;

println!("IMDB reviews: {} samples, {} features", imdb.n_samples(), imdb.n_features());
```

### Cross-Validation

```rust
use scirs2_datasets::{load_iris, k_fold_split, stratified_k_fold_split, train_test_split, time_series_split};

let iris = load_iris()?;

// Standard K-fold
let folds = k_fold_split(iris.n_samples(), 5, true, Some(42))?;
for (i, (train_idx, test_idx)) in folds.iter().enumerate() {
    println!("Fold {}: {} train, {} test", i, train_idx.len(), test_idx.len());
}

// Stratified K-fold
if let Some(targets) = &iris.target {
    let strat_folds = stratified_k_fold_split(targets, 5, true, Some(42))?;
    println!("Created {} stratified folds", strat_folds.len());
}

// Train/test split — returns a `DataSplit` with x_train/x_test/y_train/y_test
let split = train_test_split(&iris, Some(0.2))?;
println!("Train: {} rows, Test: {} rows", split.x_train.nrows(), split.x_test.nrows());

// Time series split (no data leakage): n_samples, n_splits, n_test_samples, gap
let ts_folds = time_series_split(1000, 5, 10, 0)?;
```

### Caching System

```rust
use scirs2_datasets::CacheManager;

let cache = CacheManager::new()?;
let stats = cache.get_stats();
println!("Cache: {} files, {} bytes", stats.file_count, stats.total_size_bytes);

// Remove one dataset from the cache
cache.remove("iris")?;

// Clear the entire cache
cache.clear_all()?;
```

## Dataset API

Every loader and generator returns the same concrete `Dataset` struct (`scirs2_datasets::utils::Dataset`, backed by `scirs2_core::ndarray` arrays) rather than a generic trait:

```rust
pub struct Dataset {
    pub data: Array2<f64>,
    pub target: Option<Array1<f64>>,
    pub targetnames: Option<Vec<String>>,
    pub featurenames: Option<Vec<String>>,
    pub feature_descriptions: Option<Vec<String>>,
    pub description: Option<String>,
    pub metadata: HashMap<String, String>,
}
```

Key accessor methods: `n_samples()`, `n_features()`, `shape()`, `has_target()`, `featurenames()`, `targetnames()`, `description()`, `metadata()` / `get_metadata()`. Builder methods (`with_featurenames`, `with_targetnames`, `with_feature_descriptions`, `with_description`, `with_metadata`) allow constructing custom datasets fluently, e.g. `Dataset::new(data, target).with_description(...)`.

## Module Map

Modules actually declared (`mod`/`pub mod`) in `src/lib.rs` and reachable via `scirs2_datasets::`:

| Module | Contents |
|--------|----------|
| `toy` / `standard` | Iris, Boston, Digits, Wine, Breast Cancer, Diabetes |
| `generators` (+ submodules `time_series`, `graph`, `sparse`, `classification`, `regression`, `structured`, `concept_drift`, `heterogeneous`, `low_rank`, `multilabel_advanced`) | `make_classification`, `make_regression`, `make_blobs`, non-linear/manifold patterns, graph/time-series/sparse/structured generators |
| `advanced_generators` | Anomaly, adversarial, continual-learning, domain-adaptation, few-shot, multi-task dataset generators |
| `real_world` | `RealWorldDatasets` — Adult, Titanic, housing, credit, medical, text, and financial benchmark-style loaders |
| `domain_specific` | Astronomy, climate, genomics convenience loaders |
| `quantum_enhanced_generators` / `neuromorphic_data_processor` / `quantum_neuromorphic_fusion` | Quantum-inspired and neuromorphic synthetic generators |
| `m5_dataset` / `penn_treebank` / `wikitext103` / `criteo` / `imagenet100` | Synthetic large-benchmark-format datasets |
| `arrow_dataset` / `huggingface` / `hub_metadata` | HuggingFace `datasets`-format compatibility |
| `sharding` / `streaming` / `streaming_csv` / `sampling` | Dataset sharding, streaming iterators, mini-batch sampling |
| `distributed` / `distributed_core` / `distributed_loading` | Distributed dataset processing primitives |
| `utils` | Cross-validation, train/test split, sampling, scaling, feature engineering, the core `Dataset` struct |
| `cache` | Disk caching with SHA256 verification |
| `gpu` / `gpu_optimization` / `gpu_cuda` | GPU-dispatch dataset generation and benchmarking (see GPU Acceleration above) |
| `formats` / `parquet_reader` / `hdf5_dataset` / `netcdf_dataset` | Parquet / HDF5 / NetCDF3 format readers |
| `lazy_loading` | Memory-mapped, zero-copy dataset access |
| `loaders` | CSV / JSON loading, streaming chunk iterators |

Not listed here: several additional source files (`graph_datasets.rs`, `graph_benchmarks.rs`, `image_datasets.rs`, `text_datasets.rs`, `anomaly_benchmarks.rs`, `financial.rs`, `medical_datasets.rs`, `recommendation_datasets.rs`, `knowledge_graph_datasets.rs`, `synthetic_signals.rs`, `regression_benchmarks.rs`, `time_series_benchmarks.rs`, `imbalanced.rs`, `mnist_like.rs`, `vision_datasets.rs`, and others) exist under `src/` but are not declared as modules anywhere and do not compile into the crate — see the status note above.

## Performance

- **Memory-efficient loading**: memory-mapped, zero-copy access via `lazy_loading` (feature `lazy-loading`), plus chunked CSV/Parquet streaming via `scirs2-io`
- **Fast generators**: vectorized synthetic data generation using `scirs2-core`'s RNG
- **Integrity verified**: SHA256 checksums on all cached downloads
- **Cross-platform caching**: platform-specific cache directories (XDG on Linux, Application Support on macOS, AppData on Windows)
- **GPU dispatch**: real `wgpu`/`GpuNdarray` acceleration for large-workload generators and for `AdvancedGpuOptimizer` benchmarking (see GPU Acceleration above); honest CPU fallback, never a fabricated speedup
- **Test coverage**: 583/583 tests passing (default features), 621/621 passing (`--all-features`) — 0 failed, 0 skipped either way; freshly measured 2026-07-15 via `cargo nextest run -p scirs2-datasets [--all-features]`, no `--lib` fallback needed

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
