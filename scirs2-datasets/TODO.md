# scirs2-datasets TODO

## Status: v0.3.4 Released (March 18, 2026) — v0.6.2 Released (2026-07-22)

**0.6.2**: `hdf5_dataset.rs` now reads through the new oxih5-backed HDF5 module (see scirs2-io's `[0.6.2]` CHANGELOG entry for the underlying backend change) instead of the old `hdf5_lite`/C-`hdf5` paths. New `tls.rs` installs the pure-Rust `oxitls-rustcrypto-provider` TLS crypto provider before this crate's HTTP/HTTPS requests (`ExternalClient`, cache downloads) — needed because the workspace `reqwest`/`ureq` dependencies now build with `rustls-no-provider` instead of the C `aws-lc-rs`/`ring` backends.

`AdvancedGpuOptimizer` now performs real wgpu/GpuNdarray GPU dispatch instead of a simulated mock (`BenchmarkResult.gpu_time_ms`/`.speedup` are now `Option<f64>` — `None` whenever no real GPU dispatch ran, never a fabricated number; regression-tested by `test_benchmark_performance_reports_real_measurements_not_fabricated_speedup` in `src/gpu_optimization.rs`).

**Audit note (2026-07-15)**: a documentation-vs-implementation audit found that the entire "Specialized Benchmarks (New in v0.3.1)" section below (~17,900 lines across the underlying source files) is **not wired into the crate's module tree** — no `mod`/`pub mod` declaration anywhere in `src/` reaches these files, verified via a full-tree scan, so none of it compiles into the published crate or is reachable via `scirs2_datasets::`. Those items have been unchecked below with per-item detail. `LIBSVM sparse format` under "Data Format Support" was also unchecked — no LIBSVM parsing code exists anywhere in `src/`.

**Freshly measured test count (2026-07-15)**: `cargo nextest run -p scirs2-datasets` (default features) → **583 tests run: 583 passed, 0 skipped**. `cargo nextest run -p scirs2-datasets --all-features` → **621 tests run: 621 passed, 0 skipped** (extra 38 tests include `gpu_optimization::tests::test_gpu_backend_speedup_reflects_real_dispatch_when_available` and additional streaming/wikitext103 coverage gated behind non-default features). Neither run needed the `--lib` fallback — the known example-binary linker/OOM issue did not reproduce this time, though both runs took unusually long (tens of minutes) due to heavy, unrelated concurrent `cargo`/`rustc` load on the machine from other sessions, not a fault in this crate.

## v0.3.3 Completed

### Classic Toy Datasets
- [x] `load_iris` - 150 samples, 4 features, 3 classes
- [x] `load_boston` - 506 samples, 13 features, regression
- [x] `load_digits` - 1797 samples, 64 features, 10 classes
- [x] `load_wine` - 178 samples, 13 features, 3 classes
- [x] `load_breast_cancer` - 569 samples, 30 features, binary
- [x] `load_diabetes` - 442 samples, 10 features, regression
- [x] Consistent `Dataset` struct interface for all toy datasets (concrete struct with `f64` arrays, not a generic trait — see README "Dataset API")
- [x] Feature names and target names on all datasets

### Synthetic Data Generators
- [x] `make_classification` - Linear and non-linear, multi-class, redundant features
- [x] `make_regression` - Multi-output regression, configurable informative features
- [x] `make_blobs` - Gaussian blobs for clustering benchmarks
- [x] `make_circles` - Concentric circles
- [x] `make_moons` - Two interleaved half-moons
- [x] `make_spirals` - Interlaced spirals
- [x] `make_swiss_roll` - 3D Swiss roll manifold
- [x] `make_time_series` - Univariate and multivariate time series
- [ ] `make_arima_series` - ARIMA process generation — no such function exists anywhere in `src/` (checked 2026-07-15); the former README example calling it has been replaced with the real `make_ar_process`/`make_seasonal` generators
- [x] Reproducible results via seed parameter throughout

### Specialized Benchmarks (New in v0.3.1) — DISCONNECTED (see audit note above)
> Every item below has a real, substantial implementation on disk, but as of 2026-07-15 none of these files is declared via `mod`/`pub mod` anywhere in the crate (checked `src/lib.rs` and every other `.rs` file). None of their public items compile into `scirs2-datasets` or are reachable via `scirs2_datasets::`. Only re-check `[x]` once a module is actually wired back in (adding the `mod` declaration requires editing `src/lib.rs`, out of scope for a docs-only pass).
- [ ] `graph_datasets` - Cora, Citeseer, PROTEINS graph datasets — `src/graph_datasets.rs` (567 lines) exists but is unwired, **and** its actual contents are Karate-club/SBM/Barabasi-Albert/Watts-Strogatz generators, not Cora/Citeseer/PROTEINS loaders (equivalent real, wired generators ship separately via `generators::graph`)
- [ ] `graph_benchmarks` - GNN benchmark suite — `src/graph_benchmarks.rs` (1011 lines) exists, unwired
- [ ] `image_datasets` - MNIST-like, CIFAR-10 format (synthetic) — `src/image_datasets.rs` (646 lines) exists, unwired; contents are checkerboard/circles/gradient/noisy/blobs image generators, not MNIST/CIFAR-10
- [ ] `mnist_like` - Fashion-MNIST-like synthetic generation — `src/mnist_like.rs` (1209 lines) exists, unwired
- [ ] `text_datasets` - 20 Newsgroups, IMDB, NER, QA datasets — `src/text_datasets.rs` (895 lines) exists, unwired
- [ ] `anomaly_benchmarks` - KDD Cup-style, synthetic anomaly injection — `src/anomaly_benchmarks.rs` (1084 lines) exists, unwired
- [ ] `financial` - Synthetic asset prices, volatility, portfolio matrices — `src/financial.rs` (910 lines) exists, unwired
- [ ] `medical_datasets` - Synthetic MRI/CT-like volumetric datasets — `src/medical_datasets.rs` (623 lines) exists, unwired
- [ ] `recommendation_datasets` - MovieLens-like interaction matrices — `src/recommendation_datasets.rs` (662 lines) exists, unwired
- [ ] `knowledge_graph_datasets` - Entity-relation triples — `src/knowledge_graph_datasets.rs` (826 lines) exists, unwired
- [ ] `synthetic_signals` - DSP algorithm benchmark datasets — `src/synthetic_signals.rs` (796 lines) exists, unwired
- [ ] `physics` - N-body, fluid dynamics, wave equation snapshots — `src/physics/` (mod.rs/chaotic_systems.rs/fluid.rs/ode_systems.rs, 2343 lines total) exists, unwired
- [ ] `regression_benchmarks` - Comprehensive regression benchmarks — `src/regression_benchmarks.rs` (951 lines) exists, unwired; note `make_friedman1/2/3` ship separately, wired, via `generators::regression`
- [ ] `time_series_benchmarks` - M4-format time series benchmarks — `src/time_series_benchmarks.rs` (865 lines) exists, unwired; the README's former `TimeSeriesBenchmark::load(...)` example referenced a type that does not exist anywhere in `src/` and has been replaced
- [ ] `imbalanced` - Imbalanced classification datasets — `src/imbalanced.rs` (1096 lines) exists, unwired; note the real, wired `random_oversample`/`random_undersample`/`create_balanced_dataset` (a different implementation, in `utils`) remain available at the crate root regardless

Also unwired, never previously tracked here: `src/vision_datasets.rs` (975 lines), `src/ts_datasets.rs` (637 lines), `src/benchmark/` (ml_benchmarks.rs/test_functions.rs/mod.rs, 1676 lines), plus two orphaned test-only files (`adaptive_streaming_engine_tests.rs`, `real_world_tests.rs`). `src/benchmarks_new/` is an empty directory (no files).

### Dataset Utilities
- [x] `k_fold_split` - Standard K-fold splitting
- [x] `stratified_k_fold_split` - Stratified K-fold (preserves class balance)
- [x] `time_series_split` - Non-leaking time series cross-validation
- [x] `train_test_split` - Random and stratified train/test split
- [x] `random_sample`, `stratified_sample`, `bootstrap_sample`, `importance_sample`
- [x] `create_balanced_dataset`, `random_oversample`, `random_undersample`
- [x] `polynomial_features`, `create_binned_features`, `statistical_features`
- [x] `min_max_scale`, `robust_scale`, `normalize`
- [x] `CacheManager` with SHA256 integrity verification
- [x] Platform-specific cache directories

### Data Format Support
- [x] CSV loading with type inference
- [x] JSON dataset format
- [x] ARFF (Weka format) — basic/simplified parser (`external.rs::parse_arff_data`); dense numeric ARFF only, see Known Issues
- [ ] LIBSVM sparse format — no LIBSVM parsing code found anywhere in `src/` (checked 2026-07-15); unchecked as a discrepancy, this was not actually implemented
- [x] Memory-mapped loading for large datasets

### Testing and Quality
- [x] 117+ unit tests covering all public APIs
- [x] Zero-warning builds
- [x] All public APIs documented with examples

## v0.4.0 Roadmap

### Streaming Large-Scale Datasets
- [x] Async streaming iterator API for datasets that exceed available RAM — Implemented in v0.4.0 (`streaming/iterator.rs`)
- [x] Chunked CSV/Parquet streaming via `scirs2-io` — Implemented in v0.4.0 (`streaming/iterator.rs`)
- [x] Lazy evaluation for dataset transformations (map, filter, batch) — Implemented in v0.4.0 (`streaming/transforms.rs`)
- [x] `DataLoader`-style batching API for neural network training — Implemented in v0.4.0 (`streaming/dataloader.rs`)

### HuggingFace Dataset Format Compatibility
- [x] Read `datasets` format (Arrow-backed, parquet shards) — `src/arrow_dataset.rs` (`ArrowDataset`, `DatasetInfo`, `FeatureType`; magic-byte validation + directory scan in default builds; full IPC decoding with `parquet_io` feature)
- [x] Support HuggingFace Hub metadata schema (dataset cards) — `src/huggingface.rs` (`HfDatasetCard`, `parse_dataset_card`, `load_dataset_card`, `to_hf_card`, `card_to_readme`)
- [x] Load datasets from local HuggingFace cache directory — `load_dataset_card(dir: &Path)` in `src/huggingface.rs`
- [x] Convert SciRS2 datasets to HuggingFace `datasets` format — `to_hf_card` + `card_to_readme` in `src/huggingface.rs`

### Additional Benchmark Datasets
- [x] M5 competition time series (retail forecasting) — `src/m5_dataset.rs` (`M5Dataset`, `M5Config`, `M5Record`; Poisson demand + weekly seasonality + item trends)
- [x] Penn Treebank (language modelling) — `src/penn_treebank.rs` (`PennTreebankDataset`, `PennTreebankConfig`; Zipf vocab + Poisson sentence lengths + `load_from_text`)
- [x] WikiText-103 (NLP language modelling) — `src/wikitext103.rs` (`WikiText103Dataset`, `WikiText103Config`; article/paragraph hierarchy + `load_from_text`)
- [x] Criteo display advertising (click-through rate) — `src/criteo.rs` (`CriteoDataset`, `CriteoConfig`, `CriteoRecord`; 13 int + 26 cat features, binary label)
- [x] ImageNet subset (100-class synthetic) — `src/imagenet100.rs` (`ImageNet100Dataset`, `ImageNet100Config`; `Array4<f32>` NCHW, 100 classes, distinct class means)

### Distributed Dataset Processing
- [x] Shard-aware loading for multi-process/multi-node training — `ShardedLoader` in `src/sharding/mod.rs`
- [x] Dataset sharding API: split dataset into N equal parts by index — `ShardedLoader::get_shard`, `shard_by_index` in `src/sharding/mod.rs`
- [x] Consistent random shuffling across shards with same seed — `ShardedLoader::global_permutation` + `consistent_shuffle` in `src/sharding/mod.rs`
- [x] Integration with `scirs2-core` distributed primitives — `src/distributed_core.rs` (`par_map_rows`, `par_fold_rows`, `core_par_map_chunks`, `core_map_reduce_chunks`, `par_feature_stats`; backed by `scirs2_core::distributed::par_iter` and `scirs2_core::distributed::primitives`; 14 tests)

### Enhanced Generators
- [x] `make_low_rank` - Low-rank matrix completion benchmarks — `generators/low_rank.rs` + ndarray wrapper in `generators/ndarray_convenience.rs`
- [x] `make_sparse_classification` - Very high-dimensional sparse features — `generators/sparse_classification.rs` + ndarray wrapper
- [x] `make_multilabel_classification` - True multi-label (not one-hot) — `generators/classification.rs` + ndarray wrapper
- [x] `make_heterogeneous` - Mixed numeric/categorical features — `generators/heterogeneous.rs` + ndarray wrapper
- [x] `make_concept_drift` - Time series with distribution shift — `generators/concept_drift.rs` + ndarray wrapper

### Format Support
- [x] Native Parquet read via `scirs2-io` — `parquet_reader.rs` (`parquet_io` feature, `ParquetDataset` / `ColumnData`)
- [x] HDF5 dataset containers — `hdf5_dataset.rs` (magic-byte check always; full read/write with `hdf5_io` feature)
- [x] NetCDF for climate/geospatial datasets — `netcdf_dataset.rs` (pure Rust via `netcdf3`, always available)

## Known Issues

- **~17,900 lines of unwired source** (see "Specialized Benchmarks" audit note above): `graph_datasets.rs`, `graph_benchmarks.rs`, `image_datasets.rs`, `text_datasets.rs`, `anomaly_benchmarks.rs`, `financial.rs`, `medical_datasets.rs`, `recommendation_datasets.rs`, `knowledge_graph_datasets.rs`, `synthetic_signals.rs`, `regression_benchmarks.rs`, `time_series_benchmarks.rs`, `imbalanced.rs`, `mnist_like.rs`, `vision_datasets.rs`, `ts_datasets.rs`, `benchmark/` are real but not declared as modules anywhere, so they do not compile into the crate. Fixing this requires adding `mod`/`pub mod` declarations to `src/lib.rs` (a source change, tracked here for whoever picks it up next) plus re-auditing each file for correctness/currency before re-exposing it.
- `load_boston` is included for API compatibility but is deprecated in scikit-learn due to ethical concerns about the original dataset; document this prominently.
- Large datasets (>1 GB) should be accessed via streaming API (v0.4.0); attempting to load them fully into memory may cause OOM on constrained systems.
- The `download` feature requires network access at test time; CI environments without internet should use `--no-default-features` or mock the download.
- ARFF parser does not handle all relational attribute types; sparse ARFF is only partially supported.
- `cargo nextest run -p scirs2-datasets` (which also builds example binaries) has a known history of OOM-ing the linker; use `--lib` to measure real test pass/fail counts when this happens.
