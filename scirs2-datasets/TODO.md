# scirs2-datasets TODO

## Status: v0.3.4 Released (March 18, 2026) — v0.4.3 in progress (2026-05-03)

## v0.3.3 Completed

### Classic Toy Datasets
- [x] `load_iris` - 150 samples, 4 features, 3 classes
- [x] `load_boston` - 506 samples, 13 features, regression
- [x] `load_digits` - 1797 samples, 64 features, 10 classes
- [x] `load_wine` - 178 samples, 13 features, 3 classes
- [x] `load_breast_cancer` - 569 samples, 30 features, binary
- [x] `load_diabetes` - 442 samples, 10 features, regression
- [x] Consistent `Dataset<f64>` trait interface for all toy datasets
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
- [x] `make_arima_series` - ARIMA process generation
- [x] Reproducible results via seed parameter throughout

### Specialized Benchmarks (New in v0.3.1)
- [x] `graph_datasets` - Cora, Citeseer, PROTEINS graph datasets
- [x] `graph_benchmarks` - GNN benchmark suite
- [x] `image_datasets` - MNIST-like, CIFAR-10 format (synthetic)
- [x] `mnist_like` - Fashion-MNIST-like synthetic generation
- [x] `text_datasets` - 20 Newsgroups, IMDB, NER, QA datasets
- [x] `anomaly_benchmarks` - KDD Cup-style, synthetic anomaly injection
- [x] `financial` - Synthetic asset prices, volatility, portfolio matrices
- [x] `medical_datasets` - Synthetic MRI/CT-like volumetric datasets
- [x] `recommendation_datasets` - MovieLens-like interaction matrices
- [x] `knowledge_graph_datasets` - Entity-relation triples
- [x] `synthetic_signals` - DSP algorithm benchmark datasets
- [x] `physics` - N-body, fluid dynamics, wave equation snapshots
- [x] `regression_benchmarks` - Comprehensive regression benchmarks
- [x] `time_series_benchmarks` - M4-format time series benchmarks
- [x] `imbalanced` - Imbalanced classification datasets

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
- [x] ARFF (Weka format)
- [x] LIBSVM sparse format
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

- `load_boston` is included for API compatibility but is deprecated in scikit-learn due to ethical concerns about the original dataset; document this prominently.
- Large datasets (>1 GB) should be accessed via streaming API (v0.4.0); attempting to load them fully into memory may cause OOM on constrained systems.
- The `download` feature requires network access at test time; CI environments without internet should use `--no-default-features` or mock the download.
- ARFF parser does not handle all relational attribute types; sparse ARFF is only partially supported.
