# scirs2-cluster TODO

## Status: v0.6.5 (reviewed 2026-07-31)

Untouched by this release's fix work (no cluster-specific changes shipped in 0.6.5); the review
below — last performed against source on 2026-07-15 — remains accurate since the crate source is
unchanged.

Reviewed against the real `src/` implementation and a full `cargo nextest` run on 2026-07-15:
default-features 962/962 tests passing (3 skipped, 0 failed); all-features 1061/1061 tests
passing (3 skipped, 0 failed). 0 `todo!()`/`unimplemented!()` in production code (the only hit
is inside a doc-comment example). `unwrap()` usage is confined to `#[cfg(test)]` code.
Substantial functionality has been added since the "v0.3.3 Completed" snapshot below — see
README.md for the current full feature list.

## v0.3.3 Completed

### Partitional Clustering
- K-means with K-means++ and random initialization
- Mini-batch K-means for large datasets
- Parallel K-means (Rayon)
- `kmeans2` SciPy-compatible interface
- Data whitening / normalization utilities

### Hierarchical Clustering
- Agglomerative clustering: single, complete, average, Ward, centroid, median, weighted linkage
- Optimized Ward's method O(n^2 log n)
- Dendrogram utilities: `fcluster`, dendrogram traversal
- Dendrogram export: Newick, JSON
- Scikit-learn / SciPy model import/export

### Density-Based Clustering
- DBSCAN with custom distance metrics
- OPTICS (reachability plot + cluster extraction)
- HDBSCAN (hierarchical DBSCAN)
- Density peaks algorithm
- Density ratio estimation clustering

### Probabilistic and Mixture Models
- Gaussian Mixture Models (GMM) with EM: full, diagonal, spherical covariance
- Bayesian GMM with variational inference
- Dirichlet Process mixture models (nonparametric)
- Soft / probabilistic cluster assignments

### Prototype-Based and Competitive Learning
- Self-Organizing Maps (SOM) with hexagonal and rectangular topologies
- Competitive learning / winner-take-all networks
- Prototype-enhanced clustering
- Leader algorithm

### Spectral and Graph-Based
- Spectral clustering with normalized and unnormalized Laplacians
- Affinity propagation
- BIRCH (Balanced Iterative Reducing using Hierarchies)
- Mean-shift clustering

### Subspace Clustering
- Subspace clustering for high-dimensional data
- Projected clustering
- Advanced subspace methods

### Fuzzy and Soft Clustering
- Fuzzy c-means (FCM)
- Soft clustering with membership degrees
- Possibilistic c-means variant

### Topological Clustering
- Persistent homology applied to clustering (`topological_clustering.rs` — NOT the unreachable `topological/` directory, see Known Issues)
- Mapper-based topological cluster summaries

### Streaming and Online Clustering
- Online k-means (incremental update)
- ADWIN-based streaming cluster tracking
- CluStream for evolving data streams
- Reservoir sampling (`streaming.rs`)

### Time Series Clustering
- DTW-based distance for time series k-means (`time_series.rs` — the similarly-named `time_series_clustering/` directory is a separate, unreachable duplicate, see Known Issues)

### Ensemble and Consensus
- Co-association matrix consensus clustering
- Evidence Accumulation Clustering (EAC)
- Bagging-based ensemble
- Weighted voting ensemble (`ensemble/weighted.rs`)
- Stability-based cluster number selection (`stability_advanced.rs`)

### Deep Clustering
- Autoencoder-based deep embedding (`deep_cluster.rs`)
- Transformer cluster embeddings
- [ ] GNN-based clustering — NOT actually implemented: `MessagePassingNeuralNetwork` and `GraphAttentionNetwork` (`src/enhanced_clustering_features.rs`) are empty unit structs (comment: "Implementation of placeholder structures"); `new()` just returns `Self` with no fields or computation. Un-checked here since it was previously listed as done. (Note: `detect_communities` in the same file is real, non-stub Raghavan-style label-propagation community detection — the placeholder issue is specific to the GNN types, not the whole file)

### Biclustering and Co-clustering
- Biclustering (`biclustering.rs`)
- Co-clustering / information-theoretic co-clustering (`coclustering.rs`)

### Evaluation Metrics
- Silhouette coefficient
- Davies-Bouldin index
- Calinski-Harabasz index
- Gap statistic
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-measure
- Stability analysis (`cluster_metrics.rs`)

### Serialization
- Model persistence with versioned metadata
- Cross-platform compatibility metadata
- Training metrics tracking (time, memory, CPU)
- [ ] Cryptographic integrity hashing — NOT actually implemented: `ModelMetadata::integrity_hash` (`src/serialization/core.rs`) is never assigned a real computed hash anywhere in the module (stays `String::new()`), and the only "validation" is `!integrity_hash.is_empty()` (comment on that line: "Simple validation - in practice this would check the hash"). No crypto-hash crate is even a dependency of this crate. Un-checked here since it was previously listed as done

## Roadmap Status (re-verified 2026-07-15 against src/, crate now at v0.6.2)

### GPU Acceleration — scaffolding real, hardware dispatch still placeholder
- [x] GPU device/memory/backend abstraction and `GpuKMeans` (`src/gpu/`) — device selection, memory pooling, tensor-core config types are all implemented
- [x] Automatic CPU/GPU selection based on data size — `DeviceSelector` / `GpuContext::effective_backend()` implemented (always resolves to CPU fallback today, see below)
- [ ] CUDA-accelerated k-means (cuML-compatible interface) — `BackendContext::Cuda` handles are literal placeholder values (`context_handle: 0`, comment "Simplified for stub implementation"); no real CUDA dispatch
- [ ] OpenCL backend for cross-platform GPU support — same placeholder-handle status as CUDA
- [ ] GPU-based GMM EM fitting — no GPU code path found in `gmm.rs` or `src/gpu/`

### Distributed Clustering — implemented as in-process simulation, not networked multi-node
- [x] Distributed k-means via message passing — `src/distributed/` has a real `MessagePassingCoordinator`, fault tolerance, and load-balancing coordinators; transport is `std::sync::mpsc` in-process (not real network sockets — no `TcpStream`/`TcpListener` in the module)
- [x] Federated clustering across nodes — `federated_ensemble`/`FederationConfig` in `src/ensemble/convenience.rs` (in-process aggregation simulation, same caveat as above)
- [ ] Hierarchical clustering for partitioned datasets — no distributed-hierarchical implementation found

### Graph-Based Clustering Improvements — complete
- [x] Community detection: Louvain, Leiden, label propagation (via scirs2-graph)
- [x] Stochastic block model fitting
- [x] Overlapping community detection — BigCLAM, DEMON, Link Communities + evaluation metrics now implemented in `src/overlapping/` (594+397+421+331 lines)

### Online Learning Enhancements — complete
- [x] Concept drift detection with cluster structure updates — `src/drift/` (ADWIN, DDM detectors)
- [x] Self-adaptive mini-batch sizing — `current_batch_size`/`adjustment_factor` adaptive logic in `src/streaming.rs`

### Visualization — mostly complete
- [x] Native dendrogram plotting (plotters integration) — `src/plotting.rs` (real `plotters::prelude` usage)
- [x] Interactive cluster exploration API — `src/visualization/interactive.rs` (901 lines: mouse/keyboard/touch handling, region selection, nearest-point highlighting)
- [ ] Scatter plot with automatic 2D/3D projection for any dimensionality — `create_scatter_plot_2d/3d` and `DimensionalityReduction::{PCA,TSNE,UMAP}` dispatch exist, but the PCA projection itself is commented "Simplified PCA projection (stub implementation)" and the t-SNE/UMAP paths currently fall back to PCA rather than running their own algorithm

## Known Issues / Technical Debt

- `src/native_plotting/` (now a module directory, not a single file) is actually a self-contained SVG renderer with no real usage of the `plotters`/`egui` crates, despite being gated behind `#[cfg(any(feature = "plotters", feature = "egui"))]` in `lib.rs`; the module that genuinely depends on `plotters` is the separate `src/plotting.rs` (confirmed real `use plotters::prelude::*`)
- Spectral clustering eigenvalue computation uses dense fallback for very large graphs; sparse eigensolver integration still planned (unchanged)
- Deep clustering modules (`deep_cluster.rs`) do NOT depend on `scirs2-neural` — no such dependency exists in `Cargo.toml` and no `scirs2_neural` reference exists anywhere in this crate's source; the previously-noted circular-dependency risk no longer applies (or the code was already refactored to remove it)
- Some SOM convergence criteria need tuning for non-Euclidean topologies (unverified either way; left as-is)
- **Eight entire module directories under `src/` are not referenced by any `mod` statement anywhere in the crate (not even non-`pub`), so they are 100% unreachable dead code that rustc never even parses — no `dead_code` warning is possible because the compiler never sees these files.** Verified by rigorously diffing every `src/*/` directory name against every exact `mod`/`pub mod` declaration crate-wide (2026-07-15, regex-based check across all `.rs` files, not just `lib.rs`), ~12,300 lines total:
  - `src/probabilistic/` — `crp.rs`, `dpgmm.rs`, `variational_gmm.rs` (Chinese Restaurant Process, Dirichlet Process GMM, Variational Bayes GMM; 2,911 lines; has its own `predict_proba`/`predict` APIs)
  - `src/bayesian_clustering/` — `dpmm.rs`, `gaussian_mixture_bayes.rs` (1,468 lines)
  - `src/stream/` — `clustream.rs`, `denstream.rs`, `birkm.rs` (2,648 lines)
  - `src/topological/` — `mapper.rs`, `pers_homology_cluster.rs`, `filtrations.rs`, `cover_tree.rs` (2,604 lines)
  - `src/time_series_clustering/` — `DTWDistance`, `KMeansDTW`, `DBABarycenter`, `ShapeBasedDistance`/k-Shape SBD (706 lines) — note this is a *different, unreachable* module from the real, wired `time_series.rs`
  - `src/subspace_advanced/` — `lrr.rs`, `orsc.rs`, `ssc.rs` (848 lines) — a *different, unreachable* module from the real, wired `subspace_enhanced.rs` (which has its own SSC/LRSC)
  - `src/density_peaks_adv/` — `algorithm.rs`, `decision_graph.rs` (632 lines)
  - `src/finch/` — FINCH clustering (525 lines)

  Good news: this does **not** invalidate the README/TODO feature claims for GMM, Dirichlet-process mixtures, CluStream/DenStream, DTW-based time series clustering, advanced subspace clustering, or persistent-homology/Mapper — each of those has a *separate, genuinely-wired* implementation elsewhere (`gmm.rs`'s own `DirichletProcessMixtureModel`/`DpmmResult`; `soft_clustering.rs`'s `GaussianMixtureModel`; `streaming_cluster.rs`'s own `CluStream`/`CluStreamConfig`; `time_series.rs`'s own DTW k-means/k-medoids/barycenter; `subspace_enhanced.rs`'s own SSC/LRSC; `topological_clustering.rs`'s own `persistent_homology_0d`/`mapper_graph`). The orphaned directories look like an abandoned parallel rewrite — recommend either wiring them in (after reconciling API overlap with the existing wired versions) or deleting them. Given the pattern kept surfacing new instances on closer inspection, a full `cargo-udeps`-style or manual audit of every `src/*/` directory vs. `mod` graph is recommended before the next release to make sure no more are lurking
- The subspace-clustering LRSC/SSC tests (`src/subspace_enhanced.rs`) that previously timed out are now passing again (verified 2026-07-15), but remain the slowest tests in the suite (10-40s each vs. sub-second for the rest) — not currently a failure, but worth watching if it regresses
