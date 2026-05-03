# scirs2-cluster TODO

## Status: v0.3.4 Released (March 18, 2026) — v0.4.3 in progress (2026-05-03)

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
- Persistent homology applied to clustering (`topological_clustering.rs`, `topological/`)
- Mapper-based topological cluster summaries

### Streaming and Online Clustering
- Online k-means (incremental update)
- ADWIN-based streaming cluster tracking
- CluStream for evolving data streams
- Reservoir sampling (`streaming_cluster.rs`, `stream/`)

### Time Series Clustering
- DTW-based distance for time series k-means
- Temporal pattern recognition clustering (`time_series_clustering/`)

### Ensemble and Consensus
- Co-association matrix consensus clustering
- Evidence Accumulation Clustering (EAC)
- Bagging-based ensemble
- Weighted voting ensemble (`ensemble/weighted.rs`)
- Stability-based cluster number selection (`stability_advanced.rs`)

### Deep Clustering
- Autoencoder-based deep embedding (`deep_cluster.rs`)
- Transformer cluster embeddings
- GNN-based clustering

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
- Cryptographic integrity hashing
- Cross-platform compatibility metadata
- Training metrics tracking (time, memory, CPU)

## v0.4.0 Roadmap

### GPU Acceleration
- CUDA-accelerated k-means (cuML-compatible interface)
- GPU-based GMM EM fitting
- OpenCL backend for cross-platform GPU support
- Automatic CPU/GPU selection based on data size

### Distributed Clustering
- Distributed k-means via message passing
- Federated clustering across nodes
- Hierarchical clustering for partitioned datasets

### Graph-Based Clustering Improvements — Partially implemented in v0.4.0
- [x] Community detection: Louvain, Leiden, label propagation (via scirs2-graph) — Leiden and label propagation implemented in v0.4.0
- [x] Stochastic block model fitting — Implemented in v0.4.0
- Overlapping community detection

### Online Learning Enhancements
- Concept drift detection with cluster structure updates
- Self-adaptive mini-batch sizing

### Visualization
- Native dendrogram plotting (plotters integration)
- Scatter plot with automatic 2D projection for any dimensionality
- Interactive cluster exploration API

## Known Issues / Technical Debt

- `native_plotting.rs` depends on optional plotting library; ensure feature-gating is correct
- Spectral clustering eigenvalue computation uses dense fallback for very large graphs; sparse eigensolver integration planned
- Deep clustering modules depend on `scirs2-neural`; circular dependency risk if `scirs2-neural` ever imports from `scirs2-cluster`
- Some SOM convergence criteria need tuning for non-Euclidean topologies
