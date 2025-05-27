# Clustering Algorithm Comparison Guide

This guide helps you choose the right clustering algorithm for your data and use case.

## Algorithm Overview

### Partitioning Methods

#### K-means
- **When to use**: When you know the number of clusters and expect spherical, equally-sized clusters
- **Pros**: Fast, scalable, simple to understand
- **Cons**: Requires knowing k, sensitive to initialization, assumes spherical clusters
- **Time complexity**: O(n·k·i·d) where i=iterations, d=dimensions
- **Example use case**: Customer segmentation with known segments

#### Mini-batch K-means
- **When to use**: Large datasets where standard K-means is too slow
- **Pros**: Much faster than K-means, good approximation
- **Cons**: Slightly less accurate than standard K-means
- **Time complexity**: O(b·k·i·d) where b=batch size
- **Example use case**: Real-time clustering of streaming data

#### Parallel K-means
- **When to use**: Large datasets with multi-core processors available
- **Pros**: Significant speedup on multi-core systems
- **Cons**: Same limitations as standard K-means
- **Time complexity**: O(n·k·i·d/p) where p=processors
- **Example use case**: Large-scale image clustering

### Hierarchical Methods

#### Agglomerative Clustering
- **When to use**: When you need a hierarchy of clusters or don't know k
- **Pros**: Produces dendrogram, no need to specify k, can use various linkage criteria
- **Cons**: O(n²) memory, slow for large datasets
- **Time complexity**: O(n²log n) to O(n³) depending on linkage
- **Example use case**: Taxonomic classification, gene analysis

### Density-based Methods

#### DBSCAN
- **When to use**: Non-spherical clusters, noisy data, unknown number of clusters
- **Pros**: Finds arbitrary shaped clusters, robust to outliers, no need to specify k
- **Cons**: Struggles with varying densities, parameter sensitive
- **Time complexity**: O(n log n) with spatial index
- **Example use case**: Spatial data analysis, anomaly detection

#### HDBSCAN
- **When to use**: Varying density clusters, hierarchical density structure
- **Pros**: Handles varying densities, produces hierarchy, robust
- **Cons**: More complex than DBSCAN, slower
- **Time complexity**: O(n log n)
- **Example use case**: Social network analysis, astronomical data

#### OPTICS
- **When to use**: Need to explore different density parameters
- **Pros**: Parameter-free exploration, produces reachability plot
- **Cons**: Doesn't produce clusters directly, requires interpretation
- **Time complexity**: O(n log n) with spatial index
- **Example use case**: Exploratory data analysis

### Distribution-based Methods

#### Gaussian Mixture Models (GMM)
- **When to use**: Overlapping clusters, need probability assignments
- **Pros**: Soft clustering, handles elliptical clusters, probabilistic framework
- **Cons**: Sensitive to initialization, assumes Gaussian distributions
- **Time complexity**: O(n·k·d²·i) for full covariance
- **Example use case**: Speech recognition, image segmentation

### Other Methods

#### Mean Shift
- **When to use**: Unknown number of clusters, non-parametric density estimation
- **Pros**: No need to specify k, finds modes of density
- **Cons**: Computationally expensive, bandwidth parameter sensitive
- **Time complexity**: O(n²) per iteration
- **Example use case**: Image segmentation, object tracking

#### Spectral Clustering
- **When to use**: Non-convex clusters, graph-structured data
- **Pros**: Handles complex cluster shapes, graph-based
- **Cons**: Expensive eigenvalue computation, memory intensive
- **Time complexity**: O(n³) for eigenvalue decomposition
- **Example use case**: Image segmentation, community detection

#### Affinity Propagation
- **When to use**: Need to identify exemplars, no prior knowledge of k
- **Pros**: Automatically determines k, identifies representative points
- **Cons**: O(n²) memory and time, parameter sensitive
- **Time complexity**: O(n²·i)
- **Example use case**: Finding representative documents, face clustering

#### BIRCH
- **When to use**: Very large datasets, limited memory
- **Pros**: Single scan of data, handles large datasets, incremental
- **Cons**: Limited to spherical clusters, threshold parameter sensitive
- **Time complexity**: O(n)
- **Example use case**: Large-scale customer data, streaming applications

## Decision Tree

```
Start
│
├─ Know number of clusters?
│  ├─ Yes
│  │  ├─ Spherical clusters?
│  │  │  ├─ Yes
│  │  │  │  ├─ Large dataset?
│  │  │  │  │  ├─ Yes → Mini-batch K-means / Parallel K-means
│  │  │  │  │  └─ No → K-means
│  │  │  └─ No
│  │  │     ├─ Overlapping clusters? → GMM
│  │  │     └─ Complex shapes → Spectral Clustering
│  │  └─ No
│  │     ├─ Need hierarchy? → Agglomerative
│  │     ├─ Density-based? → DBSCAN/HDBSCAN
│  │     ├─ Need exemplars? → Affinity Propagation
│  │     └─ Very large data? → BIRCH
```

## Performance Comparison

| Algorithm | Time Complexity | Memory | Scalability | Handles Noise | Cluster Shape |
|-----------|----------------|---------|-------------|---------------|---------------|
| K-means | O(nki) | O(nk) | Excellent | Poor | Spherical |
| Mini-batch K-means | O(bki) | O(bk) | Excellent | Poor | Spherical |
| Agglomerative | O(n²log n) | O(n²) | Poor | Fair | Any |
| DBSCAN | O(n log n)* | O(n) | Good | Excellent | Any |
| HDBSCAN | O(n log n) | O(n) | Good | Excellent | Any |
| GMM | O(nkd²i) | O(nkd²) | Fair | Fair | Elliptical |
| Mean Shift | O(n²i) | O(n) | Poor | Good | Any |
| Spectral | O(n³) | O(n²) | Poor | Fair | Any |
| Affinity Prop | O(n²i) | O(n²) | Poor | Good | Any |
| BIRCH | O(n) | O(B) | Excellent | Fair | Spherical |

*With spatial index, otherwise O(n²)

## Parameter Selection Guidelines

### K-means Family
- **k**: Use elbow method, silhouette analysis, or domain knowledge
- **initialization**: K-means++ generally best, use parallel for large k

### DBSCAN/HDBSCAN
- **eps**: Use k-distance plot, look for "elbow"
- **min_samples**: Usually 2×dimensions, minimum 3-5

### GMM
- **n_components**: Use BIC/AIC for model selection
- **covariance_type**: Start with 'full', use 'diagonal' for high dimensions

### Mean Shift
- **bandwidth**: Use estimate_bandwidth() or cross-validation

### Spectral Clustering
- **n_clusters**: Like K-means
- **affinity**: 'rbf' for most cases, 'nearest_neighbors' for large datasets

## Code Examples

### Choosing between algorithms for 2D data

```rust
use scirs2_cluster::{vq::kmeans, density::dbscan, gmm::gaussian_mixture};
use scirs2_cluster::metrics::silhouette_score;

// Try multiple algorithms and compare
let k = 3;

// K-means
let (_, kmeans_labels) = kmeans(data.view(), k, None)?;
let kmeans_score = silhouette_score(data.view(), kmeans_labels.mapv(|x| x as i32).view())?;

// DBSCAN
let dbscan_labels = dbscan(data.view(), 0.5, 5, None)?;
let dbscan_score = silhouette_score(data.view(), dbscan_labels.view())?;

// GMM
let gmm_labels = gaussian_mixture(data.view(), Default::default())?;
let gmm_score = silhouette_score(data.view(), gmm_labels.view())?;

println!("K-means score: {}", kmeans_score);
println!("DBSCAN score: {}", dbscan_score);
println!("GMM score: {}", gmm_score);
```

### Handling large datasets

```rust
use scirs2_cluster::vq::{minibatch_kmeans, parallel_kmeans};

// For very large datasets
if n_samples > 100_000 {
    // Use mini-batch for streaming or memory constraints
    let (centroids, labels) = minibatch_kmeans(data.view(), k, None)?;
} else if n_cores > 4 {
    // Use parallel for multi-core systems
    let (centroids, labels) = parallel_kmeans(data.view(), k, None)?;
}
```

## Best Practices

1. **Always visualize your data first** if possible (2D/3D projection)
2. **Try multiple algorithms** and compare results
3. **Use multiple evaluation metrics** (silhouette, Davies-Bouldin, etc.)
4. **Consider computational constraints** (time, memory)
5. **Validate results** with domain knowledge
6. **Be aware of preprocessing needs** (scaling, normalization)

## References

- K-means: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- DBSCAN: Ester et al. (1996). "A density-based algorithm for discovering clusters"
- HDBSCAN: Campello et al. (2013). "Density-based clustering based on hierarchical density estimates"
- GMM: Dempster et al. (1977). "Maximum likelihood from incomplete data via the EM algorithm"
- Mean Shift: Comaniciu & Meer (2002). "Mean shift: A robust approach toward feature space analysis"
- Spectral: Ng et al. (2001). "On spectral clustering: Analysis and an algorithm"
- Affinity Propagation: Frey & Dueck (2007). "Clustering by passing messages between data points"
- BIRCH: Zhang et al. (1996). "BIRCH: An efficient data clustering method for very large databases"