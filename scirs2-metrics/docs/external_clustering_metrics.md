# External Clustering Metrics

This document provides a detailed guide to the external clustering metrics implemented in the `scirs2-metrics` crate. External metrics compare clustering results against a known ground truth (reference labeling) to evaluate clustering performance.

## Overview

External clustering metrics evaluate the quality of a clustering algorithm by comparing its results with a ground truth labeling. Unlike internal metrics (which only use the clustered data itself), external metrics require knowledge of the true class assignments.

These metrics are particularly useful for:
- Comparing different clustering algorithms
- Tuning algorithm hyperparameters
- Understanding specific strengths and weaknesses of clustering approaches
- Benchmarking against reference implementations

## Available Metrics

The `scirs2-metrics` crate implements the following external clustering metrics:

### 1. Adjusted Rand Index (ARI)

**Function**: `adjusted_rand_index`

```rust
pub fn adjusted_rand_index<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
```

**Description**: 
The Adjusted Rand Index measures the similarity between two clusterings by considering all pairs of samples and counting pairs that are assigned to the same or different clusters in both clusterings. The raw Rand index is then adjusted for chance, creating a metric that:
- Has an expected value of 0 for random labeling
- Is bounded above by 1 (perfect agreement)
- Can be negative (indicating agreement worse than random chance)

**Mathematical Definition**:
Let a be the number of pairs of elements that are in the same cluster in both clusterings, b be the number of pairs that are in different clusters in both clusterings, c and d be the numbers of pairs that are in the same cluster in one clustering but in different clusters in the other.

The Rand index is: RI = (a + b) / (a + b + c + d)

The ARI is the adjusted-for-chance form:
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

**Usage Example**:
```rust
use ndarray::array;
use scirs2_metrics::clustering::adjusted_rand_index;

let labels_true = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
let labels_pred = array![0, 0, 1, 1, 2, 1, 2, 2, 2];

let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
println!("Adjusted Rand Index: {}", ari);
```

**Notes**:
- The function handles different labeling schemes (e.g., if the true labels are [0,0,1,1] and predicted labels are [1,1,0,0], it will still recognize the clusters correctly)
- Returns a value in range [-1, 1] where 1 means perfect match
- Recommended as a general-purpose metric for clustering evaluation

### 2. Normalized Mutual Information (NMI)

**Function**: `normalized_mutual_info_score`

```rust
pub fn normalized_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
) -> Result<f64>
```

**Description**:
Normalized Mutual Information measures how much information is shared between the true and predicted clusterings. It quantifies the "amount of information" obtained about one clustering by observing the other clustering, normalized to scale between 0 and 1.

**Mathematical Definition**:
NMI(U, V) = 2 * MI(U, V) / [H(U) + H(V)]

Where:
- MI(U, V) is the mutual information between clusterings U and V
- H(U) and H(V) are the entropies of the clusterings

**Normalization methods**:
- "arithmetic": MI / ((H(labels_true) + H(labels_pred)) / 2)
- "geometric": MI / sqrt(H(labels_true) * H(labels_pred))
- "min": MI / min(H(labels_true), H(labels_pred))
- "max": MI / max(H(labels_true), H(labels_pred))

**Usage Example**:
```rust
use ndarray::array;
use scirs2_metrics::clustering::normalized_mutual_info_score;

let labels_true = array![0, 0, 1, 1, 2, 2];
let labels_pred = array![0, 0, 0, 1, 1, 1];

let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
println!("Normalized Mutual Information: {}", nmi);
```

**Notes**:
- Returns a value in range [0, 1] with 1 indicating perfect match
- Not adjusted for chance (unlike ARI)
- The choice of normalization method can affect the results, especially with unbalanced clusters
- Works well when comparing clusterings with different numbers of clusters

### 3. Adjusted Mutual Information (AMI)

**Function**: `adjusted_mutual_info_score`

```rust
pub fn adjusted_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
) -> Result<f64>
```

**Description**:
Adjusted Mutual Information is similar to NMI but adjusted for chance. This means the expected value of AMI for random labeling is 0, regardless of the number of clusters, making it more robust when comparing clusterings with different numbers of clusters.

**Mathematical Definition**:
AMI(U, V) = [MI(U, V) - E(MI)] / [max(H(U), H(V)) - E(MI)]

Where:
- MI(U, V) is the mutual information between clusterings U and V
- E(MI) is the expected mutual information between random clusterings
- H(U) and H(V) are the entropies of the clusterings

**Usage Example**:
```rust
use ndarray::array;
use scirs2_metrics::clustering::adjusted_mutual_info_score;

let labels_true = array![0, 0, 1, 1, 2, 2];
let labels_pred = array![0, 0, 0, 1, 1, 1];

let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
println!("Adjusted Mutual Information: {}", ami);
```

**Notes**:
- Returns a value in range [0, 1] with 1 indicating perfect match
- Adjusted for chance, making it more robust for comparing different clustering algorithms
- Particularly useful when comparing clusterings with different numbers of clusters
- Same normalization options as NMI

### 4. Homogeneity, Completeness, and V-measure

**Function**: `homogeneity_completeness_v_measure`

```rust
pub fn homogeneity_completeness_v_measure<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    beta: f64,
) -> Result<(f64, f64, f64)>
```

**Description**:
This function returns three related metrics that help analyze the quality of clustering:

1. **Homogeneity**: A clustering satisfies homogeneity if all of its clusters contain only data points that are members of a single class.
2. **Completeness**: A clustering satisfies completeness if all the data points that are members of a given class are elements of the same cluster.
3. **V-measure**: The harmonic mean of homogeneity and completeness, controlled by the beta parameter.

**Mathematical Definition**:
- Homogeneity = 1 - H(C|K) / H(C)
- Completeness = 1 - H(K|C) / H(K)
- V-measure = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)

Where:
- H(C|K) is the conditional entropy of the classes given the cluster assignments
- H(C) is the entropy of the classes
- H(K|C) is the conditional entropy of the clusters given the class assignments
- H(K) is the entropy of the clusters

**Usage Example**:
```rust
use ndarray::array;
use scirs2_metrics::clustering::homogeneity_completeness_v_measure;

let labels_true = array![0, 0, 1, 1, 2, 2];
let labels_pred = array![0, 0, 0, 1, 1, 1];

let (homogeneity, completeness, v_measure) =
    homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();
    
println!("Homogeneity: {}", homogeneity);
println!("Completeness: {}", completeness);
println!("V-measure: {}", v_measure);
```

**Notes**:
- All three metrics have values in range [0, 1] with 1 indicating perfect match
- Specific use cases:
  - Homogeneity is high when each cluster contains only members of a single class
  - Completeness is high when all members of a class are assigned to the same cluster
- The beta parameter controls the weight given to homogeneity vs. completeness:
  - beta < 1: More weight on homogeneity
  - beta = 1: Equal weight (standard V-measure)
  - beta > 1: More weight on completeness

### 5. Fowlkes-Mallows Score

**Function**: `fowlkes_mallows_score`

```rust
pub fn fowlkes_mallows_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
```

**Description**:
The Fowlkes-Mallows score is the geometric mean of the pairwise precision and recall. It can be interpreted as the geometric mean of the precision and recall when viewing clustering as a series of decisions on pairs of elements.

**Mathematical Definition**:
FMI = TP / sqrt((TP + FP) * (TP + FN))

Where:
- TP (True Positives): Pairs of points that are in the same cluster in both clusterings
- FP (False Positives): Pairs of points that are in the same cluster in the predicted clustering but not in the true clustering
- FN (False Negatives): Pairs of points that are in the same cluster in the true clustering but not in the predicted clustering

**Usage Example**:
```rust
use ndarray::array;
use scirs2_metrics::clustering::fowlkes_mallows_score;

let labels_true = array![0, 0, 1, 1, 2, 2];
let labels_pred = array![0, 0, 0, 1, 1, 1];

let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
println!("Fowlkes-Mallows Score: {}", score);
```

**Notes**:
- Returns a value in range [0, 1] with 1 indicating perfect match
- Value of 0 indicates no agreement between clusterings
- Not adjusted for chance (unlike ARI and AMI)
- Based on the concept of precision and recall from information retrieval
- Particularly useful when you're interested in a balanced view of precision and recall

## Comparing Metrics

Each external metric has its strengths and is appropriate for different scenarios:

| Metric | Strengths | Best Use Cases | Adjusted for Chance |
|--------|-----------|---------------|---------------------|
| ARI | Intuitive (based on counting pairs), accounts for chance agreement | General-purpose evaluation, baseline metric | Yes |
| NMI | Information-theoretic approach, works well with many clusters | When cluster sizes are different | No |
| AMI | Combines benefits of NMI with chance adjustment | Comparing clusterings with different numbers of clusters | Yes |
| Homogeneity & Completeness | Separate metrics for different aspects of clustering | When you need to understand if clusters are pure or if classes are preserved | No |
| V-measure | Configurable balance between homogeneity and completeness | When you need a single metric with adjustable emphasis on homogeneity vs. completeness | No |
| Fowlkes-Mallows | Based on precision and recall | Gives equal importance to precision and recall | No |

## Implementation Details

All metrics in this module:

1. **Accept Generic Types**: Can handle true and predicted labels of different types
2. **Type-Safe**: Leverage Rust's type system to ensure correct usage
3. **Error Handling**: Return `Result` type for proper error handling
4. **Handle Label Permutations**: Recognize that cluster labels are arbitrary
5. **Numerical Stability**: Include checks for potential numerical issues

## Performance Considerations

For large datasets, consider:
- Using parallelization through the `scirs2-core::parallel` module
- Sampling data for rapid evaluation during iterative algorithm development
- Using the most efficient metric for your needs (ARI is generally fastest)
- Using the batch processing capabilities in the metrics module

## References

1. Hubert, L. & Arabie, P. (1985). "Comparing partitions". Journal of Classification.
2. Strehl, A. & Ghosh, J. (2002). "Cluster Ensembles – A Knowledge Reuse Framework for Combining Multiple Partitions".
3. Vinh, N. X., Epps, J., & Bailey, J. (2010). "Information theoretic measures for clusterings comparison".
4. Rosenberg, A., & Hirschberg, J. (2007). "V-Measure: A conditional entropy-based external cluster evaluation measure".
5. Fowlkes, E. B., & Mallows, C. L. (1983). "A method for comparing two hierarchical clusterings".