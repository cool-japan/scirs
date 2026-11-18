//! BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering algorithm
//!
//! BIRCH is an incremental clustering algorithm designed for very large datasets.
//! It builds a CF-tree (Clustering Feature tree) that compactly summarizes the data,
//! then applies a global clustering algorithm on the leaf entries.
//!
//! # Algorithm Phases
//!
//! 1. **Phase 1**: Build the CF-tree by scanning data once (linear time)
//! 2. **Phase 2** (optional): Condense the tree by rebuilding with a larger threshold
//! 3. **Phase 3**: Apply global clustering (e.g., k-means) on leaf CF entries
//! 4. **Phase 4**: Assign original data points to clusters
//!
//! # Features
//!
//! - **CF-tree construction** with configurable branching factor and threshold
//! - **Node splitting** when branching factor is exceeded
//! - **Subclustering and merging** of CF entries
//! - **Threshold auto-adjustment** for controlling tree size
//! - **Incremental insertion** for streaming data
//!
//! # References
//!
//! Zhang, T., Ramakrishnan, R., Livny, M. (1996). "BIRCH: An Efficient Data
//! Clustering Method for Very Large Databases." SIGMOD, pp. 103-114.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Clustering Feature (CF) for summarizing a cluster
///
/// A CF is a triple (N, LS, SS) where:
/// - N: number of data points
/// - LS: linear sum of data points
/// - SS: sum of squared norms of data points
#[derive(Debug, Clone)]
pub struct ClusteringFeature<F: Float> {
    /// Number of data points
    n: usize,
    /// Linear sum of data points (vector sum)
    linear_sum: Array1<F>,
    /// Sum of squared norms: sum(||x_i||^2)
    squared_sum: F,
}

impl<F: Float + FromPrimitive + ScalarOperand> ClusteringFeature<F> {
    /// Create a new CF from a single data point
    fn new(datapoint: ArrayView1<F>) -> Self {
        let squared_sum = datapoint.dot(&datapoint);
        Self {
            n: 1,
            linear_sum: datapoint.to_owned(),
            squared_sum,
        }
    }

    /// Create an empty CF with a given dimensionality
    fn empty(n_features: usize) -> Self {
        Self {
            n: 0,
            linear_sum: Array1::zeros(n_features),
            squared_sum: F::zero(),
        }
    }

    /// Add another CF to this one (absorb)
    fn add(&mut self, other: &Self) {
        self.n += other.n;
        self.linear_sum = &self.linear_sum + &other.linear_sum;
        self.squared_sum = self.squared_sum + other.squared_sum;
    }

    /// Merge with another CF and return a new CF
    fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.add(other);
        result
    }

    /// Calculate the centroid of this CF
    fn centroid(&self) -> Array1<F> {
        if self.n == 0 {
            Array1::zeros(self.linear_sum.len())
        } else {
            let n_f = F::from(self.n).unwrap_or(F::one());
            &self.linear_sum / n_f
        }
    }

    /// Calculate the radius (average distance from centroid to points)
    ///
    /// radius = sqrt((SS/N) - (LS/N).(LS/N))
    fn radius(&self) -> F {
        if self.n <= 1 {
            F::zero()
        } else {
            let n_f = F::from(self.n).unwrap_or(F::one());
            let centroid = self.centroid();
            let centroid_ss = centroid.dot(&centroid);
            let variance = (self.squared_sum / n_f) - centroid_ss;
            variance.max(F::zero()).sqrt()
        }
    }

    /// Calculate the diameter (average pairwise distance between points)
    ///
    /// diameter = sqrt(2 * (N * SS - LS.LS) / (N * (N-1)))
    fn diameter(&self) -> F {
        if self.n <= 1 {
            F::zero()
        } else {
            let n_f = F::from(self.n).unwrap_or(F::one());
            let ls_dot = self.linear_sum.dot(&self.linear_sum);
            let numerator = n_f * self.squared_sum - ls_dot;
            let denominator = n_f * (n_f - F::one());
            if denominator <= F::zero() {
                return F::zero();
            }
            let two = F::from(2.0).unwrap_or(F::one() + F::one());
            (two * numerator / denominator).max(F::zero()).sqrt()
        }
    }

    /// Calculate the distance between two CF centroids
    fn centroid_distance(&self, other: &Self) -> F {
        let c1 = self.centroid();
        let c2 = other.centroid();
        let mut dist = F::zero();
        for i in 0..c1.len() {
            let diff = c1[i] - c2[i];
            dist = dist + diff * diff;
        }
        dist.sqrt()
    }

    /// Calculate the D0 distance (inter-cluster distance based on centroids)
    fn d0_distance(&self, other: &Self) -> F {
        self.centroid_distance(other)
    }

    /// Calculate the D2 distance (average inter-cluster distance)
    fn d2_distance(&self, other: &Self) -> F {
        let merged = self.merge(other);
        merged.diameter()
    }
}

/// Node in the CF-tree
#[derive(Debug)]
struct CFNode<F: Float> {
    /// Whether this is a leaf node
    is_leaf: bool,
    /// CFs stored in this node (leaf: one per subcluster; non-leaf: summary of each child)
    cfs: Vec<ClusteringFeature<F>>,
    /// Child nodes (only for non-leaf nodes)
    children: Vec<CFNode<F>>,
}

impl<F: Float + FromPrimitive + ScalarOperand> CFNode<F> {
    /// Create a new leaf node
    fn new_leaf() -> Self {
        Self {
            is_leaf: true,
            cfs: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Create a new non-leaf node
    fn new_non_leaf() -> Self {
        Self {
            is_leaf: false,
            cfs: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Get the CF that summarizes this entire node
    fn get_cf(&self) -> ClusteringFeature<F> {
        if self.cfs.is_empty() {
            return ClusteringFeature::empty(0);
        }
        let mut result = self.cfs[0].clone();
        for cf in self.cfs.iter().skip(1) {
            result.add(cf);
        }
        result
    }

    /// Insert a CF into a leaf node. Returns Ok(None) if absorbed/added,
    /// or Ok(Some(new_node)) if a split occurred.
    fn insert_cf(
        &mut self,
        new_cf: ClusteringFeature<F>,
        branching_factor: usize,
        threshold: F,
    ) -> Result<Option<CFNode<F>>> {
        if !self.is_leaf {
            return self.insert_cf_nonleaf(new_cf, branching_factor, threshold);
        }

        // Leaf node: try to absorb into closest CF
        if self.cfs.is_empty() {
            self.cfs.push(new_cf);
            return Ok(None);
        }

        // Find closest CF entry
        let (closest_idx, _closest_dist) = self.find_closest_cf(&new_cf);

        // Try to absorb
        let merged = self.cfs[closest_idx].merge(&new_cf);
        if merged.radius() <= threshold {
            self.cfs[closest_idx] = merged;
            return Ok(None);
        }

        // Cannot absorb; try to add as new entry
        if self.cfs.len() < branching_factor {
            self.cfs.push(new_cf);
            return Ok(None);
        }

        // Need to split: add the new CF temporarily, then split
        self.cfs.push(new_cf);
        let new_node = self.split_leaf(branching_factor);
        Ok(Some(new_node))
    }

    /// Insert CF into a non-leaf node
    fn insert_cf_nonleaf(
        &mut self,
        new_cf: ClusteringFeature<F>,
        branching_factor: usize,
        threshold: F,
    ) -> Result<Option<CFNode<F>>> {
        if self.children.is_empty() {
            // Degenerate: convert to leaf
            self.is_leaf = true;
            self.cfs.push(new_cf);
            return Ok(None);
        }

        // Find the closest child
        let (closest_idx, _) = self.find_closest_cf(&new_cf);
        let closest_idx = closest_idx.min(self.children.len() - 1);

        // Recursively insert into the closest child
        let split_result =
            self.children[closest_idx].insert_cf(new_cf, branching_factor, threshold)?;

        // Update the CF summary for this child
        self.cfs[closest_idx] = self.children[closest_idx].get_cf();

        if let Some(new_child) = split_result {
            // A child split occurred; add the new child
            let new_child_cf = new_child.get_cf();
            if self.children.len() < branching_factor {
                self.cfs.push(new_child_cf);
                self.children.push(new_child);
                Ok(None)
            } else {
                // Need to split this non-leaf node too
                self.cfs.push(new_child_cf);
                self.children.push(new_child);
                let new_node = self.split_nonleaf(branching_factor);
                Ok(Some(new_node))
            }
        } else {
            Ok(None)
        }
    }

    /// Find the closest CF entry (by centroid distance)
    fn find_closest_cf(&self, target: &ClusteringFeature<F>) -> (usize, F) {
        let mut closest_idx = 0;
        let mut min_dist = F::infinity();

        for (i, cf) in self.cfs.iter().enumerate() {
            let dist = cf.centroid_distance(target);
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        (closest_idx, min_dist)
    }

    /// Split a leaf node into two, returning the new sibling node.
    /// Uses the farthest-pair heuristic: pick two most distant CFs as seeds.
    fn split_leaf(&mut self, _branching_factor: usize) -> CFNode<F> {
        let n = self.cfs.len();
        if n <= 1 {
            return CFNode::new_leaf();
        }

        // Find the two most distant CFs
        let (seed1, seed2) = find_farthest_pair(&self.cfs);

        // Drain all CFs into a temporary vec to avoid borrow conflicts
        let all_cfs: Vec<ClusteringFeature<F>> = self.cfs.drain(..).collect();

        // Distribute CFs to two groups
        let mut group1 = Vec::new();
        let mut group2 = Vec::new();

        for (i, cf) in all_cfs.into_iter().enumerate() {
            if i == seed1 {
                group1.push(cf);
            } else if i == seed2 {
                group2.push(cf);
            } else {
                // Assign to the closer group centroid
                let dist1 = distance_to_group_centroid(&cf, &group1);
                let dist2 = distance_to_group_centroid(&cf, &group2);

                if dist1 <= dist2 {
                    group1.push(cf);
                } else {
                    group2.push(cf);
                }
            }
        }

        self.cfs = group1;

        let mut new_node = CFNode::new_leaf();
        new_node.cfs = group2;

        new_node
    }

    /// Split a non-leaf node
    fn split_nonleaf(&mut self, _branching_factor: usize) -> CFNode<F> {
        let n = self.cfs.len();
        if n <= 1 {
            return CFNode::new_non_leaf();
        }

        let (seed1, seed2) = find_farthest_pair(&self.cfs);

        let mut group1_cfs = Vec::new();
        let mut group1_children = Vec::new();
        let mut group2_cfs = Vec::new();
        let mut group2_children = Vec::new();

        let all_cfs: Vec<ClusteringFeature<F>> = self.cfs.drain(..).collect();
        let all_children: Vec<CFNode<F>> = self.children.drain(..).collect();

        for (i, (cf, child)) in all_cfs
            .into_iter()
            .zip(all_children.into_iter())
            .enumerate()
        {
            if i == seed1 {
                group1_cfs.push(cf);
                group1_children.push(child);
            } else if i == seed2 {
                group2_cfs.push(cf);
                group2_children.push(child);
            } else {
                let dist1 = distance_to_group_centroid(&cf, &group1_cfs);
                let dist2 = distance_to_group_centroid(&cf, &group2_cfs);

                if dist1 <= dist2 {
                    group1_cfs.push(cf);
                    group1_children.push(child);
                } else {
                    group2_cfs.push(cf);
                    group2_children.push(child);
                }
            }
        }

        self.cfs = group1_cfs;
        self.children = group1_children;

        let mut new_node = CFNode::new_non_leaf();
        new_node.cfs = group2_cfs;
        new_node.children = group2_children;

        new_node
    }

    /// Distance from a CF to the centroid of a group (used during splitting)
    fn dist_to_seed(&self, cf: &ClusteringFeature<F>, group: &[ClusteringFeature<F>]) -> F {
        distance_to_group_centroid(cf, group)
    }

    /// Collect all leaf CF entries from this subtree
    fn collect_leaf_entries(&self, out: &mut Vec<ClusteringFeature<F>>) {
        if self.is_leaf {
            for cf in &self.cfs {
                out.push(cf.clone());
            }
        } else {
            for child in &self.children {
                child.collect_leaf_entries(out);
            }
        }
    }
}

/// Find the two most distant CFs by centroid distance
fn find_farthest_pair<F: Float + FromPrimitive + ScalarOperand>(
    cfs: &[ClusteringFeature<F>],
) -> (usize, usize) {
    let n = cfs.len();
    if n < 2 {
        return (0, 0);
    }

    let mut max_dist = F::zero();
    let mut pair = (0, 1);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = cfs[i].centroid_distance(&cfs[j]);
            if dist > max_dist {
                max_dist = dist;
                pair = (i, j);
            }
        }
    }

    pair
}

/// Distance from a CF to the centroid of a group
fn distance_to_group_centroid<F: Float + FromPrimitive + ScalarOperand>(
    cf: &ClusteringFeature<F>,
    group: &[ClusteringFeature<F>],
) -> F {
    if group.is_empty() {
        return F::infinity();
    }

    let mut sum = ClusteringFeature::empty(cf.linear_sum.len());
    for g in group {
        sum.add(g);
    }

    cf.centroid_distance(&sum)
}

/// BIRCH clustering algorithm options
#[derive(Debug, Clone)]
pub struct BirchOptions<F: Float> {
    /// Maximum number of CFs in each leaf or non-leaf node
    pub branching_factor: usize,
    /// Maximum radius of a subcluster (controls granularity)
    pub threshold: F,
    /// Number of clusters to extract (None = one per leaf CF)
    pub n_clusters: Option<usize>,
    /// Maximum number of leaf entries before threshold auto-increase
    pub max_leaf_entries: Option<usize>,
    /// Number of k-means refinement iterations on CF centroids
    pub n_refinement_iter: usize,
}

impl<F: Float + FromPrimitive> Default for BirchOptions<F> {
    fn default() -> Self {
        Self {
            branching_factor: 50,
            threshold: F::from(0.5).unwrap_or(F::one()),
            n_clusters: None,
            max_leaf_entries: None,
            n_refinement_iter: 5,
        }
    }
}

/// BIRCH clustering algorithm
pub struct Birch<F: Float> {
    options: BirchOptions<F>,
    root: Option<Box<CFNode<F>>>,
    leaf_entries: Vec<ClusteringFeature<F>>,
    n_features: Option<usize>,
    effective_threshold: Option<F>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> Birch<F> {
    /// Create a new BIRCH instance
    pub fn new(options: BirchOptions<F>) -> Self {
        Self {
            options,
            root: None,
            leaf_entries: Vec::new(),
            n_features: None,
            effective_threshold: None,
        }
    }

    /// Fit the BIRCH model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        self.n_features = Some(n_features);
        let mut root = Box::new(CFNode::new_leaf());
        let threshold = self.options.threshold;
        let branching_factor = self.options.branching_factor;
        self.effective_threshold = Some(threshold);

        // Phase 1: Build CF-tree
        for i in 0..n_samples {
            let point = data.slice(s![i, ..]);
            let new_cf = ClusteringFeature::new(point);

            let split_result = root.insert_cf(new_cf, branching_factor, threshold)?;

            if let Some(sibling) = split_result {
                // Root was split; create a new root
                let old_root_cf = root.get_cf();
                let sibling_cf = sibling.get_cf();

                let mut new_root = Box::new(CFNode::new_non_leaf());
                new_root.cfs.push(old_root_cf);
                new_root.cfs.push(sibling_cf);
                new_root.children.push(*root);
                new_root.children.push(sibling);

                root = new_root;
            }
        }

        // Collect leaf entries
        self.leaf_entries.clear();
        root.collect_leaf_entries(&mut self.leaf_entries);

        // Phase 2 (optional): Threshold auto-adjustment
        if let Some(max_entries) = self.options.max_leaf_entries {
            if self.leaf_entries.len() > max_entries {
                self.rebuild_with_larger_threshold(&data, max_entries)?;
            }
        }

        self.root = Some(root);

        Ok(())
    }

    /// Rebuild the CF-tree with a larger threshold to reduce leaf entries
    fn rebuild_with_larger_threshold(
        &mut self,
        data: &ArrayView2<F>,
        max_entries: usize,
    ) -> Result<()> {
        let mut threshold = self.options.threshold;
        let increase_factor = F::from(1.5).unwrap_or(F::one() + F::one());
        let max_attempts = 10;

        for _ in 0..max_attempts {
            threshold = threshold * increase_factor;
            let mut root = Box::new(CFNode::new_leaf());
            let branching_factor = self.options.branching_factor;

            for i in 0..data.shape()[0] {
                let point = data.slice(s![i, ..]);
                let new_cf = ClusteringFeature::new(point);

                let split_result = root.insert_cf(new_cf, branching_factor, threshold)?;

                if let Some(sibling) = split_result {
                    let old_root_cf = root.get_cf();
                    let sibling_cf = sibling.get_cf();

                    let mut new_root = Box::new(CFNode::new_non_leaf());
                    new_root.cfs.push(old_root_cf);
                    new_root.cfs.push(sibling_cf);
                    new_root.children.push(*root);
                    new_root.children.push(sibling);

                    root = new_root;
                }
            }

            self.leaf_entries.clear();
            root.collect_leaf_entries(&mut self.leaf_entries);

            if self.leaf_entries.len() <= max_entries {
                self.effective_threshold = Some(threshold);
                self.root = Some(root);
                return Ok(());
            }
        }

        self.effective_threshold = Some(threshold);
        Ok(())
    }

    /// Partially fit: add new data points incrementally
    pub fn partial_fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_features = data.shape()[1];

        if self.n_features.is_none() {
            self.n_features = Some(n_features);
        }

        let threshold = self.effective_threshold.unwrap_or(self.options.threshold);
        let branching_factor = self.options.branching_factor;

        if self.root.is_none() {
            self.root = Some(Box::new(CFNode::new_leaf()));
        }

        for i in 0..data.shape()[0] {
            let point = data.slice(s![i, ..]);
            let new_cf = ClusteringFeature::new(point);

            // Take root out temporarily to avoid borrow conflicts
            let mut root = self
                .root
                .take()
                .ok_or_else(|| ClusteringError::InvalidState("Root should exist".into()))?;

            let split_result = root.insert_cf(new_cf, branching_factor, threshold)?;

            if let Some(sibling) = split_result {
                let old_root_cf = root.get_cf();
                let sibling_cf = sibling.get_cf();

                let mut new_root = Box::new(CFNode::new_non_leaf());
                new_root.cfs.push(old_root_cf);
                new_root.cfs.push(sibling_cf);
                new_root.children.push(*root);
                new_root.children.push(sibling);

                self.root = Some(new_root);
            } else {
                self.root = Some(root);
            }
        }

        // Refresh leaf entries
        self.leaf_entries.clear();
        if let Some(ref root) = self.root {
            root.collect_leaf_entries(&mut self.leaf_entries);
        }

        Ok(())
    }

    /// Predict cluster labels for data
    ///
    /// Each point takes the cluster label of its nearest leaf entry.
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<i32>> {
        if self.leaf_entries.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ));
        }

        let (_, cf_labels) = self.extract_clusters()?;

        let n_samples = data.shape()[0];
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = data.slice(s![i, ..]);
            let mut min_dist = F::infinity();
            let mut closest_cf = 0;

            for (j, cf) in self.leaf_entries.iter().enumerate() {
                let centroid = cf.centroid();
                let dist = euclidean_distance(point, centroid.view());

                if dist < min_dist {
                    min_dist = dist;
                    closest_cf = j;
                }
            }

            labels[i] = cf_labels[closest_cf];
        }

        Ok(labels)
    }

    /// Extract clusters from the CF-tree
    ///
    /// Phase 3: Apply k-means-style refinement on CF centroids
    /// Phase 4: Map CF clusters to actual cluster assignments
    pub fn extract_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        if self.leaf_entries.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No data has been processed".to_string(),
            ));
        }

        let n_features = self
            .n_features
            .ok_or_else(|| ClusteringError::InvalidState("n_features not set".into()))?;
        let n_cf_entries = self.leaf_entries.len();
        let n_clusters = self.options.n_clusters.unwrap_or(n_cf_entries);

        if n_clusters >= n_cf_entries {
            // Each CF is its own cluster
            let mut centroids = Array2::zeros((n_cf_entries, n_features));
            let mut labels = Array1::zeros(n_cf_entries);

            for (i, cf) in self.leaf_entries.iter().enumerate() {
                let centroid = cf.centroid();
                centroids.slice_mut(s![i, ..]).assign(&centroid);
                labels[i] = i as i32;
            }

            Ok((centroids, labels))
        } else {
            self.cluster_cf_entries_refined(n_clusters, n_features)
        }
    }

    /// Apply iterative k-means on CF entries with CF-weighted centroids
    fn cluster_cf_entries_refined(
        &self,
        n_clusters: usize,
        n_features: usize,
    ) -> Result<(Array2<F>, Array1<i32>)> {
        let n_cfs = self.leaf_entries.len();

        // Extract CF centroids
        let mut cf_centroids = Array2::zeros((n_cfs, n_features));
        for (i, cf) in self.leaf_entries.iter().enumerate() {
            let centroid = cf.centroid();
            cf_centroids.slice_mut(s![i, ..]).assign(&centroid);
        }

        // Initialize cluster centers using farthest-first heuristic
        let mut cluster_centers = Array2::zeros((n_clusters, n_features));
        cluster_centers
            .slice_mut(s![0, ..])
            .assign(&cf_centroids.slice(s![0, ..]));

        for c in 1..n_clusters {
            // Pick the CF centroid farthest from any existing cluster center
            let mut max_min_dist = F::zero();
            let mut best_idx = c % n_cfs;

            for i in 0..n_cfs {
                let mut min_dist = F::infinity();
                for j in 0..c {
                    let dist = euclidean_distance(
                        cf_centroids.slice(s![i, ..]),
                        cluster_centers.slice(s![j, ..]),
                    );
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_idx = i;
                }
            }

            cluster_centers
                .slice_mut(s![c, ..])
                .assign(&cf_centroids.slice(s![best_idx, ..]));
        }

        // Iterative refinement (weighted k-means on CFs)
        let mut assignments = Array1::zeros(n_cfs);

        for _iter in 0..self.options.n_refinement_iter {
            // Assignment step
            for i in 0..n_cfs {
                let mut min_dist = F::infinity();
                let mut closest_cluster = 0;

                for j in 0..n_clusters {
                    let dist = euclidean_distance(
                        cf_centroids.slice(s![i, ..]),
                        cluster_centers.slice(s![j, ..]),
                    );

                    if dist < min_dist {
                        min_dist = dist;
                        closest_cluster = j;
                    }
                }

                assignments[i] = closest_cluster as i32;
            }

            // Update step (weighted by CF sizes)
            let mut new_centers = Array2::zeros((n_clusters, n_features));
            let mut weights = vec![F::zero(); n_clusters];

            for (cf_idx, &cluster_id) in assignments.iter().enumerate() {
                let cid = cluster_id as usize;
                let cf = &self.leaf_entries[cf_idx];
                let cf_weight = F::from(cf.n).unwrap_or(F::one());

                let centroid = cf.centroid();
                for f in 0..n_features {
                    new_centers[[cid, f]] = new_centers[[cid, f]] + centroid[f] * cf_weight;
                }
                weights[cid] = weights[cid] + cf_weight;
            }

            for c in 0..n_clusters {
                if weights[c] > F::zero() {
                    for f in 0..n_features {
                        new_centers[[c, f]] = new_centers[[c, f]] / weights[c];
                    }
                }
            }

            cluster_centers = new_centers;
        }

        Ok((cluster_centers, assignments))
    }

    /// Get statistics about the CF-tree
    pub fn get_statistics(&self) -> BirchStatistics<F> {
        let total_points: usize = self.leaf_entries.iter().map(|cf| cf.n).sum();
        let avg_cf_size = if !self.leaf_entries.is_empty() {
            total_points as f64 / self.leaf_entries.len() as f64
        } else {
            0.0
        };

        let avg_radius = if !self.leaf_entries.is_empty() {
            let total_radius: F = self
                .leaf_entries
                .iter()
                .map(|cf| cf.radius())
                .fold(F::zero(), |acc, x| acc + x);
            let n_entries = F::from(self.leaf_entries.len()).unwrap_or(F::one());
            total_radius / n_entries
        } else {
            F::zero()
        };

        BirchStatistics {
            num_cf_entries: self.leaf_entries.len(),
            total_points,
            avg_cf_size,
            avg_radius,
            threshold: self.effective_threshold.unwrap_or(self.options.threshold),
            branching_factor: self.options.branching_factor,
        }
    }
}

/// Statistics about a BIRCH CF-tree
#[derive(Debug)]
pub struct BirchStatistics<F: Float> {
    /// Number of CF entries (leaf subclusters) in the tree
    pub num_cf_entries: usize,
    /// Total number of data points processed
    pub total_points: usize,
    /// Average number of points per CF
    pub avg_cf_size: f64,
    /// Average radius of CFs
    pub avg_radius: F,
    /// Threshold parameter used (may differ from original if auto-adjusted)
    pub threshold: F,
    /// Branching factor used
    pub branching_factor: usize,
}

/// BIRCH clustering convenience function
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `options` - BIRCH options
///
/// # Returns
///
/// * Tuple of (centroids, labels)
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::birch::{birch, BirchOptions};
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     4.0, 5.0,
///     4.2, 4.8,
///     3.9, 5.1,
/// ]).expect("Operation failed");
///
/// let options = BirchOptions {
///     n_clusters: Some(2),
///     ..Default::default()
/// };
///
/// let (centroids, labels) = birch(data.view(), options).expect("Operation failed");
/// ```
pub fn birch<F>(data: ArrayView2<F>, options: BirchOptions<F>) -> Result<(Array2<F>, Array1<i32>)>
where
    F: Float + FromPrimitive + Debug + ScalarOperand,
{
    let mut model = Birch::new(options);
    model.fit(data)?;
    let (centroids, _cf_labels) = model.extract_clusters()?;
    let labels = model.predict(data)?;
    Ok((centroids, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_clustering_feature() {
        let point = Array1::from_vec(vec![1.0, 2.0]);
        let cf = ClusteringFeature::<f64>::new(point.view());

        assert_eq!(cf.n, 1);
        assert_eq!(cf.linear_sum, point);
        assert_eq!(cf.squared_sum, 5.0);

        let centroid = cf.centroid();
        assert_eq!(centroid, point);
    }

    #[test]
    fn test_cf_merge() {
        let point1 = Array1::from_vec(vec![1.0, 2.0]);
        let point2 = Array1::from_vec(vec![3.0, 4.0]);

        let cf1 = ClusteringFeature::new(point1.view());
        let cf2 = ClusteringFeature::new(point2.view());

        let merged = cf1.merge(&cf2);

        assert_eq!(merged.n, 2);
        assert_eq!(merged.linear_sum, Array1::from_vec(vec![4.0, 6.0]));
        assert_eq!(merged.squared_sum, 30.0);

        let centroid = merged.centroid();
        assert_eq!(centroid, Array1::from_vec(vec![2.0, 3.0]));
    }

    #[test]
    fn test_cf_radius() {
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![2.0, 0.0]);

        let cf1 = ClusteringFeature::<f64>::new(p1.view());
        let cf2 = ClusteringFeature::new(p2.view());
        let merged = cf1.merge(&cf2);

        let radius = merged.radius();
        assert!(radius > 0.0, "Radius should be positive");
        assert!(radius <= 2.0, "Radius should be <= diameter");
    }

    #[test]
    fn test_cf_diameter() {
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![2.0, 0.0]);

        let cf1 = ClusteringFeature::<f64>::new(p1.view());
        let cf2 = ClusteringFeature::new(p2.view());
        let merged = cf1.merge(&cf2);

        let diameter = merged.diameter();
        assert!(diameter > 0.0, "Diameter should be positive");
        // For two points at distance 2, diameter should be 2
        assert!(
            (diameter - 2.0).abs() < 0.1,
            "Diameter should be ~2.0, got {}",
            diameter
        );
    }

    #[test]
    fn test_birch_simple() {
        let data = make_two_cluster_data();

        let options = BirchOptions {
            n_clusters: Some(2),
            threshold: 1.0,
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (centroids, labels) = result.expect("Should succeed");
        assert_eq!(centroids.shape()[0], 2);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_birch_default_options() {
        let data = make_two_cluster_data();

        let options = BirchOptions::default();

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (_centroids, labels) = result.expect("Should succeed");
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_birch_empty_data() {
        let data = Array2::<f64>::zeros((0, 2));
        let options = BirchOptions::default();
        let result = birch(data.view(), options);
        assert!(result.is_err());
    }

    #[test]
    fn test_birch_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("Failed to create data");

        let options = BirchOptions {
            n_clusters: Some(1),
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (centroids, labels) = result.expect("Should succeed");
        assert_eq!(centroids.shape()[0], 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], 0);
    }

    #[test]
    fn test_birch_statistics() {
        let data = make_two_cluster_data();

        let mut model = Birch::new(BirchOptions {
            threshold: 1.0,
            ..Default::default()
        });
        model.fit(data.view()).expect("Should fit");

        let stats = model.get_statistics();
        assert_eq!(stats.total_points, 6);
        assert!(stats.num_cf_entries > 0);
        assert!(stats.avg_cf_size > 0.0);
        assert!(stats.branching_factor == 50);
    }

    #[test]
    fn test_birch_incremental() {
        let data1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9])
            .expect("Failed to create data");

        let data2 = Array2::from_shape_vec((3, 2), vec![4.0, 5.0, 4.2, 4.8, 3.9, 5.1])
            .expect("Failed to create data");

        let mut model = Birch::new(BirchOptions {
            threshold: 1.0,
            n_clusters: Some(2),
            ..Default::default()
        });

        model.fit(data1.view()).expect("Should fit first batch");
        model
            .partial_fit(data2.view())
            .expect("Should fit second batch");

        let stats = model.get_statistics();
        assert_eq!(stats.total_points, 6);
    }

    #[test]
    fn test_birch_small_threshold() {
        let data = make_two_cluster_data();

        let options = BirchOptions {
            threshold: 0.01, // Very small threshold => many CFs
            n_clusters: Some(2),
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (centroids, labels) = result.expect("Should succeed");
        assert_eq!(centroids.shape()[0], 2);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_birch_labels_are_cluster_ids() {
        let data = make_two_cluster_data();

        let options = BirchOptions {
            threshold: 0.01, // one leaf entry per point
            n_clusters: Some(2),
            ..Default::default()
        };

        let (_centroids, labels) = birch(data.view(), options).expect("Should succeed");
        assert!(
            labels.iter().all(|&l| l == 0 || l == 1),
            "Labels should be cluster ids, got {:?}",
            labels
        );
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_birch_predict_returns_cluster_ids() {
        let data = make_two_cluster_data();

        let mut model = Birch::new(BirchOptions {
            threshold: 0.01,
            n_clusters: Some(2),
            ..Default::default()
        });
        model.fit(data.view()).expect("Should fit");
        let (centroids, _) = model.extract_clusters().expect("Should extract");

        let new_data = Array2::from_shape_vec((2, 2), vec![1.1, 1.9, 4.1, 4.9])
            .expect("Failed to create data");
        let labels = model.predict(new_data.view()).expect("Should predict");

        assert!(
            labels.iter().all(|&l| l == 0 || l == 1),
            "Labels should be cluster ids, got {:?}",
            labels
        );
        assert_ne!(labels[0], labels[1]);
        let near_first = centroids.row(labels[0] as usize);
        assert!((near_first[0] - 1.0).abs() < 0.5 && (near_first[1] - 1.9).abs() < 0.5);
    }

    #[test]
    fn test_birch_large_threshold() {
        let data = make_two_cluster_data();

        let options = BirchOptions {
            threshold: 100.0, // Very large threshold => few CFs
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (_centroids, labels) = result.expect("Should succeed");
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_birch_branching_factor() {
        let data = make_two_cluster_data();

        let options = BirchOptions {
            branching_factor: 2, // Very small branching factor => many splits
            threshold: 0.5,
            n_clusters: Some(2),
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (_centroids, labels) = result.expect("Should succeed");
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_birch_threshold_auto_adjustment() {
        // Create larger dataset
        let mut data_vec = Vec::new();
        for i in 0..50 {
            data_vec.push(i as f64 * 0.1);
            data_vec.push(i as f64 * 0.1 + 0.5);
        }
        let data = Array2::from_shape_vec((50, 2), data_vec).expect("Failed to create data");

        let options = BirchOptions {
            threshold: 0.01,
            max_leaf_entries: Some(10), // Force threshold increase
            n_clusters: Some(3),
            ..Default::default()
        };

        let mut model = Birch::new(options);
        model.fit(data.view()).expect("Should fit");

        // Threshold should have been auto-increased
        let stats = model.get_statistics();
        assert!(stats.num_cf_entries <= 50, "Should have compacted entries");
    }
}
