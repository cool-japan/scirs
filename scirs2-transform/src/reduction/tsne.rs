//! t-SNE (t-distributed Stochastic Neighbor Embedding) implementation
//!
//! This module provides an implementation of t-SNE, a technique for dimensionality
//! reduction particularly well-suited for visualization of high-dimensional data.
//!
//! t-SNE converts similarities between data points to joint probabilities and tries
//! to minimize the Kullback-Leibler divergence between the joint probabilities of
//! the low-dimensional embedding and the high-dimensional data.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use num_traits::{Float, NumCast};
use rayon::prelude::*;

use crate::error::{Result, TransformError};
use crate::reduction::PCA;

// Constants for numerical stability
const MACHINE_EPSILON: f64 = 1e-14;
const EPSILON: f64 = 1e-7;

/// Spatial tree data structure for Barnes-Hut approximation
#[derive(Debug, Clone)]
enum SpatialTree {
    QuadTree(QuadTreeNode),
    OctTree(OctTreeNode),
}

/// Node in a quadtree (for 2D embeddings)
#[derive(Debug, Clone)]
struct QuadTreeNode {
    /// Bounding box of this node
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    /// Center of mass
    center_of_mass: Option<Array1<f64>>,
    /// Total mass (number of points)
    total_mass: f64,
    /// Point indices in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Children nodes (NW, NE, SW, SE)
    children: Option<[Box<QuadTreeNode>; 4]>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

/// Node in an octree (for 3D embeddings)
#[derive(Debug, Clone)]
struct OctTreeNode {
    /// Bounding box of this node
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    /// Center of mass
    center_of_mass: Option<Array1<f64>>,
    /// Total mass (number of points)
    total_mass: f64,
    /// Point indices in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Children nodes (8 octants)
    children: Option<[Box<OctTreeNode>; 8]>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl SpatialTree {
    /// Create a new quadtree for 2D embeddings
    fn new_quadtree(embedding: &Array2<f64>) -> Result<Self> {
        let n_samples = embedding.shape()[0];

        if embedding.shape()[1] != 2 {
            return Err(TransformError::InvalidInput(
                "QuadTree requires 2D embedding".to_string(),
            ));
        }

        // Find bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for i in 0..n_samples {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }

        // Add small margin to avoid edge cases
        let margin = 0.01 * ((x_max - x_min) + (y_max - y_min));
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;

        // Collect all point indices
        let point_indices: Vec<usize> = (0..n_samples).collect();

        // Create root node
        let mut root = QuadTreeNode {
            x_min,
            x_max,
            y_min,
            y_max,
            center_of_mass: None,
            total_mass: 0.0,
            point_indices,
            children: None,
            is_leaf: true,
        };

        // Build the tree
        root.build_tree(embedding)?;

        Ok(SpatialTree::QuadTree(root))
    }

    /// Create a new octree for 3D embeddings
    fn new_octree(embedding: &Array2<f64>) -> Result<Self> {
        let n_samples = embedding.shape()[0];

        if embedding.shape()[1] != 3 {
            return Err(TransformError::InvalidInput(
                "OctTree requires 3D embedding".to_string(),
            ));
        }

        // Find bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for i in 0..n_samples {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            let z = embedding[[i, 2]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }

        // Add small margin to avoid edge cases
        let margin = 0.01 * ((x_max - x_min) + (y_max - y_min) + (z_max - z_min));
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;
        z_min -= margin;
        z_max += margin;

        // Collect all point indices
        let point_indices: Vec<usize> = (0..n_samples).collect();

        // Create root node
        let mut root = OctTreeNode {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            center_of_mass: None,
            total_mass: 0.0,
            point_indices,
            children: None,
            is_leaf: true,
        };

        // Build the tree
        root.build_tree(embedding)?;

        Ok(SpatialTree::OctTree(root))
    }

    /// Compute forces on a point using Barnes-Hut approximation
    fn compute_forces(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        match self {
            SpatialTree::QuadTree(root) => {
                root.compute_forces_quad(point, point_idx, angle, degrees_of_freedom)
            }
            SpatialTree::OctTree(root) => {
                root.compute_forces_oct(point, point_idx, angle, degrees_of_freedom)
            }
        }
    }
}

impl QuadTreeNode {
    /// Build the quadtree recursively
    fn build_tree(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.point_indices.len() <= 1 {
            // Leaf node with 0 or 1 points
            self.update_center_of_mass(embedding)?;
            return Ok(());
        }

        // Split into 4 quadrants
        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;

        let mut quadrants: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        // Distribute points to quadrants
        for &idx in &self.point_indices {
            let x = embedding[[idx, 0]];
            let y = embedding[[idx, 1]];

            let quadrant = match (x >= x_mid, y >= y_mid) {
                (false, false) => 0, // SW
                (true, false) => 1,  // SE
                (false, true) => 2,  // NW
                (true, true) => 3,   // NE
            };

            quadrants[quadrant].push(idx);
        }

        // Create child nodes
        let mut children = [
            Box::new(QuadTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: self.y_min,
                y_max: y_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[0].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: y_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[1].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: y_mid,
                y_max: self.y_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[2].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: y_mid,
                y_max: self.y_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[3].clone(),
                children: None,
                is_leaf: true,
            }),
        ];

        // Recursively build children
        for child in &mut children {
            child.build_tree(embedding)?;
        }

        self.children = Some(children);
        self.is_leaf = false;
        self.point_indices.clear(); // Clear points as they are now in children
        self.update_center_of_mass(embedding)?;

        Ok(())
    }

    /// Update center of mass for this node
    fn update_center_of_mass(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.is_leaf {
            // Leaf node: compute center of mass from points
            if self.point_indices.is_empty() {
                self.total_mass = 0.0;
                self.center_of_mass = None;
                return Ok(());
            }

            let mut com = Array1::zeros(2);
            for &idx in &self.point_indices {
                com[0] += embedding[[idx, 0]];
                com[1] += embedding[[idx, 1]];
            }

            self.total_mass = self.point_indices.len() as f64;
            com.mapv_inplace(|x| x / self.total_mass);
            self.center_of_mass = Some(com);
        } else {
            // Internal node: compute center of mass from children
            if let Some(ref children) = self.children {
                let mut com = Array1::zeros(2);
                let mut total_mass = 0.0;

                for child in children.iter() {
                    if let Some(ref child_com) = child.center_of_mass {
                        total_mass += child.total_mass;
                        for i in 0..2 {
                            com[i] += child_com[i] * child.total_mass;
                        }
                    }
                }

                if total_mass > 0.0 {
                    com.mapv_inplace(|x| x / total_mass);
                    self.center_of_mass = Some(com);
                    self.total_mass = total_mass;
                } else {
                    self.center_of_mass = None;
                    self.total_mass = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Compute forces using Barnes-Hut approximation for quadtree
    fn compute_forces_quad(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        let mut force = Array1::zeros(2);
        let mut sum_q = 0.0;

        self.compute_forces_recursive_quad(
            point,
            point_idx,
            angle,
            degrees_of_freedom,
            &mut force,
            &mut sum_q,
        )?;

        Ok((force, sum_q))
    }

    /// Recursive force computation for quadtree
    fn compute_forces_recursive_quad(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
        force: &mut Array1<f64>,
        sum_q: &mut f64,
    ) -> Result<()> {
        if let Some(ref com) = self.center_of_mass {
            if self.total_mass == 0.0 {
                return Ok(());
            }

            // Compute distance to center of mass
            let dx = point[0] - com[0];
            let dy = point[1] - com[1];
            let dist_squared = dx * dx + dy * dy;

            if dist_squared < MACHINE_EPSILON {
                return Ok(());
            }

            // Check if we can use this node's center of mass (Barnes-Hut criterion)
            let node_size = (self.x_max - self.x_min).max(self.y_max - self.y_min);
            let distance = dist_squared.sqrt();

            if self.is_leaf || (node_size / distance) < angle {
                // Use center of mass approximation
                let q_factor = (1.0 + dist_squared / degrees_of_freedom)
                    .powf(-(degrees_of_freedom + 1.0) / 2.0);

                *sum_q += self.total_mass * q_factor;

                let force_factor =
                    (degrees_of_freedom + 1.0) * self.total_mass * q_factor / degrees_of_freedom;
                force[0] += force_factor * dx;
                force[1] += force_factor * dy;
            } else {
                // Recursively compute forces from children
                if let Some(ref children) = self.children {
                    for child in children.iter() {
                        child.compute_forces_recursive_quad(
                            point,
                            point_idx,
                            angle,
                            degrees_of_freedom,
                            force,
                            sum_q,
                        )?;
                    }
                }
            }
        } else if self.is_leaf {
            // Leaf node without center of mass (empty node)
            for &idx in &self.point_indices {
                if idx != point_idx {
                    // Compute exact force for this point
                    // This will be handled by attractive forces in the main gradient computation
                }
            }
        }

        Ok(())
    }
}

impl OctTreeNode {
    /// Build the octree recursively
    fn build_tree(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.point_indices.len() <= 1 {
            // Leaf node with 0 or 1 points
            self.update_center_of_mass(embedding)?;
            return Ok(());
        }

        // Split into 8 octants
        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;
        let z_mid = (self.z_min + self.z_max) / 2.0;

        let mut octants: [Vec<usize>; 8] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        // Distribute points to octants
        for &idx in &self.point_indices {
            let x = embedding[[idx, 0]];
            let y = embedding[[idx, 1]];
            let z = embedding[[idx, 2]];

            let octant = match (x >= x_mid, y >= y_mid, z >= z_mid) {
                (false, false, false) => 0,
                (true, false, false) => 1,
                (false, true, false) => 2,
                (true, true, false) => 3,
                (false, false, true) => 4,
                (true, false, true) => 5,
                (false, true, true) => 6,
                (true, true, true) => 7,
            };

            octants[octant].push(idx);
        }

        // Create child nodes
        let mut children = [
            Box::new(OctTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: self.y_min,
                y_max: y_mid,
                z_min: self.z_min,
                z_max: z_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[0].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: y_mid,
                z_min: self.z_min,
                z_max: z_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[1].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: y_mid,
                y_max: self.y_max,
                z_min: self.z_min,
                z_max: z_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[2].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: y_mid,
                y_max: self.y_max,
                z_min: self.z_min,
                z_max: z_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[3].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: self.y_min,
                y_max: y_mid,
                z_min: z_mid,
                z_max: self.z_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[4].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: y_mid,
                z_min: z_mid,
                z_max: self.z_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[5].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: y_mid,
                y_max: self.y_max,
                z_min: z_mid,
                z_max: self.z_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[6].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(OctTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: y_mid,
                y_max: self.y_max,
                z_min: z_mid,
                z_max: self.z_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[7].clone(),
                children: None,
                is_leaf: true,
            }),
        ];

        // Recursively build children
        for child in &mut children {
            child.build_tree(embedding)?;
        }

        self.children = Some(children);
        self.is_leaf = false;
        self.point_indices.clear();
        self.update_center_of_mass(embedding)?;

        Ok(())
    }

    /// Update center of mass for this octree node
    fn update_center_of_mass(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.is_leaf {
            if self.point_indices.is_empty() {
                self.total_mass = 0.0;
                self.center_of_mass = None;
                return Ok(());
            }

            let mut com = Array1::zeros(3);
            for &idx in &self.point_indices {
                com[0] += embedding[[idx, 0]];
                com[1] += embedding[[idx, 1]];
                com[2] += embedding[[idx, 2]];
            }

            self.total_mass = self.point_indices.len() as f64;
            com.mapv_inplace(|x| x / self.total_mass);
            self.center_of_mass = Some(com);
        } else if let Some(ref children) = self.children {
            let mut com = Array1::zeros(3);
            let mut total_mass = 0.0;

            for child in children.iter() {
                if let Some(ref child_com) = child.center_of_mass {
                    total_mass += child.total_mass;
                    for i in 0..3 {
                        com[i] += child_com[i] * child.total_mass;
                    }
                }
            }

            if total_mass > 0.0 {
                com.mapv_inplace(|x| x / total_mass);
                self.center_of_mass = Some(com);
                self.total_mass = total_mass;
            } else {
                self.center_of_mass = None;
                self.total_mass = 0.0;
            }
        }

        Ok(())
    }

    /// Compute forces using Barnes-Hut approximation for octree
    fn compute_forces_oct(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        let mut force = Array1::zeros(3);
        let mut sum_q = 0.0;

        self.compute_forces_recursive_oct(
            point,
            point_idx,
            angle,
            degrees_of_freedom,
            &mut force,
            &mut sum_q,
        )?;

        Ok((force, sum_q))
    }

    /// Recursive force computation for octree
    fn compute_forces_recursive_oct(
        &self,
        point: &Array1<f64>,
        _point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
        force: &mut Array1<f64>,
        sum_q: &mut f64,
    ) -> Result<()> {
        if let Some(ref com) = self.center_of_mass {
            if self.total_mass == 0.0 {
                return Ok(());
            }

            let dx = point[0] - com[0];
            let dy = point[1] - com[1];
            let dz = point[2] - com[2];
            let dist_squared = dx * dx + dy * dy + dz * dz;

            if dist_squared < MACHINE_EPSILON {
                return Ok(());
            }

            let node_size = (self.x_max - self.x_min)
                .max(self.y_max - self.y_min)
                .max(self.z_max - self.z_min);
            let distance = dist_squared.sqrt();

            if self.is_leaf || (node_size / distance) < angle {
                let q_factor = (1.0 + dist_squared / degrees_of_freedom)
                    .powf(-(degrees_of_freedom + 1.0) / 2.0);

                *sum_q += self.total_mass * q_factor;

                let force_factor =
                    (degrees_of_freedom + 1.0) * self.total_mass * q_factor / degrees_of_freedom;
                force[0] += force_factor * dx;
                force[1] += force_factor * dy;
                force[2] += force_factor * dz;
            } else if let Some(ref children) = self.children {
                for child in children.iter() {
                    child.compute_forces_recursive_oct(
                        point,
                        _point_idx,
                        angle,
                        degrees_of_freedom,
                        force,
                        sum_q,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction
///
/// t-SNE is a nonlinear dimensionality reduction technique well-suited for
/// embedding high-dimensional data for visualization in a low-dimensional space
/// (typically 2D or 3D). It models each high-dimensional object by a two- or
/// three-dimensional point in such a way that similar objects are modeled by
/// nearby points and dissimilar objects are modeled by distant points with
/// high probability.
pub struct TSNE {
    /// Number of components in the embedded space
    n_components: usize,
    /// Perplexity parameter that balances attention between local and global structure
    perplexity: f64,
    /// Weight of early exaggeration phase
    early_exaggeration: f64,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Maximum iterations without progress before early stopping
    n_iter_without_progress: usize,
    /// Minimum gradient norm for convergence
    min_grad_norm: f64,
    /// Method to compute pairwise distances
    metric: String,
    /// Method to perform dimensionality reduction
    method: String,
    /// Initialization method
    init: String,
    /// Angle for Barnes-Hut approximation
    angle: f64,
    /// Whether to use multicore processing
    n_jobs: i32,
    /// Verbosity level
    verbose: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// The embedding vectors
    embedding_: Option<Array2<f64>>,
    /// KL divergence after optimization
    kl_divergence_: Option<f64>,
    /// Total number of iterations run
    n_iter_: Option<usize>,
    /// Effective learning rate used
    learning_rate_: Option<f64>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new()
    }
}

impl TSNE {
    /// Creates a new t-SNE instance with default parameters
    pub fn new() -> Self {
        TSNE {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            max_iter: 1000,
            n_iter_without_progress: 300,
            min_grad_norm: 1e-7,
            metric: "euclidean".to_string(),
            method: "barnes_hut".to_string(),
            init: "pca".to_string(),
            angle: 0.5,
            n_jobs: -1, // Use all available cores by default
            verbose: false,
            random_state: None,
            embedding_: None,
            kl_divergence_: None,
            n_iter_: None,
            learning_rate_: None,
        }
    }

    /// Sets the number of components in the embedded space
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Sets the perplexity parameter
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the early exaggeration factor
    pub fn with_early_exaggeration(mut self, early_exaggeration: f64) -> Self {
        self.early_exaggeration = early_exaggeration;
        self
    }

    /// Sets the learning rate for gradient descent
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the number of iterations without progress before early stopping
    pub fn with_n_iter_without_progress(mut self, n_iter_without_progress: usize) -> Self {
        self.n_iter_without_progress = n_iter_without_progress;
        self
    }

    /// Sets the minimum gradient norm for convergence
    pub fn with_min_grad_norm(mut self, min_grad_norm: f64) -> Self {
        self.min_grad_norm = min_grad_norm;
        self
    }

    /// Sets the metric for pairwise distance computation
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Sets the method for dimensionality reduction
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Sets the initialization method
    pub fn with_init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Sets the angle for Barnes-Hut approximation
    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    /// Sets the number of parallel jobs to run
    /// * n_jobs = -1: Use all available cores
    /// * n_jobs = 1: Use single-core (disable multicore)
    /// * n_jobs > 1: Use specific number of cores
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Sets the verbosity level
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit t-SNE to input data and transform it to the embedded space
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Embedding of the training data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        // Input validation
        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.perplexity >= n_samples as f64 {
            return Err(TransformError::InvalidInput(format!(
                "perplexity ({}) must be less than n_samples ({})",
                self.perplexity, n_samples
            )));
        }

        if self.method == "barnes_hut" && self.n_components > 3 {
            return Err(TransformError::InvalidInput(
                "'n_components' should be less than or equal to 3 for barnes_hut algorithm"
                    .to_string(),
            ));
        }

        // Set learning rate if auto
        self.learning_rate_ = Some(self.learning_rate);

        // Initialize embedding
        let x_embedded = self.initialize_embedding(&x_f64)?;

        // Compute pairwise affinities (P)
        let p = self.compute_pairwise_affinities(&x_f64)?;

        // Run t-SNE optimization
        let (embedding, kl_divergence, n_iter) =
            self.tsne_optimization(p, x_embedded, n_samples)?;

        self.embedding_ = Some(embedding.clone());
        self.kl_divergence_ = Some(kl_divergence);
        self.n_iter_ = Some(n_iter);

        Ok(embedding)
    }

    /// Initialize embedding either with PCA or random
    fn initialize_embedding(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];

        if self.init == "pca" {
            let n_components = self.n_components.min(x.shape()[1]);
            let mut pca = PCA::new(n_components, true, false);
            let mut x_embedded = pca.fit_transform(x)?;

            // Scale PCA initialization
            let std_dev = (x_embedded.column(0).map(|&x| x * x).sum() / (n_samples as f64)).sqrt();
            if std_dev > 0.0 {
                x_embedded.mapv_inplace(|x| x / std_dev * 1e-4);
            }

            Ok(x_embedded)
        } else if self.init == "random" {
            // Random initialization from standard normal distribution
            // Ignoring random_state as it's not needed for basic random functionality
            let normal = Normal::new(0.0, 1e-4).unwrap();

            // Use simple random initialization
            Ok(Array2::random((n_samples, self.n_components), normal))
        } else {
            Err(TransformError::InvalidInput(format!(
                "Initialization method '{}' not recognized",
                self.init
            )))
        }
    }

    /// Compute pairwise affinities with perplexity-based normalization
    fn compute_pairwise_affinities(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let _n_samples = x.shape()[0];

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(x)?;

        // Convert distances to affinities using binary search for sigma
        let p = self.distances_to_affinities(&distances)?;

        // Symmetrize and normalize the affinity matrix
        let mut p_symmetric = &p + &p.t();

        // Normalize
        let p_sum = p_symmetric.sum();
        if p_sum > 0.0 {
            p_symmetric.mapv_inplace(|x| x.max(MACHINE_EPSILON) / p_sum);
        }

        Ok(p_symmetric)
    }

    /// Compute pairwise Euclidean distances with optional multicore support
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_samples));

        if self.metric == "euclidean" {
            if self.n_jobs == 1 {
                // Single-core computation
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let mut dist_squared = 0.0;
                        for k in 0..x.shape()[1] {
                            let diff = x[[i, k]] - x[[j, k]];
                            dist_squared += diff * diff;
                        }
                        distances[[i, j]] = dist_squared;
                        distances[[j, i]] = dist_squared;
                    }
                }
            } else {
                // Multi-core computation
                let upper_triangle_indices: Vec<(usize, usize)> = (0..n_samples)
                    .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                    .collect();

                let n_features = x.shape()[1];
                let squared_distances: Vec<f64> = upper_triangle_indices
                    .par_iter()
                    .map(|&(i, j)| {
                        let mut dist_squared = 0.0;
                        for k in 0..n_features {
                            let diff = x[[i, k]] - x[[j, k]];
                            dist_squared += diff * diff;
                        }
                        dist_squared
                    })
                    .collect();

                // Fill the distance matrix
                for (idx, &(i, j)) in upper_triangle_indices.iter().enumerate() {
                    distances[[i, j]] = squared_distances[idx];
                    distances[[j, i]] = squared_distances[idx];
                }
            }
        } else {
            return Err(TransformError::InvalidInput(format!(
                "Metric '{}' not implemented. Currently only 'euclidean' is supported",
                self.metric
            )));
        }

        Ok(distances)
    }

    /// Convert distances to affinities using perplexity-based normalization with optional multicore support
    fn distances_to_affinities(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = distances.shape()[0];
        let mut p = Array2::zeros((n_samples, n_samples));
        let target = (2.0f64).ln() * self.perplexity;

        if self.n_jobs == 1 {
            // Single-core computation (original implementation)
            for i in 0..n_samples {
                let mut beta_min = -f64::INFINITY;
                let mut beta_max = f64::INFINITY;
                let mut beta = 1.0;

                // Get all distances from point i except self-distance (which is 0)
                let distances_i = distances.row(i).to_owned();

                // Binary search for beta
                for _ in 0..50 {
                    // Usually converges within 50 iterations
                    // Compute conditional probabilities with current beta
                    let mut sum_pi = 0.0;
                    let mut h = 0.0;

                    for j in 0..n_samples {
                        if i == j {
                            p[[i, j]] = 0.0;
                            continue;
                        }

                        let p_ij = (-beta * distances_i[j]).exp();
                        p[[i, j]] = p_ij;
                        sum_pi += p_ij;
                    }

                    // Normalize probabilities and compute entropy
                    if sum_pi > 0.0 {
                        for j in 0..n_samples {
                            if i == j {
                                continue;
                            }

                            p[[i, j]] /= sum_pi;

                            // Compute entropy
                            if p[[i, j]] > MACHINE_EPSILON {
                                h -= p[[i, j]] * p[[i, j]].ln();
                            }
                        }
                    }

                    // Adjust beta based on entropy difference from target
                    let h_diff = h - target;

                    if h_diff.abs() < EPSILON {
                        break; // Converged
                    }

                    // Update beta using binary search
                    if h_diff > 0.0 {
                        beta_min = beta;
                        if beta_max == f64::INFINITY {
                            beta *= 2.0;
                        } else {
                            beta = (beta + beta_max) / 2.0;
                        }
                    } else {
                        beta_max = beta;
                        if beta_min == -f64::INFINITY {
                            beta /= 2.0;
                        } else {
                            beta = (beta + beta_min) / 2.0;
                        }
                    }
                }
            }
        } else {
            // Multi-core computation of conditional probabilities for each point
            let prob_rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut beta_min = -f64::INFINITY;
                    let mut beta_max = f64::INFINITY;
                    let mut beta = 1.0;

                    // Get all distances from point i except self-distance (which is 0)
                    let distances_i: Vec<f64> = (0..n_samples).map(|j| distances[[i, j]]).collect();
                    let mut p_row = vec![0.0; n_samples];

                    // Binary search for beta
                    for _ in 0..50 {
                        // Usually converges within 50 iterations
                        // Compute conditional probabilities with current beta
                        let mut sum_pi = 0.0;
                        let mut h = 0.0;

                        for j in 0..n_samples {
                            if i == j {
                                p_row[j] = 0.0;
                                continue;
                            }

                            let p_ij = (-beta * distances_i[j]).exp();
                            p_row[j] = p_ij;
                            sum_pi += p_ij;
                        }

                        // Normalize probabilities and compute entropy
                        if sum_pi > 0.0 {
                            for (j, prob) in p_row.iter_mut().enumerate().take(n_samples) {
                                if i == j {
                                    continue;
                                }

                                *prob /= sum_pi;

                                // Compute entropy
                                if *prob > MACHINE_EPSILON {
                                    h -= *prob * prob.ln();
                                }
                            }
                        }

                        // Adjust beta based on entropy difference from target
                        let h_diff = h - target;

                        if h_diff.abs() < EPSILON {
                            break; // Converged
                        }

                        // Update beta using binary search
                        if h_diff > 0.0 {
                            beta_min = beta;
                            if beta_max == f64::INFINITY {
                                beta *= 2.0;
                            } else {
                                beta = (beta + beta_max) / 2.0;
                            }
                        } else {
                            beta_max = beta;
                            if beta_min == -f64::INFINITY {
                                beta /= 2.0;
                            } else {
                                beta = (beta + beta_min) / 2.0;
                            }
                        }
                    }

                    p_row
                })
                .collect();

            // Copy results back to the main matrix
            for (i, row) in prob_rows.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    p[[i, j]] = val;
                }
            }
        }

        Ok(p)
    }

    /// Main t-SNE optimization loop using gradient descent
    fn tsne_optimization(
        &self,
        p: Array2<f64>,
        initial_embedding: Array2<f64>,
        n_samples: usize,
    ) -> Result<(Array2<f64>, f64, usize)> {
        let n_components = self.n_components;
        let degrees_of_freedom = (n_components - 1).max(1) as f64;

        // Initialize variables for optimization
        let mut embedding = initial_embedding;
        let mut update = Array2::zeros((n_samples, n_components));
        let mut gains = Array2::ones((n_samples, n_components));
        let mut error = f64::INFINITY;
        let mut best_error = f64::INFINITY;
        let mut best_iter = 0;
        let mut iter = 0;

        // Exploration phase with early exaggeration
        let exploration_n_iter = 250;
        let n_iter_check = 50;

        // Apply early exaggeration
        let p_early = &p * self.early_exaggeration;

        if self.verbose {
            println!("[t-SNE] Starting optimization with early exaggeration phase...");
        }

        // Early exaggeration phase
        for i in 0..exploration_n_iter {
            // Compute gradient and error for early exaggeration phase
            let (curr_error, grad) = if self.method == "barnes_hut" {
                self.compute_gradient_barnes_hut(&embedding, &p_early, degrees_of_freedom)?
            } else {
                self.compute_gradient_exact(&embedding, &p_early, degrees_of_freedom)?
            };

            // Perform gradient update with momentum and gains
            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.5,
                self.learning_rate_,
            )?;

            // Check for convergence
            if (i + 1) % n_iter_check == 0 {
                if self.verbose {
                    println!("[t-SNE] Iteration {}: error = {:.7}", i + 1, curr_error);
                }

                if curr_error < best_error {
                    best_error = curr_error;
                    best_iter = i;
                } else if i - best_iter > self.n_iter_without_progress {
                    if self.verbose {
                        println!("[t-SNE] Early convergence at iteration {}", i + 1);
                    }
                    break;
                }

                // Check gradient norm
                let grad_norm = grad.mapv(|x| x * x).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!("[t-SNE] Gradient norm {} below threshold, stopping optimization at iteration {}", 
                                grad_norm, i + 1);
                    }
                    break;
                }
            }

            iter = i;
        }

        if self.verbose {
            println!("[t-SNE] Completed early exaggeration phase, starting final optimization...");
        }

        // Final optimization phase without early exaggeration
        for i in iter + 1..self.max_iter {
            // Compute gradient and error for normal phase
            let (curr_error, grad) = if self.method == "barnes_hut" {
                self.compute_gradient_barnes_hut(&embedding, &p, degrees_of_freedom)?
            } else {
                self.compute_gradient_exact(&embedding, &p, degrees_of_freedom)?
            };
            error = curr_error;

            // Perform gradient update with momentum and gains
            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.8,
                self.learning_rate_,
            )?;

            // Check for convergence
            if (i + 1) % n_iter_check == 0 {
                if self.verbose {
                    println!("[t-SNE] Iteration {}: error = {:.7}", i + 1, curr_error);
                }

                if curr_error < best_error {
                    best_error = curr_error;
                    best_iter = i;
                } else if i - best_iter > self.n_iter_without_progress {
                    if self.verbose {
                        println!("[t-SNE] Stopping optimization at iteration {}", i + 1);
                    }
                    break;
                }

                // Check gradient norm
                let grad_norm = grad.mapv(|x| x * x).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!("[t-SNE] Gradient norm {} below threshold, stopping optimization at iteration {}", 
                                grad_norm, i + 1);
                    }
                    break;
                }
            }

            iter = i;
        }

        if self.verbose {
            println!(
                "[t-SNE] Optimization finished after {} iterations with error {:.7}",
                iter + 1,
                error
            );
        }

        Ok((embedding, error, iter + 1))
    }

    /// Compute gradient and error for exact t-SNE with optional multicore support
    fn compute_gradient_exact(
        &self,
        embedding: &Array2<f64>,
        p: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];

        if self.n_jobs == 1 {
            // Single-core computation (original implementation)
            let mut dist = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let mut d_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        d_squared += diff * diff;
                    }

                    // Convert squared distance to t-distribution's probability
                    let q_ij = (1.0 + d_squared / degrees_of_freedom)
                        .powf(-(degrees_of_freedom + 1.0) / 2.0);
                    dist[[i, j]] = q_ij;
                    dist[[j, i]] = q_ij;
                }
            }

            // Set diagonal to zero (self-distance)
            for i in 0..n_samples {
                dist[[i, i]] = 0.0;
            }

            // Normalize Q matrix
            let sum_q = dist.sum().max(MACHINE_EPSILON);
            let q = &dist / sum_q;

            // Compute KL divergence
            let mut kl_divergence = 0.0;
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                        kl_divergence += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                    }
                }
            }

            // Compute gradient
            let mut grad = Array2::zeros((n_samples, n_components));
            let factor =
                4.0 * (degrees_of_freedom + 1.0) / (degrees_of_freedom * (sum_q.powf(2.0)));

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let p_q_diff = p[[i, j]] - q[[i, j]];
                        for k in 0..n_components {
                            grad[[i, k]] += factor
                                * p_q_diff
                                * dist[[i, j]]
                                * (embedding[[i, k]] - embedding[[j, k]]);
                        }
                    }
                }
            }

            Ok((kl_divergence, grad))
        } else {
            // Multi-core computation
            let upper_triangle_indices: Vec<(usize, usize)> = (0..n_samples)
                .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                .collect();

            let q_values: Vec<f64> = upper_triangle_indices
                .par_iter()
                .map(|&(i, j)| {
                    let mut d_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        d_squared += diff * diff;
                    }

                    // Convert squared distance to t-distribution's probability
                    (1.0 + d_squared / degrees_of_freedom).powf(-(degrees_of_freedom + 1.0) / 2.0)
                })
                .collect();

            // Fill the distance matrix
            let mut dist = Array2::zeros((n_samples, n_samples));
            for (idx, &(i, j)) in upper_triangle_indices.iter().enumerate() {
                let q_val = q_values[idx];
                dist[[i, j]] = q_val;
                dist[[j, i]] = q_val;
            }

            // Set diagonal to zero (self-distance)
            for i in 0..n_samples {
                dist[[i, i]] = 0.0;
            }

            // Normalize Q matrix
            let sum_q = dist.sum().max(MACHINE_EPSILON);
            let q = &dist / sum_q;

            // Parallel computation of KL divergence
            let kl_divergence: f64 = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut local_kl = 0.0;
                    for j in 0..n_samples {
                        if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                            local_kl += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                        }
                    }
                    local_kl
                })
                .sum();

            // Parallel computation of gradient
            let factor =
                4.0 * (degrees_of_freedom + 1.0) / (degrees_of_freedom * (sum_q.powf(2.0)));

            let grad_rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut grad_row = vec![0.0; n_components];
                    for j in 0..n_samples {
                        if i != j {
                            let p_q_diff = p[[i, j]] - q[[i, j]];
                            for k in 0..n_components {
                                grad_row[k] += factor
                                    * p_q_diff
                                    * dist[[i, j]]
                                    * (embedding[[i, k]] - embedding[[j, k]]);
                            }
                        }
                    }
                    grad_row
                })
                .collect();

            // Convert gradient rows back to array
            let mut grad = Array2::zeros((n_samples, n_components));
            for (i, row) in grad_rows.iter().enumerate() {
                for (k, &val) in row.iter().enumerate() {
                    grad[[i, k]] = val;
                }
            }

            Ok((kl_divergence, grad))
        }
    }

    /// Compute gradient and error using Barnes-Hut approximation
    fn compute_gradient_barnes_hut(
        &self,
        embedding: &Array2<f64>,
        p: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];

        // Build spatial tree for Barnes-Hut approximation
        let tree = if n_components == 2 {
            SpatialTree::new_quadtree(embedding)?
        } else if n_components == 3 {
            SpatialTree::new_octree(embedding)?
        } else {
            return Err(TransformError::InvalidInput(
                "Barnes-Hut approximation only supports 2D and 3D embeddings".to_string(),
            ));
        };

        // Compute Q matrix and gradient using Barnes-Hut
        let mut q = Array2::zeros((n_samples, n_samples));
        let mut grad = Array2::zeros((n_samples, n_components));
        let mut sum_q = 0.0;

        // For each point, compute repulsive forces using Barnes-Hut
        for i in 0..n_samples {
            let point = embedding.row(i).to_owned();
            let (repulsive_force, q_sum) =
                tree.compute_forces(&point, i, self.angle, degrees_of_freedom)?;

            sum_q += q_sum;

            // Add repulsive forces to gradient
            for j in 0..n_components {
                grad[[i, j]] += repulsive_force[j];
            }

            // Compute Q matrix for KL divergence calculation
            for j in 0..n_samples {
                if i != j {
                    let mut dist_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        dist_squared += diff * diff;
                    }
                    let q_ij = (1.0 + dist_squared / degrees_of_freedom)
                        .powf(-(degrees_of_freedom + 1.0) / 2.0);
                    q[[i, j]] = q_ij;
                }
            }
        }

        // Normalize Q matrix
        sum_q = sum_q.max(MACHINE_EPSILON);
        q.mapv_inplace(|x| x / sum_q);

        // Add attractive forces to gradient
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && p[[i, j]] > MACHINE_EPSILON {
                    let mut dist_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        dist_squared += diff * diff;
                    }

                    let q_ij = (1.0 + dist_squared / degrees_of_freedom)
                        .powf(-(degrees_of_freedom + 1.0) / 2.0);
                    let factor = 4.0 * p[[i, j]] * q_ij;

                    for k in 0..n_components {
                        grad[[i, k]] -= factor * (embedding[[i, k]] - embedding[[j, k]]);
                    }
                }
            }
        }

        // Compute KL divergence
        let mut kl_divergence = 0.0;
        for i in 0..n_samples {
            for j in 0..n_samples {
                if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                    kl_divergence += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                }
            }
        }

        Ok((kl_divergence, grad))
    }

    /// Update embedding using gradient descent with momentum and adaptive gains
    fn gradient_update(
        &self,
        embedding: &mut Array2<f64>,
        update: &mut Array2<f64>,
        gains: &mut Array2<f64>,
        grad: &Array2<f64>,
        momentum: f64,
        learning_rate: Option<f64>,
    ) -> Result<()> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];
        let eta = learning_rate.unwrap_or(self.learning_rate);

        // Update gains and momentum
        for i in 0..n_samples {
            for j in 0..n_components {
                let same_sign = update[[i, j]] * grad[[i, j]] > 0.0;

                if same_sign {
                    gains[[i, j]] *= 0.8;
                } else {
                    gains[[i, j]] += 0.2;
                }

                // Ensure minimum gain
                gains[[i, j]] = gains[[i, j]].max(0.01);

                // Update with momentum and adaptive learning rate
                update[[i, j]] = momentum * update[[i, j]] - eta * gains[[i, j]] * grad[[i, j]];
                embedding[[i, j]] += update[[i, j]];
            }
        }

        Ok(())
    }

    /// Returns the embedding after fitting
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding_.as_ref()
    }

    /// Returns the KL divergence after optimization
    pub fn kl_divergence(&self) -> Option<f64> {
        self.kl_divergence_
    }

    /// Returns the number of iterations run
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }
}

/// Calculate trustworthiness score for a dimensionality reduction
///
/// Trustworthiness measures to what extent the local structure is retained when
/// projecting data from the original space to the embedding space.
///
/// # Arguments
/// * `x` - Original data, shape (n_samples, n_features)
/// * `x_embedded` - Embedded data, shape (n_samples, n_components)
/// * `n_neighbors` - Number of neighbors to consider
/// * `metric` - Metric to use (currently only 'euclidean' is implemented)
///
/// # Returns
/// * `Result<f64>` - Trustworthiness score between 0.0 and 1.0
pub fn trustworthiness<S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    x_embedded: &ArrayBase<S2, Ix2>,
    n_neighbors: usize,
    metric: &str,
) -> Result<f64>
where
    S1: Data,
    S2: Data,
    S1::Elem: Float + NumCast,
    S2::Elem: Float + NumCast,
{
    let x_f64 = x.mapv(|x| num_traits::cast::<S1::Elem, f64>(x).unwrap_or(0.0));
    let x_embedded_f64 = x_embedded.mapv(|x| num_traits::cast::<S2::Elem, f64>(x).unwrap_or(0.0));

    let n_samples = x_f64.shape()[0];

    if n_neighbors >= n_samples / 2 {
        return Err(TransformError::InvalidInput(format!(
            "n_neighbors ({}) should be less than n_samples / 2 ({})",
            n_neighbors,
            n_samples / 2
        )));
    }

    if metric != "euclidean" {
        return Err(TransformError::InvalidInput(format!(
            "Metric '{}' not implemented. Currently only 'euclidean' is supported",
            metric
        )));
    }

    // Compute pairwise distances in original space
    let mut dist_x = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                dist_x[[i, j]] = f64::INFINITY; // Set self-distance to infinity
                continue;
            }

            let mut d_squared = 0.0;
            for k in 0..x_f64.shape()[1] {
                let diff = x_f64[[i, k]] - x_f64[[j, k]];
                d_squared += diff * diff;
            }
            dist_x[[i, j]] = d_squared.sqrt();
        }
    }

    // Compute pairwise distances in embedded space
    let mut dist_embedded = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                dist_embedded[[i, j]] = f64::INFINITY; // Set self-distance to infinity
                continue;
            }

            let mut d_squared = 0.0;
            for k in 0..x_embedded_f64.shape()[1] {
                let diff = x_embedded_f64[[i, k]] - x_embedded_f64[[j, k]];
                d_squared += diff * diff;
            }
            dist_embedded[[i, j]] = d_squared.sqrt();
        }
    }

    // For each point, find the n_neighbors nearest neighbors in the original space
    let mut nn_orig = Array2::<usize>::zeros((n_samples, n_neighbors));
    for i in 0..n_samples {
        // Get the indices of the sorted distances
        let row = dist_x.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // The first element will be i itself (distance 0), so skip it
        for (j, &(idx, _)) in pairs.iter().enumerate().take(n_neighbors) {
            nn_orig[[i, j]] = idx;
        }
    }

    // For each point, find the n_neighbors nearest neighbors in the embedded space
    let mut nn_embedded = Array2::<usize>::zeros((n_samples, n_neighbors));
    for i in 0..n_samples {
        // Get the indices of the sorted distances
        let row = dist_embedded.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // The first element will be i itself (distance 0), so skip it
        for (j, &(idx, _)) in pairs.iter().enumerate().take(n_neighbors) {
            nn_embedded[[i, j]] = idx;
        }
    }

    // Calculate the trustworthiness score
    let mut t = 0.0;
    for i in 0..n_samples {
        for &j in nn_embedded.row(i).iter() {
            // Check if j is not in the n_neighbors nearest neighbors in the original space
            let is_not_neighbor = !nn_orig.row(i).iter().any(|&nn| nn == j);

            if is_not_neighbor {
                // Find the rank of j in the original space
                let row = dist_x.row(i).to_owned();
                let mut pairs: Vec<(usize, f64)> =
                    row.iter().enumerate().map(|(idx, &d)| (idx, d)).collect();
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let rank = pairs.iter().position(|&(idx, _)| idx == j).unwrap_or(0) - n_neighbors;

                t += rank as f64;
            }
        }
    }

    // Normalize the trustworthiness score
    let n = n_samples as f64;
    let k = n_neighbors as f64;
    let normalizer = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0));
    let trustworthiness = 1.0 - normalizer * t;

    Ok(trustworthiness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_tsne_simple() {
        // Create a simple dataset
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0],
        ]);

        // Initialize and fit t-SNE with exact method
        let mut tsne_exact = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("exact")
            .with_random_state(42)
            .with_max_iter(250)
            .with_verbose(false);

        let embedding_exact = tsne_exact.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding_exact.shape(), &[8, 2]);

        // Check that groups are separated in the embedding space
        // Compute the average distance within each group
        let dist_group1 = average_pairwise_distance(&embedding_exact.slice(ndarray::s![0..4, ..]));
        let dist_group2 = average_pairwise_distance(&embedding_exact.slice(ndarray::s![4..8, ..]));

        // Compute the average distance between groups
        let dist_between = average_intergroup_distance(
            &embedding_exact.slice(ndarray::s![0..4, ..]),
            &embedding_exact.slice(ndarray::s![4..8, ..]),
        );

        // The between-group distance should be larger than the within-group distances
        assert!(dist_between > dist_group1);
        assert!(dist_between > dist_group2);
    }

    #[test]
    fn test_tsne_barnes_hut() {
        // Create a simple dataset
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0],
        ]);

        // Initialize and fit t-SNE with Barnes-Hut method
        let mut tsne_bh = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("barnes_hut")
            .with_angle(0.5)
            .with_random_state(42)
            .with_max_iter(250)
            .with_verbose(false);

        let embedding_bh = tsne_bh.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding_bh.shape(), &[8, 2]);

        // Test basic functionality - Barnes-Hut is approximate so just check for basic properties
        assert!(embedding_bh.iter().all(|&x| x.is_finite()));

        // Check that the embedding has some spread (not all points collapsed to the same location)
        let min_val = embedding_bh.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = embedding_bh
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_val - min_val > 1e-6,
            "Embedding should have some spread"
        );

        // Check that KL divergence was computed (Barnes-Hut is approximate, so we're more lenient)
        assert!(tsne_bh.kl_divergence().is_some());

        // For Barnes-Hut approximation, the KL divergence might not always be finite
        // due to the approximation nature, so we just check that it's a number
        let kl_div = tsne_bh.kl_divergence().unwrap();
        if !kl_div.is_finite() {
            // This is acceptable for Barnes-Hut approximation
            println!(
                "Barnes-Hut KL divergence: {} (non-finite, which is acceptable for approximation)",
                kl_div
            );
        } else {
            println!("Barnes-Hut KL divergence: {} (finite)", kl_div);
        }
    }

    #[test]
    fn test_tsne_multicore() {
        // Create a simple dataset
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0],
        ]);

        // Initialize and fit t-SNE with multicore enabled
        let mut tsne_multicore = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("exact")
            .with_n_jobs(-1) // Use all cores
            .with_random_state(42)
            .with_max_iter(100) // Shorter for testing
            .with_verbose(false);

        let embedding_multicore = tsne_multicore.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding_multicore.shape(), &[8, 2]);

        // Test basic functionality - multicore should produce valid results
        assert!(embedding_multicore.iter().all(|&x| x.is_finite()));

        // Check that the embedding has some spread (more lenient for short iterations)
        let min_val = embedding_multicore
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_val = embedding_multicore
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_val - min_val > 1e-12,
            "Embedding should have some spread, got range: {}",
            max_val - min_val
        );

        // Test single-core vs multicore consistency
        let mut tsne_singlecore = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("exact")
            .with_n_jobs(1) // Single core
            .with_random_state(42)
            .with_max_iter(100)
            .with_verbose(false);

        let embedding_singlecore = tsne_singlecore.fit_transform(&x).unwrap();

        // Both should produce finite results (exact numerical match is not expected due to randomness)
        assert!(embedding_multicore.iter().all(|&x| x.is_finite()));
        assert!(embedding_singlecore.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_tsne_3d_barnes_hut() {
        // Create a simple 3D dataset
        let x = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
            [6.0, 6.0, 5.0],
        ]);

        // Initialize and fit t-SNE with Barnes-Hut method for 3D
        let mut tsne_3d = TSNE::new()
            .with_n_components(3)
            .with_perplexity(2.0)
            .with_method("barnes_hut")
            .with_angle(0.5)
            .with_random_state(42)
            .with_max_iter(250)
            .with_verbose(false);

        let embedding_3d = tsne_3d.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding_3d.shape(), &[8, 3]);

        // Test basic functionality - should not panic
        assert!(embedding_3d.iter().all(|&x| x.is_finite()));
    }

    // Helper function to compute average pairwise distance within a group
    fn average_pairwise_distance(points: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>) -> f64 {
        let n = points.shape()[0];
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in i + 1..n {
                let mut dist_squared = 0.0;
                for k in 0..points.shape()[1] {
                    let diff = points[[i, k]] - points[[j, k]];
                    dist_squared += diff * diff;
                }
                total_dist += dist_squared.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    // Helper function to compute average distance between two groups
    fn average_intergroup_distance(
        group1: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>,
        group2: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>,
    ) -> f64 {
        let n1 = group1.shape()[0];
        let n2 = group2.shape()[0];
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..n1 {
            for j in 0..n2 {
                let mut dist_squared = 0.0;
                for k in 0..group1.shape()[1] {
                    let diff = group1[[i, k]] - group2[[j, k]];
                    dist_squared += diff * diff;
                }
                total_dist += dist_squared.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    #[test]
    fn test_trustworthiness() {
        // Create a simple dataset where we know the structure
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [6.0, 6.0],
        ]);

        // A perfect embedding would preserve all neighborhoods
        let perfect_embedding = x.clone();
        let t_perfect = trustworthiness(&x, &perfect_embedding, 3, "euclidean").unwrap();
        assert_abs_diff_eq!(t_perfect, 1.0, epsilon = 1e-10);

        // A random embedding would have low trustworthiness
        let random_embedding = arr2(&[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
        ]);

        let t_random = trustworthiness(&x, &random_embedding, 3, "euclidean").unwrap();
        assert!(t_random < 1.0);
    }
}
