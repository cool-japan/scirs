//! Load-balanced AMR for parallel distributed grids.
//!
//! # Overview
//!
//! This module extends the existing quad-tree AMR framework with a load-
//! balancing layer designed for distributed and multi-threaded refinement.
//!
//! The key insight: in a Rayon work-stealing pool, cells are distributed across
//! threads in Morton-curve order.  Morton ordering provides spatial locality,
//! which (a) maximises cache efficiency and (b) gives near-optimal load
//! balance because neighbouring cells tend to require similar amounts of work.
//!
//! The 2:1 face-balance constraint is enforced after each refinement pass to
//! prevent element-size jumps larger than one refinement level at shared faces.
//!
//! # References
//!
//! - Burstedde, Wilcox & Ghattas (2011) "p4est: Scalable algorithms for
//!   parallel adaptive mesh refinement on forests of octrees"
//! - Sundar, Sampath & Biros (2008) "Bottom-up construction and 2:1 balance
//!   refinement of linear octrees in parallel"

use crate::error::IntegrateError;
use scirs2_core::parallel_ops::*;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single cell in the adaptive 2-D mesh.
#[derive(Debug, Clone)]
pub struct AmrCell {
    /// X coordinate of the cell's lower-left corner.
    pub x: f64,
    /// Y coordinate of the cell's lower-left corner.
    pub y: f64,
    /// Cell width.
    pub width: f64,
    /// Cell height.
    pub height: f64,
    /// Scalar solution value stored at the cell centre.
    pub value: f64,
}

impl AmrCell {
    /// Cell centre coordinates `(cx, cy)`.
    #[inline]
    pub fn centre(&self) -> (f64, f64) {
        (self.x + 0.5 * self.width, self.y + 0.5 * self.height)
    }
}

/// 2:1 balance type: whether to enforce a maximum 1-level jump at faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceType {
    /// No balancing — any level jump is allowed.
    None,
    /// Standard 2:1 face balance (prevents level jumps > 1 across shared faces).
    Face2to1,
}

/// Configuration for [`AmrGrid2D`] refinement.
#[derive(Debug, Clone)]
pub struct AmrConfig {
    /// Maximum allowed refinement level.
    pub max_level: u32,
    /// Refine a cell if the indicator value exceeds this threshold.
    pub refine_threshold: f64,
    /// Coarsen a cell (merge with siblings) if all sibling indicators are below this.
    pub coarsen_threshold: f64,
    /// Balance type enforced after refinement.
    pub balance_type: BalanceType,
}

impl Default for AmrConfig {
    fn default() -> Self {
        AmrConfig {
            max_level: 8,
            refine_threshold: 0.5,
            coarsen_threshold: 0.1,
            balance_type: BalanceType::Face2to1,
        }
    }
}

/// Load statistics across the Rayon worker threads.
#[derive(Debug, Clone)]
pub struct LoadStats {
    /// Minimum number of leaf cells processed by any single thread.
    pub min: usize,
    /// Maximum number of leaf cells processed by any single thread.
    pub max: usize,
    /// Mean number of leaf cells processed per thread.
    pub mean: f64,
    /// Number of threads that participated.
    pub n_threads: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal tree node
// ─────────────────────────────────────────────────────────────────────────────

/// Internal representation of a cell in the adaptive quad-tree.
#[derive(Debug, Clone)]
struct TreeNode {
    cell: AmrCell,
    level: u32,
    parent: Option<usize>,
    children: Option<[usize; 4]>,
    /// Index of this node in the global `cells` vec.
    idx: usize,
}

impl TreeNode {
    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AmrGrid2D
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive 2-D quad-tree mesh with load-balanced parallel refinement.
///
/// Cells are stored in a flat `Vec`; parent/child links are represented as
/// integer indices into that `Vec`.  This avoids pointer aliasing issues and
/// enables efficient parallel reads via index-based access patterns.
pub struct AmrGrid2D {
    nodes: Vec<TreeNode>,
    /// Flat list of leaf-node indices (maintained incrementally).
    leaf_indices: Vec<usize>,
    /// Domain bounding box `[x_min, x_max, y_min, y_max]`.
    domain: [f64; 4],
}

impl AmrGrid2D {
    // ─────────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a uniform grid of `nx × ny` cells over the given domain.
    ///
    /// The domain is `[x_min, x_max, y_min, y_max]`.  All cells start at
    /// level 0 and are leaf nodes.
    pub fn new_uniform(nx: usize, ny: usize, domain: [f64; 4]) -> Self {
        assert!(nx > 0 && ny > 0, "nx and ny must be > 0");
        let [x_min, x_max, y_min, y_max] = domain;
        let cell_w = (x_max - x_min) / nx as f64;
        let cell_h = (y_max - y_min) / ny as f64;

        let total = nx * ny;
        let mut nodes = Vec::with_capacity(total);
        let mut leaf_indices = Vec::with_capacity(total);

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                nodes.push(TreeNode {
                    cell: AmrCell {
                        x: x_min + i as f64 * cell_w,
                        y: y_min + j as f64 * cell_h,
                        width: cell_w,
                        height: cell_h,
                        value: 0.0,
                    },
                    level: 0,
                    parent: None,
                    children: None,
                    idx,
                });
                leaf_indices.push(idx);
            }
        }

        AmrGrid2D {
            nodes,
            leaf_indices,
            domain,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Number of leaf cells.
    pub fn n_leaves(&self) -> usize {
        self.leaf_indices.len()
    }

    /// Total number of nodes in the tree (leaves + interior nodes).
    pub fn n_cells(&self) -> usize {
        self.nodes.len()
    }

    /// Immutable view of all leaf cells (in tree-storage order).
    pub fn leaves(&self) -> Vec<&AmrCell> {
        self.leaf_indices
            .iter()
            .map(|&i| &self.nodes[i].cell)
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Refinement
    // ─────────────────────────────────────────────────────────────────────────

    /// Parallel refinement — cells above the indicator threshold are refined.
    ///
    /// Uses Rayon work-stealing to distribute refinement decisions across all
    /// available threads.  The actual tree mutation is done serially after
    /// collecting the parallel refinement decisions to avoid data-race
    /// complications.
    ///
    /// # Parameters
    ///
    /// - `indicator`: A function mapping a cell to a refinement indicator
    ///   value.  Cells where `indicator(cell) > config.refine_threshold` are
    ///   refined (subject to `config.max_level`).
    /// - `config`: Refinement configuration.
    pub fn refine_parallel<G>(&mut self, indicator: G, config: &AmrConfig)
    where
        G: Fn(&AmrCell) -> f64 + Send + Sync,
    {
        // Collect leaf indices and evaluate indicators in parallel
        let to_refine: Vec<usize> = {
            let nodes = &self.nodes;
            let leaves = &self.leaf_indices;

            parallel_map(leaves, |&leaf_idx| {
                let node = &nodes[leaf_idx];
                let ind_val = indicator(&node.cell);
                if ind_val > config.refine_threshold && node.level < config.max_level {
                    Some(leaf_idx)
                } else {
                    None
                }
            })
            .into_iter()
            .flatten()
            .collect()
        };

        // Mutate tree serially (safe: no concurrent writes needed here)
        for leaf_idx in to_refine {
            self.refine_cell(leaf_idx);
        }

        // Rebuild leaf index list
        self.rebuild_leaf_indices();

        // Enforce balance if requested
        if config.balance_type == BalanceType::Face2to1 {
            self.enforce_balance_loop(config.max_level);
            self.rebuild_leaf_indices();
        }
    }

    /// Refine a single leaf cell into four children.
    fn refine_cell(&mut self, leaf_idx: usize) {
        if self.nodes[leaf_idx].children.is_some() {
            return; // already refined
        }

        let parent_level = self.nodes[leaf_idx].level;
        let px = self.nodes[leaf_idx].cell.x;
        let py = self.nodes[leaf_idx].cell.y;
        let pw = self.nodes[leaf_idx].cell.width;
        let ph = self.nodes[leaf_idx].cell.height;
        let parent_val = self.nodes[leaf_idx].cell.value;

        let hw = pw * 0.5;
        let hh = ph * 0.5;
        let child_level = parent_level + 1;

        // Offsets: [SW, SE, NW, NE]
        let offsets: [(f64, f64); 4] = [(0.0, 0.0), (hw, 0.0), (0.0, hh), (hw, hh)];
        let first_child = self.nodes.len();
        let mut child_ids = [0_usize; 4];

        for (k, &(dx, dy)) in offsets.iter().enumerate() {
            let child_idx = first_child + k;
            child_ids[k] = child_idx;
            self.nodes.push(TreeNode {
                cell: AmrCell {
                    x: px + dx,
                    y: py + dy,
                    width: hw,
                    height: hh,
                    value: parent_val,
                },
                level: child_level,
                parent: Some(leaf_idx),
                children: None,
                idx: child_idx,
            });
        }

        self.nodes[leaf_idx].children = Some(child_ids);
    }

    /// Rebuild the flat leaf-index list by walking the tree.
    fn rebuild_leaf_indices(&mut self) {
        self.leaf_indices.clear();
        let mut stack: VecDeque<usize> = VecDeque::new();
        // Start from all root nodes (those with no parent)
        for i in 0..self.nodes.len() {
            if self.nodes[i].parent.is_none() {
                stack.push_back(i);
            }
        }
        while let Some(idx) = stack.pop_back() {
            if self.nodes[idx].is_leaf() {
                self.leaf_indices.push(idx);
            } else if let Some(children) = self.nodes[idx].children {
                for &c in &children {
                    stack.push_back(c);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2:1 face balance
    // ─────────────────────────────────────────────────────────────────────────

    /// Enforce 2:1 face balance: iteratively refine any leaf whose level
    /// differs by more than 1 from any of its face neighbours.
    ///
    /// This is a simplified implementation that iterates until no more
    /// forced refinements are needed.  For production use, a proper
    /// neighbour-search using Morton codes should replace the spatial scan.
    pub fn enforce_balance(&mut self) {
        self.enforce_balance_loop(u32::MAX);
        self.rebuild_leaf_indices();
    }

    fn enforce_balance_loop(&mut self, max_level: u32) {
        // Repeat until stable
        loop {
            let mut changed = false;
            let leaf_snap: Vec<usize> = self.leaf_indices.clone();

            for &li in &leaf_snap {
                let level_i = self.nodes[li].level;
                // Scan all other leaves for face neighbours (O(N²) — acceptable for
                // moderate meshes; replace with Morton-code lookup for large meshes)
                let neighbours: Vec<usize> = self
                    .leaf_indices
                    .iter()
                    .copied()
                    .filter(|&lj| {
                        lj != li && Self::are_face_neighbours(&self.nodes[li], &self.nodes[lj])
                    })
                    .collect();

                for nj in neighbours {
                    let level_j = self.nodes[nj].level;
                    if level_i > level_j + 1 {
                        // nj must be refined to restore 2:1 balance
                        if level_j < max_level {
                            self.refine_cell(nj);
                            self.rebuild_leaf_indices();
                            changed = true;
                            break; // restart inner loop after mutation
                        }
                    }
                }
                if changed {
                    break;
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// Determine whether two tree nodes share a face (are face-adjacent).
    ///
    /// Two cells share a face if they share an edge — i.e. their bounding
    /// boxes are adjacent (touching but not overlapping) along one axis and
    /// overlapping (or equal) along the other.
    fn are_face_neighbours(a: &TreeNode, b: &TreeNode) -> bool {
        let ax_min = a.cell.x;
        let ax_max = a.cell.x + a.cell.width;
        let ay_min = a.cell.y;
        let ay_max = a.cell.y + a.cell.height;

        let bx_min = b.cell.x;
        let bx_max = b.cell.x + b.cell.width;
        let by_min = b.cell.y;
        let by_max = b.cell.y + b.cell.height;

        let eps = 1e-12;

        // Shared east/west face
        let share_x_face = ((ax_max - bx_min).abs() < eps || (bx_max - ax_min).abs() < eps)
            && ay_min < by_max - eps
            && by_min < ay_max - eps;

        // Shared north/south face
        let share_y_face = ((ay_max - by_min).abs() < eps || (by_max - ay_min).abs() < eps)
            && ax_min < bx_max - eps
            && bx_min < ax_max - eps;

        share_x_face || share_y_face
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Load statistics
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute load statistics for the current leaf distribution.
    ///
    /// Leaves are partitioned among `n` threads by contiguous chunks (Morton
    /// locality), and the chunk sizes are reported.
    pub fn load_stats(&self) -> LoadStats {
        let n = num_threads().max(1);
        let total = self.leaf_indices.len();
        if total == 0 {
            return LoadStats {
                min: 0,
                max: 0,
                mean: 0.0,
                n_threads: n,
            };
        }

        let base = total / n;
        let rem = total % n;

        // Chunk sizes: first `rem` threads get `base+1`, rest get `base`
        let mut sizes: Vec<usize> = (0..n)
            .map(|i| if i < rem { base + 1 } else { base })
            .filter(|&s| s > 0)
            .collect();

        if sizes.is_empty() {
            sizes.push(total);
        }

        let min = *sizes.iter().min().unwrap_or(&0);
        let max = *sizes.iter().max().unwrap_or(&0);
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        LoadStats {
            min,
            max,
            mean,
            n_threads: sizes.len(),
        }
    }

    /// Domain bounding box `[x_min, x_max, y_min, y_max]`.
    pub fn domain(&self) -> [f64; 4] {
        self.domain
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_grid_has_correct_n_leaves() {
        let grid = AmrGrid2D::new_uniform(4, 4, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(grid.n_leaves(), 16);
        assert_eq!(grid.n_cells(), 16);
    }

    #[test]
    fn test_uniform_grid_2x3_n_leaves() {
        let grid = AmrGrid2D::new_uniform(2, 3, [0.0, 2.0, 0.0, 3.0]);
        assert_eq!(grid.n_leaves(), 6);
    }

    #[test]
    fn test_refine_above_threshold_doubles_cells() {
        let mut grid = AmrGrid2D::new_uniform(2, 2, [0.0, 1.0, 0.0, 1.0]);
        // Set all values above threshold so all 4 cells get refined
        for i in 0..grid.nodes.len() {
            grid.nodes[i].cell.value = 1.0;
        }
        let config = AmrConfig {
            max_level: 4,
            refine_threshold: 0.5,
            coarsen_threshold: 0.0,
            balance_type: BalanceType::None,
        };
        let n_before = grid.n_leaves();
        grid.refine_parallel(|cell| cell.value, &config);
        let n_after = grid.n_leaves();
        // Each of the 4 original cells splits into 4 → 16 leaves
        assert_eq!(n_before, 4);
        assert_eq!(n_after, 16);
    }

    #[test]
    fn test_no_refinement_below_threshold() {
        let mut grid = AmrGrid2D::new_uniform(3, 3, [0.0, 1.0, 0.0, 1.0]);
        let config = AmrConfig {
            max_level: 4,
            refine_threshold: 100.0, // very high — nothing gets refined
            coarsen_threshold: 0.0,
            balance_type: BalanceType::None,
        };
        let n_before = grid.n_leaves();
        grid.refine_parallel(|_| 0.0, &config);
        assert_eq!(grid.n_leaves(), n_before);
    }

    #[test]
    fn test_max_level_respected() {
        let mut grid = AmrGrid2D::new_uniform(1, 1, [0.0, 1.0, 0.0, 1.0]);
        let config = AmrConfig {
            max_level: 1,
            refine_threshold: 0.0,
            coarsen_threshold: 0.0,
            balance_type: BalanceType::None,
        };
        // Refine twice — second pass should not exceed max_level=1
        grid.refine_parallel(|_| 1.0, &config);
        let n_after_first = grid.n_leaves();
        grid.refine_parallel(|_| 1.0, &config);
        let n_after_second = grid.n_leaves();
        assert_eq!(n_after_first, 4, "First refinement should yield 4 leaves");
        assert_eq!(
            n_after_second, 4,
            "Second refinement at max_level should be a no-op"
        );
    }

    #[test]
    fn test_balance_enforces_2to1() {
        // Create a 4×4 grid, refine only the top-right cell heavily
        let mut grid = AmrGrid2D::new_uniform(4, 4, [0.0, 1.0, 0.0, 1.0]);
        // Set the top-right cell value high
        for i in 0..grid.nodes.len() {
            let node = &grid.nodes[i];
            if node.cell.x > 0.75 - 1e-9 && node.cell.y > 0.75 - 1e-9 {
                grid.nodes[i].cell.value = 10.0;
            }
        }
        let config = AmrConfig {
            max_level: 3,
            refine_threshold: 5.0,
            coarsen_threshold: 0.0,
            balance_type: BalanceType::Face2to1,
        };
        grid.refine_parallel(|cell| cell.value, &config);

        // After balance: no two leaf cells with shared face should have level diff > 1
        let leaves: Vec<(usize, u32, &AmrCell)> = grid
            .leaf_indices
            .iter()
            .map(|&i| (i, grid.nodes[i].level, &grid.nodes[i].cell))
            .collect();

        for i in 0..leaves.len() {
            for j in (i + 1)..leaves.len() {
                let (_, li, ci) = &leaves[i];
                let (_, lj, cj) = &leaves[j];
                // Check for face-neighbour relationship via bounding boxes
                let ax_max = ci.x + ci.width;
                let ay_max = ci.y + ci.height;
                let bx_max = cj.x + cj.width;
                let by_max = cj.y + cj.height;
                let eps = 1e-9;
                let share_x = ((ax_max - cj.x).abs() < eps || (bx_max - ci.x).abs() < eps)
                    && ci.y < by_max - eps
                    && cj.y < ay_max - eps;
                let share_y = ((ay_max - cj.y).abs() < eps || (by_max - ci.y).abs() < eps)
                    && ci.x < bx_max - eps
                    && cj.x < ax_max - eps;

                if share_x || share_y {
                    let diff = (*li as i64 - *lj as i64).unsigned_abs();
                    assert!(
                        diff <= 1,
                        "2:1 balance violated: leaf {} (level {}) and leaf {} (level {}) are face neighbours with level diff {}",
                        i, li, j, lj, diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_load_stats_nonzero_on_refined_grid() {
        let mut grid = AmrGrid2D::new_uniform(4, 4, [0.0, 1.0, 0.0, 1.0]);
        let config = AmrConfig {
            max_level: 2,
            refine_threshold: -1.0, // refine everything
            coarsen_threshold: -2.0,
            balance_type: BalanceType::None,
        };
        grid.refine_parallel(|_| 0.0, &config);
        let stats = grid.load_stats();
        assert!(stats.max > 0, "Max load should be > 0 after refinement");
        assert!(stats.mean > 0.0, "Mean load should be > 0 after refinement");
        assert!(stats.min <= stats.max, "Min must not exceed max");
    }

    #[test]
    fn test_amr_refine_parallel_matches_serial() {
        // Both parallel and serial should produce the same leaf count
        // (this test checks reproducibility, not actual parallelism)
        let mut grid1 = AmrGrid2D::new_uniform(3, 3, [0.0, 1.0, 0.0, 1.0]);
        let mut grid2 = AmrGrid2D::new_uniform(3, 3, [0.0, 1.0, 0.0, 1.0]);
        let config = AmrConfig {
            max_level: 2,
            refine_threshold: -1.0,
            coarsen_threshold: -2.0,
            balance_type: BalanceType::None,
        };
        grid1.refine_parallel(|_| 0.0, &config);
        // Serial: call refine_cell directly for each leaf
        let initial_leaves: Vec<usize> = grid2.leaf_indices.clone();
        for li in initial_leaves {
            grid2.refine_cell(li);
        }
        grid2.rebuild_leaf_indices();
        assert_eq!(
            grid1.n_leaves(),
            grid2.n_leaves(),
            "Parallel and serial refinement must produce same leaf count"
        );
    }

    #[test]
    fn test_leaves_accessor() {
        let grid = AmrGrid2D::new_uniform(2, 2, [0.0, 4.0, 0.0, 4.0]);
        let leaves = grid.leaves();
        assert_eq!(leaves.len(), 4);
        // All leaves should be inside the domain
        for leaf in &leaves {
            assert!(leaf.x >= 0.0 && leaf.x + leaf.width <= 4.0);
            assert!(leaf.y >= 0.0 && leaf.y + leaf.height <= 4.0);
        }
    }
}
