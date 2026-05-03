//! Full AMR quad-tree/oct-tree dynamic refinement framework.
//!
//! # Modules
//!
//! - [`quadtree`]: 2-D quad-tree with Morton ordering, `CellId`, `CellData`,
//!   `RefinementCriterion`, `GradientCriterion`
//! - [`octree`]: 3-D oct-tree with 3-D Morton ordering (`OctTree`, `OctCellId`, …)
//! - [`operators`]: Conservative prolongation and restriction operators
//!   (`AmrOperators`)
//! - [`level_set`]: Level-set interface tracking on a quad-tree (`LevelSet`)
//!
//! # Quick Example
//!
//! ```
//! use scirs2_integrate::amr::quadtree::{QuadTree, GradientCriterion};
//! use scirs2_integrate::amr::operators::AmrOperators;
//!
//! // Build a 2-D quad-tree over [0,1]²
//! let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 4, 2);
//! let root = scirs2_integrate::amr::quadtree::CellId::new(0, 0);
//! tree.set_values(root, &[0.0, 5.5]);
//!
//! // Refine cells where the inter-variable gradient > 3
//! let crit = GradientCriterion::new(3.0);
//! tree.adapt(&crit);
//! assert!(tree.leaves().len() > 1);
//!
//! // Conservative prolongation / restriction
//! let ops = AmrOperators::new();
//! let parent_vals = vec![1.0, 2.0];
//! let mut children = [
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//! ];
//! ops.prolongate_2d_values(&parent_vals, &mut children);
//! let restricted = ops.restrict_2d_values(&children);
//! assert!((restricted[0] - 1.0).abs() < 1e-12);
//! ```

pub mod level_set;
pub mod load_balanced;
pub mod octree;
pub mod operators;
pub mod quadtree;

// ── Top-level re-exports ────────────────────────────────────────────────────

// Quad-tree types
pub use quadtree::{
    CellData, CellId, GradientCriterion, MagnitudeCriterion, Morton2D, QuadTree,
    RefinementCriterion,
};

// Oct-tree types
pub use octree::{
    GradientCriterion3D, Morton3D, OctCellData, OctCellId, OctTree, RefinementCriterion3D,
};

// Operators
pub use operators::{prolongate_2d, prolongate_3d, restrict_2d, restrict_3d, AmrOperators};

// Level-set
pub use level_set::LevelSet;

// Load-balanced AMR
pub use load_balanced::{AmrCell, AmrConfig, AmrGrid2D, BalanceType, LoadStats};
