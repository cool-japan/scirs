//! Advanced Adaptive Mesh Refinement (AMR) with sophisticated error indicators
//!
//! This module provides state-of-the-art adaptive mesh refinement techniques
//! including gradient-based refinement, feature detection, load balancing,
//! and hierarchical mesh management for complex PDE solutions.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Advanced AMR manager with multiple refinement strategies
pub struct AdvancedAMRManager<F: IntegrateFloat> {
    /// Current mesh hierarchy
    pub mesh_hierarchy: MeshHierarchy<F>,
    /// Refinement criteria
    pub refinement_criteria: Vec<Box<dyn RefinementCriterion<F>>>,
    /// Load balancing strategy
    pub load_balancer: Option<Box<dyn LoadBalancer<F>>>,
    /// Maximum refinement levels
    pub max_levels: usize,
    /// Minimum cell size
    pub min_cell_size: F,
    /// Coarsening tolerance
    pub coarsening_tolerance: F,
    /// Error tolerance for refinement
    pub refinement_tolerance: F,
    /// Adaptation frequency
    pub adaptation_frequency: usize,
    /// Current adaptation step
    current_step: usize,
}

/// Hierarchical mesh structure supporting multiple levels
#[derive(Debug, Clone)]
pub struct MeshHierarchy<F: IntegrateFloat> {
    /// Mesh levels (0 = coarsest)
    pub levels: Vec<AdaptiveMeshLevel<F>>,
    /// Parent-child relationships
    pub hierarchy_map: HashMap<CellId, Vec<CellId>>,
    /// Ghost cell information for parallel processing
    pub ghost_cells: HashMap<usize, Vec<CellId>>, // level -> ghost cells
}

/// Single level in adaptive mesh
#[derive(Debug, Clone)]
pub struct AdaptiveMeshLevel<F: IntegrateFloat> {
    /// Level number (0 = coarsest)
    pub level: usize,
    /// Active cells at this level
    pub cells: HashMap<CellId, AdaptiveCell<F>>,
    /// Grid spacing at this level
    pub grid_spacing: F,
    /// Boundary information
    pub boundary_cells: HashSet<CellId>,
}

/// Individual adaptive cell
#[derive(Debug, Clone)]
pub struct AdaptiveCell<F: IntegrateFloat> {
    /// Unique cell identifier
    pub id: CellId,
    /// Cell center coordinates
    pub center: Array1<F>,
    /// Cell size
    pub size: F,
    /// Solution value(s) in cell
    pub solution: Array1<F>,
    /// Error estimate for cell
    pub error_estimate: F,
    /// Refinement flag
    pub refinement_flag: RefinementFlag,
    /// Activity status
    pub is_active: bool,
    /// Neighboring cell IDs
    pub neighbors: Vec<CellId>,
    /// Parent cell ID (if refined)
    pub parent: Option<CellId>,
    /// Children cell IDs (if coarsened)
    pub children: Vec<CellId>,
}

/// Cell identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId {
    pub level: usize,
    pub index: usize,
}

/// Refinement action flags
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefinementFlag {
    /// No action needed
    None,
    /// Cell should be refined
    Refine,
    /// Cell should be coarsened
    Coarsen,
    /// Cell marked for potential refinement
    Tagged,
}

/// Trait for refinement criteria
pub trait RefinementCriterion<F: IntegrateFloat>: Send + Sync {
    /// Evaluate refinement criterion for a cell
    fn evaluate(&self, cell: &AdaptiveCell<F>, neighbors: &[&AdaptiveCell<F>]) -> F;

    /// Get criterion name
    fn name(&self) -> &'static str;

    /// Get criterion weight in combined evaluation
    fn weight(&self) -> F {
        F::one()
    }
}

/// Gradient-based refinement criterion
pub struct GradientRefinementCriterion<F: IntegrateFloat> {
    /// Component to analyze (None = all components)
    pub component: Option<usize>,
    /// Gradient threshold
    pub threshold: F,
    /// Relative tolerance
    pub relative_tolerance: F,
}

/// Feature detection refinement criterion
pub struct FeatureDetectionCriterion<F: IntegrateFloat> {
    /// Feature detection threshold
    pub threshold: F,
    /// Feature types to detect
    pub feature_types: Vec<FeatureType>,
    /// Window size for feature detection
    pub window_size: usize,
}

/// Curvature-based refinement criterion
pub struct CurvatureRefinementCriterion<F: IntegrateFloat> {
    /// Curvature threshold
    pub threshold: F,
    /// Approximation order for curvature calculation
    pub approximation_order: usize,
}

/// Load balancing strategy trait
pub trait LoadBalancer<F: IntegrateFloat>: Send + Sync {
    /// Balance computational load across processors/threads
    fn balance(&self, hierarchy: &mut MeshHierarchy<F>) -> IntegrateResult<()>;
}

/// Zoltan-style geometric load balancer
pub struct GeometricLoadBalancer<F: IntegrateFloat> {
    /// Number of partitions
    pub num_partitions: usize,
    /// Load imbalance tolerance
    pub imbalance_tolerance: F,
    /// Partitioning method
    pub method: PartitioningMethod,
}

/// Types of features to detect
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// Sharp gradients
    SharpGradient,
    /// Discontinuities
    Discontinuity,
    /// Local extrema
    LocalExtremum,
    /// Oscillatory behavior
    Oscillation,
    /// Boundary layers
    BoundaryLayer,
}

/// Partitioning methods
#[derive(Debug, Clone, Copy)]
pub enum PartitioningMethod {
    /// Recursive coordinate bisection
    RCB,
    /// Space-filling curves
    SFC,
    /// Graph partitioning
    Graph,
}

/// AMR adaptation result
pub struct AMRAdaptationResult<F: IntegrateFloat> {
    /// Number of cells refined
    pub cells_refined: usize,
    /// Number of cells coarsened
    pub cells_coarsened: usize,
    /// Total active cells after adaptation
    pub total_active_cells: usize,
    /// Load balance quality metric
    pub load_balance_quality: F,
    /// Memory usage change
    pub memory_change: i64,
    /// Adaptation time
    pub adaptation_time: std::time::Duration,
}

impl<F: IntegrateFloat> AdvancedAMRManager<F> {
    /// Create new advanced AMR manager
    pub fn new(_initial_mesh: AdaptiveMeshLevel<F>, max_levels: usize, min_cellsize: F) -> Self {
        let mesh_hierarchy = MeshHierarchy {
            levels: vec![_initial_mesh],
            hierarchy_map: HashMap::new(),
            ghost_cells: HashMap::new(),
        };

        Self {
            mesh_hierarchy,
            refinement_criteria: Vec::new(),
            load_balancer: None,
            max_levels,
            min_cell_size: min_cellsize,
            coarsening_tolerance: F::from(0.1).unwrap(),
            refinement_tolerance: F::from(1.0).unwrap(),
            adaptation_frequency: 1,
            current_step: 0,
        }
    }

    /// Add refinement criterion
    pub fn add_criterion(&mut self, criterion: Box<dyn RefinementCriterion<F>>) {
        self.refinement_criteria.push(criterion);
    }

    /// Set load balancer
    pub fn set_load_balancer(&mut self, balancer: Box<dyn LoadBalancer<F>>) {
        self.load_balancer = Some(balancer);
    }

    /// Perform adaptive mesh refinement
    pub fn adapt_mesh(&mut self, solution: &Array2<F>) -> IntegrateResult<AMRAdaptationResult<F>> {
        let start_time = std::time::Instant::now();
        let initial_cells = self.count_active_cells();

        self.current_step += 1;

        // Skip adaptation if not at adaptation frequency
        if self.current_step % self.adaptation_frequency != 0 {
            return Ok(AMRAdaptationResult {
                cells_refined: 0,
                cells_coarsened: 0,
                total_active_cells: initial_cells,
                load_balance_quality: F::one(),
                memory_change: 0,
                adaptation_time: start_time.elapsed(),
            });
        }

        // Step 1: Update solution values in cells
        self.update_cell_solutions(solution)?;

        // Step 2: Evaluate refinement criteria
        self.evaluate_refinement_criteria()?;

        // Step 3: Flag cells for refinement/coarsening
        let _refine_count_coarsen_count = self.flag_cells_for_adaptation()?;

        // Step 4: Perform refinement
        let cells_refined = self.refine_flagged_cells()?;

        // Step 5: Perform coarsening
        let cells_coarsened = self.coarsen_flagged_cells()?;

        // Step 6: Load balancing
        let load_balance_quality = if let Some(ref balancer) = self.load_balancer {
            balancer.balance(&mut self.mesh_hierarchy)?;
            self.assess_load_balance()
        } else {
            F::one()
        };

        // Step 7: Update ghost cells
        self.update_ghost_cells()?;

        let final_cells = self.count_active_cells();
        let memory_change = (final_cells as i64 - initial_cells as i64) * 8; // Rough estimate

        Ok(AMRAdaptationResult {
            cells_refined,
            cells_coarsened,
            total_active_cells: final_cells,
            load_balance_quality,
            memory_change,
            adaptation_time: start_time.elapsed(),
        })
    }

    /// Update solution values in mesh cells
    fn update_cell_solutions(&mut self, solution: &Array2<F>) -> IntegrateResult<()> {
        // Map solution array to mesh cells
        // This is a simplified mapping - in practice would need sophisticated interpolation
        for level in &mut self.mesh_hierarchy.levels {
            for cell in level.cells.values_mut() {
                if cell.is_active {
                    // Simple mapping - in practice would use proper interpolation
                    let i = (cell.center[0] * F::from(solution.nrows()).unwrap())
                        .to_usize()
                        .unwrap_or(0)
                        .min(solution.nrows() - 1);
                    let j = if solution.ncols() > 1 && cell.center.len() > 1 {
                        (cell.center[1] * F::from(solution.ncols()).unwrap())
                            .to_usize()
                            .unwrap_or(0)
                            .min(solution.ncols() - 1)
                    } else {
                        0
                    };

                    // Update cell solution (simplified)
                    if cell.solution.len() == 1 {
                        cell.solution[0] = solution[[i, j]];
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluate all refinement criteria for all cells
    fn evaluate_refinement_criteria(&mut self) -> IntegrateResult<()> {
        for level in &mut self.mesh_hierarchy.levels {
            let cellids: Vec<CellId> = level.cells.keys().cloned().collect();

            for cellid in cellids {
                if let Some(cell) = level.cells.get(&cellid) {
                    if !cell.is_active {
                        continue;
                    }

                    // Get neighboring cells
                    let neighbor_cells: Vec<&AdaptiveCell<F>> = cell
                        .neighbors
                        .iter()
                        .filter_map(|&neighbor_id| level.cells.get(&neighbor_id))
                        .collect();

                    // Evaluate all criteria
                    let mut total_error = F::zero();
                    let mut total_weight = F::zero();

                    for criterion in &self.refinement_criteria {
                        let error = criterion.evaluate(cell, &neighbor_cells);
                        let weight = criterion.weight();
                        total_error += error * weight;
                        total_weight += weight;
                    }

                    // Normalize error estimate
                    let error_estimate = if total_weight > F::zero() {
                        total_error / total_weight
                    } else {
                        F::zero()
                    };

                    // Update cell error estimate
                    if let Some(cell) = level.cells.get_mut(&cellid) {
                        cell.error_estimate = error_estimate;
                    }
                }
            }
        }
        Ok(())
    }

    /// Flag cells for refinement or coarsening
    fn flag_cells_for_adaptation(&mut self) -> IntegrateResult<(usize, usize)> {
        let mut refine_count = 0;
        let mut coarsen_count = 0;

        // Collect cells that can be coarsened first to avoid borrowing issues
        let mut cells_to_check: Vec<(usize, CellId, F, usize, F)> = Vec::new();

        for level in &self.mesh_hierarchy.levels {
            for cell in level.cells.values() {
                if cell.is_active {
                    cells_to_check.push((
                        level.level,
                        cell.id,
                        cell.error_estimate,
                        level.level,
                        cell.size,
                    ));
                }
            }
        }

        // Now flag cells based on collected information
        for (level_idx, cellid, error_estimate, level_num, cell_size) in cells_to_check {
            if let Some(level) = self.mesh_hierarchy.levels.get_mut(level_idx) {
                if let Some(cell) = level.cells.get_mut(&cellid) {
                    // Refinement criterion
                    if error_estimate > self.refinement_tolerance
                        && level_num < self.max_levels
                        && cell_size > self.min_cell_size
                    {
                        cell.refinement_flag = RefinementFlag::Refine;
                        refine_count += 1;
                    }
                    // Coarsening criterion (simplified check)
                    else if error_estimate < self.coarsening_tolerance && level_num > 0 {
                        cell.refinement_flag = RefinementFlag::Coarsen;
                        coarsen_count += 1;
                    } else {
                        cell.refinement_flag = RefinementFlag::None;
                    }
                }
            }
        }

        Ok((refine_count, coarsen_count))
    }

    /// Check if cell can be coarsened (all siblings must be flagged)
    fn can_coarsen_cell(&self, cell: &AdaptiveCell<F>) -> bool {
        if let Some(parent_id) = cell.parent {
            // Check if all sibling cells are also flagged for coarsening
            if let Some(parent_children) = self.mesh_hierarchy.hierarchy_map.get(&parent_id) {
                for &child_id in parent_children {
                    if let Some(level) = self.mesh_hierarchy.levels.get(child_id.level) {
                        if let Some(sibling) = level.cells.get(&child_id) {
                            if sibling.refinement_flag != RefinementFlag::Coarsen {
                                return false;
                            }
                        }
                    }
                }
                return true;
            }
        }
        false
    }

    /// Refine flagged cells
    fn refine_flagged_cells(&mut self) -> IntegrateResult<usize> {
        let mut cells_refined = 0;

        // Process each level separately to avoid borrowing issues
        for level_idx in 0..self.mesh_hierarchy.levels.len() {
            let cells_to_refine: Vec<CellId> = self.mesh_hierarchy.levels[level_idx]
                .cells
                .values()
                .filter(|cell| cell.refinement_flag == RefinementFlag::Refine)
                .map(|cell| cell.id)
                .collect();

            for cellid in cells_to_refine {
                self.refine_cell(cellid)?;
                cells_refined += 1;
            }
        }

        Ok(cells_refined)
    }

    /// Refine a single cell
    fn refine_cell(&mut self, cellid: CellId) -> IntegrateResult<()> {
        let parent_cell = if let Some(level) = self.mesh_hierarchy.levels.get(cellid.level) {
            level.cells.get(&cellid).cloned()
        } else {
            return Err(IntegrateError::ValueError("Invalid cell level".to_string()));
        };

        let parent_cell =
            parent_cell.ok_or_else(|| IntegrateError::ValueError("Cell not found".to_string()))?;

        // Create child level if it doesn't exist
        let child_level = cellid.level + 1;
        while self.mesh_hierarchy.levels.len() <= child_level {
            let new_level = AdaptiveMeshLevel {
                level: self.mesh_hierarchy.levels.len(),
                cells: HashMap::new(),
                grid_spacing: if let Some(last_level) = self.mesh_hierarchy.levels.last() {
                    last_level.grid_spacing / F::from(2.0).unwrap()
                } else {
                    F::one()
                },
                boundary_cells: HashSet::new(),
            };
            self.mesh_hierarchy.levels.push(new_level);
        }

        // Create child cells (2D refinement = 4 children, 3D = 8 children)
        let num_children = 2_usize.pow(parent_cell.center.len() as u32);
        let mut child_ids = Vec::new();
        let child_size = parent_cell.size / F::from(2.0).unwrap();

        for child_idx in 0..num_children {
            let child_id = CellId {
                level: child_level,
                index: self.mesh_hierarchy.levels[child_level].cells.len(),
            };

            // Compute child center
            let mut child_center = parent_cell.center.clone();
            let offset = child_size / F::from(2.0).unwrap();

            // Binary representation determines position
            for dim in 0..parent_cell.center.len() {
                if (child_idx >> dim) & 1 == 1 {
                    child_center[dim] += offset;
                } else {
                    child_center[dim] -= offset;
                }
            }

            let child_cell = AdaptiveCell {
                id: child_id,
                center: child_center,
                size: child_size,
                solution: parent_cell.solution.clone(), // Inherit parent solution
                error_estimate: F::zero(),
                refinement_flag: RefinementFlag::None,
                is_active: true,
                neighbors: Vec::new(),
                parent: Some(cellid),
                children: Vec::new(),
            };

            self.mesh_hierarchy.levels[child_level]
                .cells
                .insert(child_id, child_cell);
            child_ids.push(child_id);
        }

        // Update hierarchy map
        self.mesh_hierarchy
            .hierarchy_map
            .insert(cellid, child_ids.clone());

        // Deactivate parent cell
        if let Some(parent) = self.mesh_hierarchy.levels[cellid.level]
            .cells
            .get_mut(&cellid)
        {
            parent.is_active = false;
            parent.children = child_ids;
        }

        // Update neighbor relationships
        self.update_neighbor_relationships(child_level)?;

        Ok(())
    }

    /// Coarsen flagged cells
    fn coarsen_flagged_cells(&mut self) -> IntegrateResult<usize> {
        let mut cells_coarsened = 0;

        // Process from finest to coarsest level
        for level_idx in (1..self.mesh_hierarchy.levels.len()).rev() {
            let parent_cells_to_activate: Vec<CellId> = self.mesh_hierarchy.levels[level_idx]
                .cells
                .values()
                .filter(|cell| cell.refinement_flag == RefinementFlag::Coarsen)
                .filter_map(|cell| cell.parent)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            for parent_id in parent_cells_to_activate {
                if self.coarsen_to_parent(parent_id)? {
                    cells_coarsened += 1;
                }
            }
        }

        Ok(cells_coarsened)
    }

    /// Coarsen children back to parent cell
    fn coarsen_to_parent(&mut self, parentid: CellId) -> IntegrateResult<bool> {
        let child_ids = if let Some(children) = self.mesh_hierarchy.hierarchy_map.get(&parentid) {
            children.clone()
        } else {
            return Ok(false);
        };

        // Verify all children are flagged for coarsening
        for &child_id in &child_ids {
            if let Some(level) = self.mesh_hierarchy.levels.get(child_id.level) {
                if let Some(child) = level.cells.get(&child_id) {
                    if child.refinement_flag != RefinementFlag::Coarsen {
                        return Ok(false);
                    }
                }
            }
        }

        // Average child solutions for parent
        let mut avg_solution = Array1::zeros(child_ids.len());
        if !child_ids.is_empty() {
            if let Some(first_child_level) = self.mesh_hierarchy.levels.get(child_ids[0].level) {
                if let Some(first_child) = first_child_level.cells.get(&child_ids[0]) {
                    avg_solution = Array1::zeros(first_child.solution.len());

                    for &child_id in &child_ids {
                        if let Some(child_level) = self.mesh_hierarchy.levels.get(child_id.level) {
                            if let Some(child) = child_level.cells.get(&child_id) {
                                avg_solution = &avg_solution + &child.solution;
                            }
                        }
                    }
                    avg_solution /= F::from(child_ids.len()).unwrap();
                }
            }
        }

        // Reactivate parent cell
        if let Some(parent_level) = self.mesh_hierarchy.levels.get_mut(parentid.level) {
            if let Some(parent) = parent_level.cells.get_mut(&parentid) {
                parent.is_active = true;
                parent.solution = avg_solution;
                parent.children.clear();
                parent.refinement_flag = RefinementFlag::None;
            }
        }

        // Remove children from hierarchy
        for &child_id in &child_ids {
            if let Some(child_level) = self.mesh_hierarchy.levels.get_mut(child_id.level) {
                child_level.cells.remove(&child_id);
            }
        }

        // Remove from hierarchy map
        self.mesh_hierarchy.hierarchy_map.remove(&parentid);

        Ok(true)
    }

    /// Update neighbor relationships after refinement
    fn update_neighbor_relationships(&mut self, level: usize) -> IntegrateResult<()> {
        // Collect neighbor relationships first to avoid borrowing conflicts
        let mut all_neighbor_relationships: Vec<(CellId, Vec<CellId>)> = Vec::new();

        if let Some(mesh_level) = self.mesh_hierarchy.levels.get(level) {
            let cellids: Vec<CellId> = mesh_level.cells.keys().cloned().collect();

            // Build spatial hash map for efficient neighbor searching
            let mut spatial_hash: HashMap<(i32, i32, i32), Vec<CellId>> = HashMap::new();
            let grid_spacing = mesh_level.grid_spacing;

            // Hash all cells based on their spatial location
            for cellid in &cellids {
                if let Some(cell) = mesh_level.cells.get(cellid) {
                    if cell.center.len() >= 3 {
                        let hash_x = (cell.center[0] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);
                        let hash_y = (cell.center[1] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);
                        let hash_z = (cell.center[2] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);

                        spatial_hash
                            .entry((hash_x, hash_y, hash_z))
                            .or_default()
                            .push(*cellid);
                    }
                }
            }

            for cellid in &cellids {
                if let Some(cell) = mesh_level.cells.get(cellid) {
                    let mut neighbors = Vec::new();

                    if cell.center.len() >= 3 {
                        let hash_x = (cell.center[0] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);
                        let hash_y = (cell.center[1] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);
                        let hash_z = (cell.center[2] / grid_spacing)
                            .floor()
                            .to_i32()
                            .unwrap_or(0);

                        // Search in 27 neighboring hash buckets (3x3x3)
                        for dx in -1..=1 {
                            for dy in -1..=1 {
                                for dz in -1..=1 {
                                    let hash_key = (hash_x + dx, hash_y + dy, hash_z + dz);

                                    if let Some(potential_neighbors) = spatial_hash.get(&hash_key) {
                                        for &neighbor_id in potential_neighbors {
                                            if neighbor_id != *cellid {
                                                if let Some(neighbor_cell) =
                                                    mesh_level.cells.get(&neighbor_id)
                                                {
                                                    // Check if cells are actually neighbors (face/edge/vertex sharing)
                                                    if self.are_cells_neighbors(cell, neighbor_cell)
                                                    {
                                                        neighbors.push(neighbor_id);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    all_neighbor_relationships.push((*cellid, neighbors));
                }
            }
        }

        // Now apply all neighbor relationships with mutable access
        if let Some(mesh_level) = self.mesh_hierarchy.levels.get_mut(level) {
            for (cellid, neighbors) in all_neighbor_relationships {
                if let Some(cell) = mesh_level.cells.get_mut(&cellid) {
                    cell.neighbors = neighbors;
                }
            }
        }

        // Now update inter-level neighbors separately to avoid borrowing conflicts
        let cellids: Vec<CellId> = if let Some(mesh_level) = self.mesh_hierarchy.levels.get(level) {
            mesh_level.cells.keys().cloned().collect()
        } else {
            Vec::new()
        };

        for cellid in cellids {
            self.update_interlevel_neighbors(cellid, level)?;
        }

        Ok(())
    }

    /// Check if two cells are geometric neighbors
    fn are_cells_neighbors(&self, cell1: &AdaptiveCell<F>, cell2: &AdaptiveCell<F>) -> bool {
        if cell1.center.len() != cell2.center.len() || cell1.center.len() < 3 {
            return false;
        }

        let max_size = cell1.size.max(cell2.size);
        let tolerance = max_size * F::from(1.1).unwrap(); // 10% tolerance

        // Calculate distance between cell centers
        let mut distance_squared = F::zero();
        for i in 0..cell1.center.len() {
            let diff = cell1.center[i] - cell2.center[i];
            distance_squared += diff * diff;
        }

        let distance = distance_squared.sqrt();

        // Cells are neighbors if distance is approximately equal to sum of half-sizes
        let expected_distance = (cell1.size + cell2.size) / F::from(2.0).unwrap();

        distance <= tolerance && distance >= expected_distance * F::from(0.7).unwrap()
    }

    /// Update neighbor relationships across different mesh levels
    fn update_interlevel_neighbors(&mut self, cellid: CellId, level: usize) -> IntegrateResult<()> {
        // Collect neighbor relationships first to avoid borrowing conflicts
        let mut coarser_neighbors = Vec::new();
        let mut finer_neighbors = Vec::new();

        // Check neighbors at level-1 (coarser level)
        if level > 0 {
            if let (Some(current_level), Some(coarser_level)) = (
                self.mesh_hierarchy.levels.get(level),
                self.mesh_hierarchy.levels.get(level - 1),
            ) {
                if let Some(current_cell) = current_level.cells.get(&cellid) {
                    for (coarser_cellid, coarser_cell) in &coarser_level.cells {
                        if self.are_cells_neighbors(current_cell, coarser_cell) {
                            coarser_neighbors.push(*coarser_cellid);
                        }
                    }
                }
            }
        }

        // Check neighbors at level+1 (finer level)
        if level + 1 < self.mesh_hierarchy.levels.len() {
            if let (Some(current_level), Some(finer_level)) = (
                self.mesh_hierarchy.levels.get(level),
                self.mesh_hierarchy.levels.get(level + 1),
            ) {
                if let Some(current_cell) = current_level.cells.get(&cellid) {
                    for (finer_cellid, finer_cell) in &finer_level.cells {
                        if self.are_cells_neighbors(current_cell, finer_cell) {
                            finer_neighbors.push(*finer_cellid);
                        }
                    }
                }
            }
        }

        // Now apply the neighbor relationships with mutable access
        if let Some(current_level) = self.mesh_hierarchy.levels.get_mut(level) {
            if let Some(current_cell) = current_level.cells.get_mut(&cellid) {
                for coarser_id in coarser_neighbors {
                    if !current_cell.neighbors.contains(&coarser_id) {
                        current_cell.neighbors.push(coarser_id);
                    }
                }

                for finer_id in finer_neighbors {
                    if !current_cell.neighbors.contains(&finer_id) {
                        current_cell.neighbors.push(finer_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Update ghost cells for parallel processing
    fn update_ghost_cells(&mut self) -> IntegrateResult<()> {
        // Clear existing ghost cells
        self.mesh_hierarchy.ghost_cells.clear();

        // Identify boundary cells and their external neighbors for each level
        for level_idx in 0..self.mesh_hierarchy.levels.len() {
            let mut ghost_cells_for_level = Vec::new();
            let mut boundary_cells = HashSet::new();

            // First pass: identify boundary cells (cells with fewer neighbors than expected)
            let expected_neighbors = self.calculate_expected_neighbors();

            if let Some(mesh_level) = self.mesh_hierarchy.levels.get(level_idx) {
                for (cellid, cell) in &mesh_level.cells {
                    // A cell is on the boundary if it has fewer neighbors than expected
                    // or if it's marked as a boundary cell
                    if cell.neighbors.len() < expected_neighbors
                        || mesh_level.boundary_cells.contains(cellid)
                    {
                        boundary_cells.insert(*cellid);
                    }
                }

                // Second pass: create ghost cells for parallel processing
                for boundary_cellid in &boundary_cells {
                    if let Some(boundary_cell) = mesh_level.cells.get(boundary_cellid) {
                        // Create ghost cells in the expected neighbor positions
                        let ghost_cells =
                            self.create_ghost_cells_for_boundary(boundary_cell, level_idx)?;
                        ghost_cells_for_level.extend(ghost_cells);
                    }
                }

                // Third pass: handle inter-level ghost cells
                self.create_interlevel_ghost_cells(level_idx, &mut ghost_cells_for_level)?;
            }

            self.mesh_hierarchy
                .ghost_cells
                .insert(level_idx, ghost_cells_for_level);
        }

        Ok(())
    }

    /// Calculate expected number of neighbors for a regular cell
    fn calculate_expected_neighbors(&self) -> usize {
        // For a 3D structured grid, a regular internal cell should have 6 face neighbors
        // For 2D: 4 neighbors, for 1D: 2 neighbors
        // This is a simplification - actual count depends on mesh structure
        6
    }

    /// Create ghost cells for a boundary cell
    fn create_ghost_cells_for_boundary(
        &self,
        boundary_cell: &AdaptiveCell<F>,
        level: usize,
    ) -> IntegrateResult<Vec<CellId>> {
        let mut ghost_cells = Vec::new();

        if boundary_cell.center.len() >= 3 {
            let cell_size = boundary_cell.size;

            // Create ghost cells in the 6 cardinal directions (±x, ±y, ±z)
            let directions = [
                [F::one(), F::zero(), F::zero()],  // +x
                [-F::one(), F::zero(), F::zero()], // -x
                [F::zero(), F::one(), F::zero()],  // +y
                [F::zero(), -F::one(), F::zero()], // -y
                [F::zero(), F::zero(), F::one()],  // +z
                [F::zero(), F::zero(), -F::one()], // -z
            ];

            for (dir_idx, direction) in directions.iter().enumerate() {
                // Calculate ghost _cell position
                let mut ghost_center = boundary_cell.center.clone();
                for i in 0..3 {
                    ghost_center[i] += direction[i] * cell_size;
                }

                // Check if a real _cell exists at this position
                if !self.cell_exists_at_position(&ghost_center, level) {
                    // Create ghost _cell ID (using high indices to avoid conflicts)
                    let ghost_id = CellId {
                        level,
                        index: 1_000_000 + boundary_cell.id.index * 10 + dir_idx,
                    };

                    ghost_cells.push(ghost_id);
                }
            }
        }

        Ok(ghost_cells)
    }

    /// Create ghost cells for inter-level communication
    fn create_interlevel_ghost_cells(
        &self,
        level: usize,
        ghost_cells: &mut Vec<CellId>,
    ) -> IntegrateResult<()> {
        // Handle ghost _cells needed for communication between mesh levels

        // Check if we need ghost _cells from coarser level
        if level > 0 {
            if let Some(current_level) = self.mesh_hierarchy.levels.get(level) {
                for (cellid, cell) in &current_level.cells {
                    // If this fine cell doesn't have a parent at the coarser level,
                    // it might need ghost cell communication
                    if cell.parent.is_none() {
                        let ghost_id = CellId {
                            level: level - 1,
                            index: 2_000_000 + cellid.index,
                        };
                        ghost_cells.push(ghost_id);
                    }
                }
            }
        }

        // Check if we need ghost _cells from finer level
        if level + 1 < self.mesh_hierarchy.levels.len() {
            if let Some(current_level) = self.mesh_hierarchy.levels.get(level) {
                for (cellid, cell) in &current_level.cells {
                    // If this coarse cell has children at the finer level,
                    // it might need ghost cell communication
                    if !cell.children.is_empty() {
                        let ghost_id = CellId {
                            level: level + 1,
                            index: 3_000_000 + cellid.index,
                        };
                        ghost_cells.push(ghost_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a cell exists at the given position and level
    fn cell_exists_at_position(&self, position: &Array1<F>, level: usize) -> bool {
        if let Some(mesh_level) = self.mesh_hierarchy.levels.get(level) {
            let tolerance = mesh_level.grid_spacing * F::from(0.1).unwrap();

            for cell in mesh_level.cells.values() {
                if position.len() == cell.center.len() {
                    let mut distance_squared = F::zero();
                    for i in 0..position.len() {
                        let diff = position[i] - cell.center[i];
                        distance_squared += diff * diff;
                    }

                    if distance_squared.sqrt() < tolerance {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Count total active cells across all levels
    fn count_active_cells(&self) -> usize {
        self.mesh_hierarchy
            .levels
            .iter()
            .map(|level| level.cells.values().filter(|cell| cell.is_active).count())
            .sum()
    }

    /// Assess load balance quality
    fn assess_load_balance(&self) -> F {
        let total_cells = self.count_active_cells();
        if total_cells == 0 {
            return F::one(); // Empty mesh is perfectly balanced
        }

        // Calculate multiple load balance metrics
        let cell_distribution_balance = self.calculate_cell_distribution_balance();
        let computational_load_balance = self.calculate_computational_load_balance();
        let communication_overhead_balance = self.calculate_communication_balance();
        let memory_distribution_balance = self.calculate_memory_balance();

        // Weighted combination of different balance metrics
        let weight_cell = F::from(0.3).unwrap();
        let weight_compute = F::from(0.4).unwrap();
        let weight_comm = F::from(0.2).unwrap();
        let weight_memory = F::from(0.1).unwrap();

        let overall_balance = weight_cell * cell_distribution_balance
            + weight_compute * computational_load_balance
            + weight_comm * communication_overhead_balance
            + weight_memory * memory_distribution_balance;

        // Clamp to [0, 1] range where 1.0 = perfect balance
        overall_balance.min(F::one()).max(F::zero())
    }

    /// Calculate cell count distribution balance across levels
    fn calculate_cell_distribution_balance(&self) -> F {
        if self.mesh_hierarchy.levels.is_empty() {
            return F::one();
        }

        // Calculate cells per level
        let mut cells_per_level: Vec<usize> = Vec::new();
        let mut total_cells = 0;

        for level in &self.mesh_hierarchy.levels {
            let active_cells = level.cells.values().filter(|c| c.is_active).count();
            cells_per_level.push(active_cells);
            total_cells += active_cells;
        }

        if total_cells == 0 {
            return F::one();
        }

        // Calculate variance in cell distribution
        let mean_cells = total_cells as f64 / cells_per_level.len() as f64;
        let variance: f64 = cells_per_level
            .iter()
            .map(|&count| {
                let diff = count as f64 - mean_cells;
                diff * diff
            })
            .sum::<f64>()
            / cells_per_level.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean_cells > 0.0 {
            std_dev / mean_cells
        } else {
            0.0
        };

        // Convert to balance score (lower variation = better balance)
        let balance = (1.0 - coefficient_of_variation.min(1.0)).max(0.0);
        F::from(balance).unwrap_or(F::zero())
    }

    /// Calculate computational load balance based on cell error estimates
    fn calculate_computational_load_balance(&self) -> F {
        let mut level_computational_loads: Vec<F> = Vec::new();
        let mut total_load = F::zero();

        for level in &self.mesh_hierarchy.levels {
            let mut level_load = F::zero();

            for cell in level.cells.values() {
                if cell.is_active {
                    // Computational cost is proportional to error estimate and refinement complexity
                    let cell_cost = cell.error_estimate * cell.size * cell.size; // O(h^2) scaling
                    level_load += cell_cost;
                }
            }

            level_computational_loads.push(level_load);
            total_load += level_load;
        }

        if total_load <= F::zero() {
            return F::one();
        }

        // Calculate coefficient of variation for computational loads
        let mean_load = total_load / F::from(level_computational_loads.len()).unwrap();
        let mut variance = F::zero();

        for &load in &level_computational_loads {
            let diff = load - mean_load;
            variance += diff * diff;
        }

        variance /= F::from(level_computational_loads.len()).unwrap();
        let std_dev = variance.sqrt();

        let coeff_var = if mean_load > F::zero() {
            std_dev / mean_load
        } else {
            F::zero()
        };

        // Convert to balance score
        let balance = F::one() - coeff_var.min(F::one());
        balance.max(F::zero())
    }

    /// Calculate communication balance based on ghost cell overhead
    fn calculate_communication_balance(&self) -> F {
        let mut level_comm_costs: Vec<F> = Vec::new();
        let mut total_comm_cost = F::zero();

        for (level_idx, level) in self.mesh_hierarchy.levels.iter().enumerate() {
            let active_cells = level.cells.values().filter(|c| c.is_active).count();
            let ghost_cells = self
                .mesh_hierarchy
                .ghost_cells
                .get(&level_idx)
                .map(|ghosts| ghosts.len())
                .unwrap_or(0);

            // Communication cost is proportional to ghost cells per active cell
            let comm_cost = if active_cells > 0 {
                F::from(ghost_cells as f64 / active_cells as f64).unwrap_or(F::zero())
            } else {
                F::zero()
            };

            level_comm_costs.push(comm_cost);
            total_comm_cost += comm_cost;
        }

        if level_comm_costs.is_empty() || total_comm_cost <= F::zero() {
            return F::one();
        }

        // Calculate variance in communication costs
        let mean_comm = total_comm_cost / F::from(level_comm_costs.len()).unwrap();
        let mut variance = F::zero();

        for &cost in &level_comm_costs {
            let diff = cost - mean_comm;
            variance += diff * diff;
        }

        variance /= F::from(level_comm_costs.len()).unwrap();
        let std_dev = variance.sqrt();

        let coeff_var = if mean_comm > F::zero() {
            std_dev / mean_comm
        } else {
            F::zero()
        };

        // Convert to balance score
        let balance = F::one() - coeff_var.min(F::one());
        balance.max(F::zero())
    }

    /// Calculate memory distribution balance
    fn calculate_memory_balance(&self) -> F {
        let mut level_memory_usage: Vec<F> = Vec::new();
        let mut total_memory = F::zero();

        for level in &self.mesh_hierarchy.levels {
            // Estimate memory usage: cells + solution data + neighbor lists
            let cell_count = level.cells.len();
            let total_neighbors: usize = level.cells.values().map(|c| c.neighbors.len()).sum();

            let solution_size: usize = level.cells.values().map(|c| c.solution.len()).sum();

            // Memory estimate (simplified)
            let memory_estimate = F::from(cell_count + total_neighbors + solution_size).unwrap();
            level_memory_usage.push(memory_estimate);
            total_memory += memory_estimate;
        }

        if level_memory_usage.is_empty() || total_memory <= F::zero() {
            return F::one();
        }

        // Calculate memory distribution balance
        let mean_memory = total_memory / F::from(level_memory_usage.len()).unwrap();
        let mut variance = F::zero();

        for &memory in &level_memory_usage {
            let diff = memory - mean_memory;
            variance += diff * diff;
        }

        variance /= F::from(level_memory_usage.len()).unwrap();
        let std_dev = variance.sqrt();

        let coeff_var = if mean_memory > F::zero() {
            std_dev / mean_memory
        } else {
            F::zero()
        };

        // Convert to balance score
        let balance = F::one() - coeff_var.min(F::one());
        balance.max(F::zero())
    }
}

// Refinement criterion implementations
impl<F: IntegrateFloat + Send + Sync> RefinementCriterion<F> for GradientRefinementCriterion<F> {
    fn evaluate(&self, cell: &AdaptiveCell<F>, neighbors: &[&AdaptiveCell<F>]) -> F {
        if neighbors.is_empty() {
            return F::zero();
        }

        let mut max_gradient = F::zero();

        for neighbor in neighbors {
            let gradient = if let Some(comp) = self.component {
                if comp < cell.solution.len() && comp < neighbor.solution.len() {
                    (cell.solution[comp] - neighbor.solution[comp]).abs() / cell.size
                } else {
                    F::zero()
                }
            } else {
                // Use L2 norm of solution difference
                let diff = &cell.solution - &neighbor.solution;
                diff.mapv(|x| x.powi(2)).sum().sqrt() / cell.size
            };

            max_gradient = max_gradient.max(gradient);
        }

        // Relative criterion
        let solution_magnitude = if let Some(comp) = self.component {
            cell.solution
                .get(comp)
                .map(|&x| x.abs())
                .unwrap_or(F::zero())
        } else {
            cell.solution.mapv(|x| x.abs()).sum()
        };

        if solution_magnitude > F::zero() {
            max_gradient / solution_magnitude
        } else {
            max_gradient
        }
    }

    fn name(&self) -> &'static str {
        "Gradient"
    }
}

impl<F: IntegrateFloat + Send + Sync> RefinementCriterion<F> for FeatureDetectionCriterion<F> {
    fn evaluate(&self, cell: &AdaptiveCell<F>, neighbors: &[&AdaptiveCell<F>]) -> F {
        let mut feature_score = F::zero();

        for &feature_type in &self.feature_types {
            match feature_type {
                FeatureType::SharpGradient => {
                    // Detect sharp gradients
                    if neighbors.len() >= 2 {
                        let gradients: Vec<F> = neighbors
                            .iter()
                            .map(|n| (&cell.solution - &n.solution).mapv(|x| x.abs()).sum())
                            .collect();

                        let max_grad = gradients.iter().fold(F::zero(), |acc, &x| acc.max(x));
                        let avg_grad = gradients.iter().fold(F::zero(), |acc, &x| acc + x)
                            / F::from(gradients.len()).unwrap();

                        if avg_grad > F::zero() {
                            feature_score += max_grad / avg_grad;
                        }
                    }
                }
                FeatureType::LocalExtremum => {
                    // Detect local extrema
                    let cell_value = cell.solution.mapv(|x| x.abs()).sum();
                    let mut is_extremum = true;

                    for neighbor in neighbors {
                        let neighbor_value = neighbor.solution.mapv(|x| x.abs()).sum();
                        if (neighbor_value - cell_value).abs() < self.threshold {
                            is_extremum = false;
                            break;
                        }
                    }

                    if is_extremum {
                        feature_score += F::one();
                    }
                }
                _ => {
                    // Other feature types would be implemented here
                }
            }
        }

        feature_score
    }

    fn name(&self) -> &'static str {
        "FeatureDetection"
    }
}

impl<F: IntegrateFloat + Send + Sync> RefinementCriterion<F> for CurvatureRefinementCriterion<F> {
    fn evaluate(&self, cell: &AdaptiveCell<F>, neighbors: &[&AdaptiveCell<F>]) -> F {
        if neighbors.len() < 2 {
            return F::zero();
        }

        // Estimate curvature using second differences
        let mut curvature = F::zero();

        for component in 0..cell.solution.len() {
            let center_value = cell.solution[component];
            let neighbor_values: Vec<F> = neighbors
                .iter()
                .filter_map(|n| n.solution.get(component).copied())
                .collect();

            if neighbor_values.len() >= 2 {
                // Simple second difference approximation
                let avg_neighbor = neighbor_values.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(neighbor_values.len()).unwrap();

                let second_diff = (avg_neighbor - center_value).abs() / (cell.size * cell.size);
                curvature += second_diff;
            }
        }

        curvature
    }

    fn name(&self) -> &'static str {
        "Curvature"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amr_manager_creation() {
        let initial_level = AdaptiveMeshLevel {
            level: 0,
            cells: HashMap::new(),
            grid_spacing: 1.0,
            boundary_cells: HashSet::new(),
        };

        let amr = AdvancedAMRManager::new(initial_level, 5, 0.01);
        assert_eq!(amr.max_levels, 5);
        assert_eq!(amr.mesh_hierarchy.levels.len(), 1);
    }

    #[test]
    fn test_gradient_criterion() {
        let cell = AdaptiveCell {
            id: CellId { level: 0, index: 0 },
            center: Array1::from_vec(vec![0.5, 0.5]),
            size: 0.1,
            solution: Array1::from_vec(vec![1.0]),
            error_estimate: 0.0,
            refinement_flag: RefinementFlag::None,
            is_active: true,
            neighbors: vec![],
            parent: None,
            children: vec![],
        };

        let neighbor = AdaptiveCell {
            id: CellId { level: 0, index: 1 },
            center: Array1::from_vec(vec![0.6, 0.5]),
            size: 0.1,
            solution: Array1::from_vec(vec![2.0]),
            error_estimate: 0.0,
            refinement_flag: RefinementFlag::None,
            is_active: true,
            neighbors: vec![],
            parent: None,
            children: vec![],
        };

        let criterion = GradientRefinementCriterion {
            component: Some(0),
            threshold: 1.0,
            relative_tolerance: 0.1,
        };

        let neighbors = vec![&neighbor];
        let result = criterion.evaluate(&cell, &neighbors);
        assert!(result > 0.0);
    }
}
