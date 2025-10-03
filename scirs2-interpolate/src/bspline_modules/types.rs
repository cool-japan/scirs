//! Core type definitions for B-spline functionality
//!
//! This module contains all the foundational data types used throughout
//! the B-spline system.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Extrapolation mode for B-splines
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExtrapolateMode {
    /// Extrapolate based on the first and last polynomials
    #[default]
    Extrapolate,
    /// Periodic extrapolation
    Periodic,
    /// Return NaN for points outside the domain
    Nan,
    /// Return an error for points outside the domain
    Error,
}

/// Workspace for reusable memory allocations during B-spline evaluation
/// This reduces memory allocation overhead in hot paths
#[derive(Debug)]
pub struct BSplineWorkspace<T> {
    /// Reusable coefficient buffer for de Boor's algorithm
    pub(crate) coeffs: RefCell<Array1<T>>,
    /// Reusable buffer for polynomial evaluation
    pub(crate) poly_buf: RefCell<Array1<T>>,
    /// Reusable buffer for basis function computation
    pub(crate) basis_buf: RefCell<Array1<T>>,
    /// Reusable buffer for matrix operations
    pub(crate) matrix_buf: RefCell<Array2<T>>,
    /// Memory usage statistics
    pub(crate) memory_stats: RefCell<WorkspaceMemoryStats>,
}

/// Memory usage statistics for workspace optimization
#[derive(Debug, Clone, Default)]
pub struct WorkspaceMemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
    /// Number of allocations avoided by reuse
    pub allocations_avoided: usize,
    /// Number of times workspace was resized
    pub resize_count: usize,
    /// Total evaluation count
    pub evaluation_count: usize,
}

impl WorkspaceMemoryStats {
    /// Get memory efficiency ratio (allocations avoided / total evaluations)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.evaluation_count == 0 {
            0.0
        } else {
            self.allocations_avoided as f64 / self.evaluation_count as f64
        }
    }

    /// Get peak memory usage in MB
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Update memory usage statistics
    pub fn update_memory_usage(&mut self, current_bytes: usize) {
        self.current_memory_bytes = current_bytes;
        if current_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = current_bytes;
        }
    }

    /// Record an avoided allocation
    pub fn record_allocation_avoided(&mut self) {
        self.allocations_avoided += 1;
    }

    /// Record a workspace resize
    pub fn record_resize(&mut self) {
        self.resize_count += 1;
    }

    /// Record an evaluation
    pub fn record_evaluation(&mut self) {
        self.evaluation_count += 1;
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.peak_memory_bytes = 0;
        self.current_memory_bytes = 0;
        self.allocations_avoided = 0;
        self.resize_count = 0;
        self.evaluation_count = 0;
    }

    /// Get total memory saved (approximation based on avoided allocations)
    pub fn estimated_memory_saved_mb(&self, avg_allocation_size_bytes: usize) -> f64 {
        (self.allocations_avoided * avg_allocation_size_bytes) as f64 / (1024.0 * 1024.0)
    }
}

impl<T> BSplineWorkspace<T>
where
    T: Float + FromPrimitive + Zero + Clone,
{
    /// Create a new workspace with default capacity
    pub fn new() -> Self {
        Self::with_capacity(16, 16)
    }

    /// Create a new workspace with specified capacity
    pub fn with_capacity(array_capacity: usize, matrix_capacity: usize) -> Self {
        Self {
            coeffs: RefCell::new(Array1::zeros(array_capacity)),
            poly_buf: RefCell::new(Array1::zeros(array_capacity)),
            basis_buf: RefCell::new(Array1::zeros(array_capacity)),
            matrix_buf: RefCell::new(Array2::zeros((matrix_capacity, matrix_capacity))),
            memory_stats: RefCell::new(WorkspaceMemoryStats::default()),
        }
    }

    /// Get current memory usage in bytes
    pub fn current_memory_bytes(&self) -> usize {
        self.memory_stats.borrow().current_memory_bytes
    }

    /// Get peak memory usage in bytes
    pub fn peak_memory_bytes(&self) -> usize {
        self.memory_stats.borrow().peak_memory_bytes
    }

    /// Get memory efficiency ratio
    pub fn efficiency_ratio(&self) -> f64 {
        self.memory_stats.borrow().efficiency_ratio()
    }

    /// Get the total number of evaluations performed
    pub fn evaluation_count(&self) -> usize {
        self.memory_stats.borrow().evaluation_count
    }

    /// Get the number of allocations avoided
    pub fn allocations_avoided(&self) -> usize {
        self.memory_stats.borrow().allocations_avoided
    }

    /// Get workspace statistics
    pub fn get_statistics(&self) -> WorkspaceMemoryStats {
        self.memory_stats.borrow().clone()
    }

    /// Reset workspace statistics
    pub fn reset_statistics(&self) {
        self.memory_stats.borrow_mut().reset();
    }

    /// Ensure the coefficient buffer has at least the specified capacity
    pub fn ensure_coeffs_capacity(&self, capacity: usize) {
        let mut coeffs = self.coeffs.borrow_mut();
        if coeffs.len() < capacity {
            *coeffs = Array1::zeros(capacity);
            self.memory_stats.borrow_mut().record_resize();
        } else {
            self.memory_stats.borrow_mut().record_allocation_avoided();
        }
    }

    /// Ensure the polynomial buffer has at least the specified capacity
    pub fn ensure_poly_capacity(&self, capacity: usize) {
        let mut poly_buf = self.poly_buf.borrow_mut();
        if poly_buf.len() < capacity {
            *poly_buf = Array1::zeros(capacity);
            self.memory_stats.borrow_mut().record_resize();
        } else {
            self.memory_stats.borrow_mut().record_allocation_avoided();
        }
    }

    /// Ensure the basis buffer has at least the specified capacity
    pub fn ensure_basis_capacity(&self, capacity: usize) {
        let mut basis_buf = self.basis_buf.borrow_mut();
        if basis_buf.len() < capacity {
            *basis_buf = Array1::zeros(capacity);
            self.memory_stats.borrow_mut().record_resize();
        } else {
            self.memory_stats.borrow_mut().record_allocation_avoided();
        }
    }

    /// Ensure the matrix buffer has at least the specified capacity
    pub fn ensure_matrix_capacity(&self, rows: usize, cols: usize) {
        let mut matrix_buf = self.matrix_buf.borrow_mut();
        if matrix_buf.nrows() < rows || matrix_buf.ncols() < cols {
            *matrix_buf = Array2::zeros((rows, cols));
            self.memory_stats.borrow_mut().record_resize();
        } else {
            self.memory_stats.borrow_mut().record_allocation_avoided();
        }
    }

    /// Update memory usage and record evaluation
    pub fn record_evaluation(&self) {
        self.memory_stats.borrow_mut().record_evaluation();

        // Estimate current memory usage
        let coeffs_size = self.coeffs.borrow().len() * std::mem::size_of::<T>();
        let poly_size = self.poly_buf.borrow().len() * std::mem::size_of::<T>();
        let basis_size = self.basis_buf.borrow().len() * std::mem::size_of::<T>();
        let matrix_size = {
            let matrix = self.matrix_buf.borrow();
            matrix.nrows() * matrix.ncols() * std::mem::size_of::<T>()
        };

        let total_size = coeffs_size + poly_size + basis_size + matrix_size;
        self.memory_stats
            .borrow_mut()
            .update_memory_usage(total_size);
    }

    /// Clear all buffers and reset to minimum size
    pub fn clear(&self) {
        *self.coeffs.borrow_mut() = Array1::zeros(1);
        *self.poly_buf.borrow_mut() = Array1::zeros(1);
        *self.basis_buf.borrow_mut() = Array1::zeros(1);
        *self.matrix_buf.borrow_mut() = Array2::zeros((1, 1));
    }

    /// Get memory usage report as a formatted string
    pub fn memory_report(&self) -> String {
        let stats = self.get_statistics();
        format!(
            "BSpline Workspace Memory Report:\n\
             Peak Memory: {:.2} MB\n\
             Current Memory: {:.2} MB\n\
             Evaluations: {}\n\
             Allocations Avoided: {}\n\
             Efficiency Ratio: {:.2}%\n\
             Resizes: {}",
            stats.peak_memory_mb(),
            stats.current_memory_bytes as f64 / (1024.0 * 1024.0),
            stats.evaluation_count,
            stats.allocations_avoided,
            stats.efficiency_ratio() * 100.0,
            stats.resize_count
        )
    }
}

impl<T> Default for BSplineWorkspace<T>
where
    T: Float + FromPrimitive + Zero + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for BSplineWorkspace<T>
where
    T: Float + FromPrimitive + Zero + Clone,
{
    fn clone(&self) -> Self {
        Self {
            coeffs: RefCell::new(self.coeffs.borrow().clone()),
            poly_buf: RefCell::new(self.poly_buf.borrow().clone()),
            basis_buf: RefCell::new(self.basis_buf.borrow().clone()),
            matrix_buf: RefCell::new(self.matrix_buf.borrow().clone()),
            memory_stats: RefCell::new(self.memory_stats.borrow().clone()),
        }
    }
}

/// Builder pattern for creating workspaces with specific configurations
pub struct BSplineWorkspaceBuilder<T> {
    array_capacity: usize,
    matrix_capacity: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for BSplineWorkspaceBuilder<T> {
    fn default() -> Self {
        Self {
            array_capacity: 16,
            matrix_capacity: 16,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> BSplineWorkspaceBuilder<T>
where
    T: Float + FromPrimitive + Zero + Clone,
{
    /// Create a new workspace builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial capacity for array buffers
    pub fn with_array_capacity(mut self, capacity: usize) -> Self {
        self.array_capacity = capacity;
        self
    }

    /// Set the initial capacity for matrix buffers
    pub fn with_matrix_capacity(mut self, capacity: usize) -> Self {
        self.matrix_capacity = capacity;
        self
    }

    /// Build the workspace
    pub fn build(self) -> BSplineWorkspace<T> {
        BSplineWorkspace::with_capacity(self.array_capacity, self.matrix_capacity)
    }
}

/// Trait for types that can provide workspace optimization
pub trait WorkspaceProvider<T> {
    /// Get or create a workspace for B-spline operations
    fn get_workspace(&self) -> &BSplineWorkspace<T>;

    /// Check if workspace optimization is enabled
    fn is_workspace_enabled(&self) -> bool;
}

/// Configuration for workspace memory management
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    /// Initial array buffer capacity
    pub initial_array_capacity: usize,
    /// Initial matrix buffer capacity
    pub initial_matrix_capacity: usize,
    /// Maximum memory usage before forced cleanup (in MB)
    pub max_memory_mb: f64,
    /// Enable automatic memory management
    pub auto_memory_management: bool,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            initial_array_capacity: 16,
            initial_matrix_capacity: 16,
            max_memory_mb: 100.0, // 100 MB default limit
            auto_memory_management: true,
        }
    }
}
