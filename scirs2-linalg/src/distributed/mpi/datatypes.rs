//! MPI Datatype Definitions
//!
//! This module provides trait definitions and implementations for MPI-compatible
//! data types, enabling efficient communication of various numerical types.

use std::ffi::{c_int, c_void};
use std::collections::HashMap;

/// Trait for MPI-compatible data types
pub trait MPIDatatype {
    fn mpi_datatype() -> c_int;
}

impl MPIDatatype for f32 {
    fn mpi_datatype() -> c_int { 0 } // MPI_FLOAT
}

impl MPIDatatype for f64 {
    fn mpi_datatype() -> c_int { 1 } // MPI_DOUBLE
}

impl MPIDatatype for i32 {
    fn mpi_datatype() -> c_int { 2 } // MPI_INT
}

impl MPIDatatype for i64 {
    fn mpi_datatype() -> c_int { 3 } // MPI_LONG_LONG
}

impl MPIDatatype for u32 {
    fn mpi_datatype() -> c_int { 4 } // MPI_UNSIGNED
}

impl MPIDatatype for u64 {
    fn mpi_datatype() -> c_int { 5 } // MPI_UNSIGNED_LONG_LONG
}

impl MPIDatatype for i8 {
    fn mpi_datatype() -> c_int { 6 } // MPI_BYTE
}

impl MPIDatatype for u8 {
    fn mpi_datatype() -> c_int { 7 } // MPI_UNSIGNED_CHAR
}

/// MPI datatype for optimized communication
#[derive(Debug)]
pub struct MPIDatatype {
    type_handle: *mut c_void,
    elementsize: usize,
    is_committed: bool,
}

/// MPI reduction operations
#[derive(Debug, Clone, Copy)]
pub enum MPIReduceOp {
    Sum,
    Product,
    Max,
    Min,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Custom(u32),
}

impl MPIReduceOp {
    pub fn to_mpi_op(self) -> c_int {
        match self {
            MPIReduceOp::Sum => 0,
            MPIReduceOp::Product => 1,
            MPIReduceOp::Max => 2,
            MPIReduceOp::Min => 3,
            MPIReduceOp::LogicalAnd => 4,
            MPIReduceOp::LogicalOr => 5,
            MPIReduceOp::BitwiseAnd => 6,
            MPIReduceOp::BitwiseOr => 7,
            MPIReduceOp::BitwiseXor => 8,
            MPIReduceOp::Custom(op) => op as c_int,
        }
    }
}

/// Manager for derived MPI datatypes
#[derive(Debug)]
pub struct DatatypeManager {
    /// Derived datatypes for efficient communication
    derived_types: HashMap<String, MPIDatatype>,
}

impl DatatypeManager {
    /// Create a new datatype manager
    pub fn new() -> Self {
        Self {
            derived_types: HashMap::new(),
        }
    }

    /// Register a new derived datatype
    pub fn register_datatype(&mut self, name: String, datatype: MPIDatatype) {
        self.derived_types.insert(name, datatype);
    }

    /// Get a registered datatype
    pub fn get_datatype(&self, name: &str) -> Option<&MPIDatatype> {
        self.derived_types.get(name)
    }

    /// Check if a datatype is registered
    pub fn has_datatype(&self, name: &str) -> bool {
        self.derived_types.contains_key(name)
    }
}

impl Default for DatatypeManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for MPIDatatype {}
unsafe impl Sync for MPIDatatype {}

impl MPIDatatype {
    /// Create a new MPI datatype
    pub fn new(type_handle: *mut c_void, elementsize: usize) -> Self {
        Self {
            type_handle,
            elementsize,
            is_committed: false,
        }
    }

    /// Get the element size
    pub fn elementsize(&self) -> usize {
        self.elementsize
    }

    /// Check if the datatype is committed
    pub fn is_committed(&self) -> bool {
        self.is_committed
    }

    /// Commit the datatype (mark as ready for use)
    pub fn commit(&mut self) {
        self.is_committed = true;
    }

    /// Get the type handle
    pub fn type_handle(&self) -> *mut c_void {
        self.type_handle
    }
}