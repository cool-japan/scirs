//! Configuration types for advanced training methods
//!
//! This module defines the core configuration structures and data types
//! used throughout the advanced training framework.

use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Task data structure for meta-learning
#[derive(Debug, Clone)]
pub struct TaskData<F: Float + Debug> {
    /// Support set inputs (for adaptation)
    pub support_x: Array2<F>,
    /// Support set outputs
    pub support_y: Array2<F>,
    /// Query set inputs (for evaluation)
    pub query_x: Array2<F>,
    /// Query set outputs
    pub query_y: Array2<F>,
}

impl<F: Float + Debug> TaskData<F> {
    /// Create new task data
    pub fn new(
        support_x: Array2<F>,
        support_y: Array2<F>,
        query_x: Array2<F>,
        query_y: Array2<F>,
    ) -> Self {
        Self {
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    /// Get support set size
    pub fn support_size(&self) -> usize {
        self.support_x.nrows()
    }

    /// Get query set size
    pub fn query_size(&self) -> usize {
        self.query_x.nrows()
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.support_x.ncols()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.support_y.ncols()
    }
}
