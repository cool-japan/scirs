//! # EnhancedSysIdConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedSysIdConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{EnhancedSysIdConfig, IdentificationMethod, ModelStructure};

impl Default for EnhancedSysIdConfig {
    fn default() -> Self {
        Self {
            model_structure: ModelStructure::ARX,
            method: IdentificationMethod::PEM,
            max_order: 10,
            order_selection: true,
            regularization: 0.0,
            forgetting_factor: 0.98,
            outlier_detection: false,
            cv_folds: Some(5),
            parallel: true,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}
