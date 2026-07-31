// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Common array operations implemented with the array protocol.
//!
//! This module provides implementations of common array operations using
//! the array protocol. These operations can work with any array type that
//! implements the ArrayProtocol trait, including GPU arrays, distributed arrays,
//! and custom third-party array implementations.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use ::ndarray::{Ix1, Ix2, IxDyn};

use crate::array_protocol::{
    get_implementing_args, ArrayFunction, ArrayProtocol, NdarrayWrapper, NotImplemented,
};
use crate::error::CoreError;
// Note: num_traits not needed for current implementation

/// Error type for array operations.
#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    /// The operation is not implemented for the given array types.
    #[error("Operation not implemented: {0}")]
    NotImplemented(String),
    /// The array shapes are incompatible for the operation.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    /// The array types are incompatible for the operation.
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),
    /// Other error during operation.
    #[error("Operation error: {0}")]
    Other(String),
}

impl From<NotImplemented> for OperationError {
    fn from(_: NotImplemented) -> Self {
        Self::NotImplemented("Operation not implemented for these array types".to_string())
    }
}

impl From<CoreError> for OperationError {
    fn from(err: CoreError) -> Self {
        Self::Other(err.to_string())
    }
}

// Define array operations using the array protocol

// Define a macro for implementing array operations
#[macro_export]
macro_rules! array_function_dispatch {
    // For normal functions
    (fn $name:ident($($arg:ident: $arg_ty:ty),*) -> Result<$ret:ty, $err:ty> $body:block, $funcname:expr) => {
        pub fn $name($($arg: $arg_ty),*) -> Result<$ret, $err> $body
    };

    // For normal functions with trailing commas
    (fn $name:ident($($arg:ident: $arg_ty:ty,)*) -> Result<$ret:ty, $err:ty> $body:block, $funcname:expr) => {
        pub fn $name($($arg: $arg_ty),*) -> Result<$ret, $err> $body
    };

    // For generic functions
    (fn $name:ident<$($type_param:ident $(: $type_bound:path)?),*>($($arg:ident: $arg_ty:ty),*) -> Result<$ret:ty, $err:ty> $body:block, $funcname:expr) => {
        pub fn $name <$($type_param $(: $type_bound)?),*>($($arg: $arg_ty),*) -> Result<$ret, $err> $body
    };

    // For generic functions with trailing commas
    (fn $name:ident<$($type_param:ident $(: $type_bound:path)?),*>($($arg:ident: $arg_ty:ty,)*) -> Result<$ret:ty, $err:ty> $body:block, $funcname:expr) => {
        pub fn $name <$($type_param $(: $type_bound)?),*>($($arg: $arg_ty),*) -> Result<$ret, $err> $body
    };
}

// Matrix multiplication operation
array_function_dispatch!(
    fn matmul(
        a: &dyn ArrayProtocol,
        b: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_b = Box::new(b.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a, boxed_b];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::matmul", &boxed_args);
        if implementing_args.is_empty() {
            // Comprehensive fallback implementation for ndarray types

            // f64 types with Ix2 dimension (static dimension size)
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            ) {
                let a_array_owned = a_array.as_array().clone();
                let b_array_owned = b_array.as_array().clone();
                let (m, k) = a_array_owned.dim();
                let (_, n) = b_array_owned.dim();
                let mut result = crate::ndarray::Array2::<f64>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_array_owned[[i, l]] * b_array_owned[[l, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f64 types with IxDyn dimension (dynamic dimension size)
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
            ) {
                let a_array_owned = a_array.as_array().to_owned();
                let b_array_owned = b_array.as_array().to_owned();
                let a_dim = a_array_owned.shape();
                let b_dim = b_array_owned.shape();
                if a_dim.len() != 2 || b_dim.len() != 2 || a_dim[1] != b_dim[0] {
                    return Err(OperationError::ShapeMismatch(format!(
                        "Invalid shapes for matmul: {a_dim:?} and {b_dim:?}"
                    )));
                }
                let (m, k) = (a_dim[0], a_dim[1]);
                let n = b_dim[1];
                let mut result = crate::ndarray::Array2::<f64>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_array_owned[[i, l]] * b_array_owned[[l, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f32 types with Ix2 dimension
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
            ) {
                let a_array_owned = a_array.as_array().clone();
                let b_array_owned = b_array.as_array().clone();
                let (m, k) = a_array_owned.dim();
                let (_, n) = b_array_owned.dim();
                let mut result = crate::ndarray::Array2::<f32>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_array_owned[[i, l]] * b_array_owned[[l, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f32 types with IxDyn dimension
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
            ) {
                let a_array_owned = a_array.as_array().to_owned();
                let b_array_owned = b_array.as_array().to_owned();
                let a_dim = a_array_owned.shape();
                let b_dim = b_array_owned.shape();
                if a_dim.len() != 2 || b_dim.len() != 2 || a_dim[1] != b_dim[0] {
                    return Err(OperationError::ShapeMismatch(format!(
                        "Invalid shapes for matmul: {a_dim:?} and {b_dim:?}"
                    )));
                }
                let (m, k) = (a_dim[0], a_dim[1]);
                let n = b_dim[1];
                let mut result = crate::ndarray::Array2::<f32>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_array_owned[[i, l]] * b_array_owned[[l, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            return Err(OperationError::NotImplemented(
                "matmul not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::matmul"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone()), Box::new(b.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::matmul"
);

// Element-wise addition operation
array_function_dispatch!(
    fn add(
        a: &dyn ArrayProtocol,
        b: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_b = Box::new(b.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a, boxed_b];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::add", &boxed_args);
        if implementing_args.is_empty() {
            // Comprehensive fallback implementation for ndarray types

            // f64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
            ) {
                let result = a_array.as_array() + b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            return Err(OperationError::NotImplemented(
                "add not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::add"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone()), Box::new(b.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::add"
);

// Element-wise subtraction operation
array_function_dispatch!(
    fn subtract(
        a: &dyn ArrayProtocol,
        b: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_b = Box::new(b.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a, boxed_b];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::subtract", &boxed_args);
        if implementing_args.is_empty() {
            // Comprehensive fallback implementation for ndarray types

            // f64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
            ) {
                let result = a_array.as_array() - b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            return Err(OperationError::NotImplemented(
                "subtract not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::subtract"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone()), Box::new(b.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::subtract"
);

// Element-wise multiplication operation
array_function_dispatch!(
    fn multiply(
        a: &dyn ArrayProtocol,
        b: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_b = Box::new(b.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a, boxed_b];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::multiply", &boxed_args);
        if implementing_args.is_empty() {
            // Comprehensive fallback implementation for ndarray types

            // f64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // f32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i32 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix1>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, Ix2>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            // i64 types
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix1>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, Ix2>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            if let (Some(a_array), Some(b_array)) = (
                a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
                b.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
            ) {
                let result = a_array.as_array() * b_array.as_array();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }

            return Err(OperationError::NotImplemented(
                "multiply not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::multiply"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone()), Box::new(b.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::multiply"
);

// Reduction operation: sum
array_function_dispatch!(
    fn sum(a: &dyn ArrayProtocol, axis: Option<usize>) -> Result<Box<dyn Any>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::sum", &boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            // Try with Ix2 dimension first (most common case)
            if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                match axis {
                    Some(ax) => {
                        let result = a_array.as_array().sum_axis(crate::ndarray::Axis(ax));
                        return Ok(Box::new(NdarrayWrapper::new(result)));
                    }
                    None => {
                        let result = a_array.as_array().sum();
                        return Ok(Box::new(result));
                    }
                }
            }
            // Try with IxDyn dimension (used in tests)
            else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
                match axis {
                    Some(ax) => {
                        let result = a_array.as_array().sum_axis(crate::ndarray::Axis(ax));
                        return Ok(Box::new(NdarrayWrapper::new(result)));
                    }
                    None => {
                        let result = a_array.as_array().sum();
                        return Ok(Box::new(result));
                    }
                }
            }
            return Err(OperationError::NotImplemented(
                "sum not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        if let Some(ax) = axis {
            kwargs.insert("axis".to_string(), Box::new(ax) as Box<dyn Any>);
        }

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::sum"),
            &[TypeId::of::<Box<dyn Any>>()],
            &[Box::new(a.box_clone())],
            &kwargs,
        )?;

        Ok(result)
    },
    "scirs2::array_protocol::operations::sum"
);

// Transpose operation
array_function_dispatch!(
    fn transpose(a: &dyn ArrayProtocol) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::transpose", &boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            // Try with Ix2 dimension first (most common case)
            if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                let result = a_array.as_array().t().to_owned();
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            // Try with IxDyn dimension (used in tests)
            else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
                // For dynamic dimension, we need to check if it's a 2D array
                let a_dim = a_array.as_array().shape();
                if a_dim.len() != 2 {
                    return Err(OperationError::ShapeMismatch(format!(
                        "Transpose requires a 2D array, got shape: {a_dim:?}"
                    )));
                }

                // Create a transposed array
                let (m, n) = (a_dim[0], a_dim[1]);
                let mut result = crate::ndarray::Array2::<f64>::zeros((n, m));

                // Fill the transposed array
                for i in 0..m {
                    for j in 0..n {
                        result[[j, i]] = a_array.as_array()[[i, j]];
                    }
                }

                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            return Err(OperationError::NotImplemented(
                "transpose not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::transpose"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::transpose"
);

// Element-wise application of a function implementation
#[allow(dead_code)]
pub fn apply_elementwise<F>(
    a: &dyn ArrayProtocol,
    f: F,
) -> Result<Box<dyn ArrayProtocol>, OperationError>
where
    F: Fn(f64) -> f64 + 'static,
{
    // Get implementing args
    let boxed_a = Box::new(a.box_clone());
    let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
    let implementing_args = get_implementing_args(
        "scirs2::array_protocol::operations::apply_elementwise",
        &boxed_args,
    );
    if implementing_args.is_empty() {
        // Fallback implementation for ndarray types
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
            let result = a_array.as_array().mapv(f);
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        return Err(OperationError::NotImplemented(
            "apply_elementwise not implemented for this array type".to_string(),
        ));
    }

    // For this operation, we need to handle the function specially
    // In a real implementation, we would need to serialize the function or use a predefined set
    // Here we'll just use the fallback implementation for simplicity
    if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        let result = a_array.as_array().mapv(f);
        Ok(Box::new(NdarrayWrapper::new(result)))
    } else {
        Err(OperationError::NotImplemented(
            "apply_elementwise not implemented for this array type".to_string(),
        ))
    }
}

// Concatenate operation
array_function_dispatch!(
    fn concatenate(
        arrays: &[&dyn ArrayProtocol],
        axis: usize,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        if arrays.is_empty() {
            return Err(OperationError::Other(
                "No arrays provided for concatenation".to_string(),
            ));
        }

        // Convert each array to Box<dyn Any>
        let boxed_arrays: Vec<Box<dyn Any>> = arrays
            .iter()
            .map(|&a| Box::new(a.box_clone()) as Box<dyn Any>)
            .collect();

        let implementing_args = get_implementing_args(
            "scirs2::array_protocol::operations::concatenate",
            &boxed_arrays,
        );
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            // For simplicity, we'll handle just the 2D f64 case
            let mut ndarray_arrays = Vec::new();
            for &array in arrays {
                if let Some(a) = array.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                    ndarray_arrays.push(a.as_array().view());
                } else {
                    return Err(OperationError::TypeMismatch(
                        "All arrays must be NdarrayWrapper<f64, Ix2>".to_string(),
                    ));
                }
            }

            let result = match crate::ndarray::stack(crate::ndarray::Axis(axis), &ndarray_arrays) {
                Ok(arr) => arr,
                Err(e) => return Err(OperationError::Other(format!("Concatenation failed: {e}"))),
            };

            return Ok(Box::new(NdarrayWrapper::new(result)));
        }

        // Delegate to the implementation
        let array_boxed_clones: Vec<Box<dyn Any>> = arrays
            .iter()
            .map(|&a| Box::new(a.box_clone()) as Box<dyn Any>)
            .collect();

        let mut kwargs = HashMap::new();
        kwargs.insert(axis.to_string(), Box::new(axis) as Box<dyn Any>);

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::concatenate"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &array_boxed_clones,
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::concatenate"
);

// Reshape operation
array_function_dispatch!(
    fn reshape(
        a: &dyn ArrayProtocol,
        shape: &[usize],
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::reshape", &boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            // Try with Ix2 dimension first (most common case)
            if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                let result = match a_array.as_array().clone().into_shape_with_order(shape) {
                    Ok(arr) => arr,
                    Err(e) => {
                        return Err(OperationError::ShapeMismatch(format!(
                            "Reshape failed: {e}"
                        )))
                    }
                };
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            // Try with IxDyn dimension (used in tests)
            else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
                let result = match a_array.as_array().clone().into_shape_with_order(shape) {
                    Ok(arr) => arr,
                    Err(e) => {
                        return Err(OperationError::ShapeMismatch(format!(
                            "Reshape failed: {e}"
                        )))
                    }
                };
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            return Err(OperationError::NotImplemented(
                "reshape not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert(
            "shape".to_string(),
            Box::new(shape.to_vec()) as Box<dyn Any>,
        );

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::reshape"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone())],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::reshape"
);

// Linear algebra operations

// Type alias for SVD return type
type SVDResult = (
    Box<dyn ArrayProtocol>,
    Box<dyn ArrayProtocol>,
    Box<dyn ArrayProtocol>,
);

// SVD decomposition operation
array_function_dispatch!(
    fn svd(a: &dyn ArrayProtocol) -> Result<SVDResult, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::svd", &boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                // Real SVD via the OxiBLAS-backed LAPACK bindings (only
                // available when the `linalg` feature is enabled, since that
                // feature gates the optional oxiblas-* dependencies).
                #[cfg(feature = "linalg")]
                {
                    let svd_result = crate::linalg::svd_ndarray(a_array.as_array())
                        .map_err(|e| OperationError::Other(format!("SVD failed: {e}")))?;
                    return Ok((
                        Box::new(NdarrayWrapper::new(svd_result.u)),
                        Box::new(NdarrayWrapper::new(svd_result.s)),
                        Box::new(NdarrayWrapper::new(svd_result.vt)),
                    ));
                }
                #[cfg(not(feature = "linalg"))]
                {
                    return Err(OperationError::NotImplemented(
                        "svd requires the `linalg` feature (OxiBLAS-backed decomposition) to be enabled"
                            .to_string(),
                    ));
                }
            }
            return Err(OperationError::NotImplemented(
                "svd not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::svd"),
            &[TypeId::of::<(
                Box<dyn ArrayProtocol>,
                Box<dyn ArrayProtocol>,
                Box<dyn ArrayProtocol>,
            )>()],
            &[Box::new(a.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<(
            Box<dyn ArrayProtocol>,
            Box<dyn ArrayProtocol>,
            Box<dyn ArrayProtocol>,
        )>() {
            Ok(tuple) => Ok(*tuple),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to SVD tuple".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::svd"
);

// Inverse operation
array_function_dispatch!(
    fn inverse(a: &dyn ArrayProtocol) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_a = Box::new(a.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
        let implementing_args =
            get_implementing_args("scirs2::array_protocol::operations::inverse", &boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                let (m, n) = a_array.as_array().dim();
                if m != n {
                    return Err(OperationError::ShapeMismatch(
                        "Matrix must be square for inversion".to_string(),
                    ));
                }

                // Real matrix inversion via the OxiBLAS-backed LAPACK
                // bindings (only available when the `linalg` feature is
                // enabled, since that feature gates the optional oxiblas-*
                // dependencies).
                #[cfg(feature = "linalg")]
                {
                    let inv = crate::linalg::inv_ndarray(a_array.as_array()).map_err(|e| {
                        OperationError::Other(format!("Matrix inversion failed: {e}"))
                    })?;
                    return Ok(Box::new(NdarrayWrapper::new(inv)));
                }
                #[cfg(not(feature = "linalg"))]
                {
                    return Err(OperationError::NotImplemented(
                        "inverse requires the `linalg` feature (OxiBLAS-backed decomposition) to be enabled"
                            .to_string(),
                    ));
                }
            }
            return Err(OperationError::NotImplemented(
                "inverse not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &ArrayFunction::new("scirs2::array_protocol::operations::inverse"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(a.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast result to ArrayProtocol".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::operations::inverse"
);

// Scalar multiplication operation (implemented without macro due to generic constraints)
#[allow(dead_code)]
pub fn multiply_by_scalar_f64(
    a: &dyn ArrayProtocol,
    scalar: f64,
) -> Result<Box<dyn ArrayProtocol>, OperationError> {
    // Get implementing args
    let boxed_a = Box::new(a.box_clone());
    let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
    let implementing_args = get_implementing_args(
        "scirs2::array_protocol::operations::multiply_by_scalar_f64",
        &boxed_args,
    );
    if implementing_args.is_empty() {
        // Fallback implementation for ndarray types
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        return Err(OperationError::NotImplemented(
            "multiply_by_scalar not implemented for this array type".to_string(),
        ));
    }

    // Delegate to the implementation
    let mut kwargs = HashMap::new();
    kwargs.insert(scalar.to_string(), Box::new(scalar) as Box<dyn Any>);

    let array_ref = implementing_args[0].1;

    let result = array_ref.array_function(
        &ArrayFunction::new("scirs2::array_protocol::operations::multiply_by_scalar_f64"),
        &[TypeId::of::<Box<dyn ArrayProtocol>>()],
        &[Box::new(a.box_clone())],
        &kwargs,
    )?;

    // Try to downcast the result
    match result.downcast::<Box<dyn ArrayProtocol>>() {
        Ok(array) => Ok(*array),
        Err(_) => Err(OperationError::Other(
            "Failed to downcast result to ArrayProtocol".to_string(),
        )),
    }
}

// Scalar multiplication for f32
#[allow(dead_code)]
pub fn multiply_by_scalar_f32(
    a: &dyn ArrayProtocol,
    scalar: f32,
) -> Result<Box<dyn ArrayProtocol>, OperationError> {
    // Get implementing args
    let boxed_a = Box::new(a.box_clone());
    let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
    let implementing_args = get_implementing_args(
        "scirs2::array_protocol::operations::multiply_by_scalar_f32",
        &boxed_args,
    );
    if implementing_args.is_empty() {
        // Fallback implementation for ndarray types
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix1>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, Ix2>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>() {
            let result = a_array.as_array() * scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        return Err(OperationError::NotImplemented(
            "multiply_by_scalar not implemented for this array type".to_string(),
        ));
    }

    // Delegate to the implementation
    let mut kwargs = HashMap::new();
    kwargs.insert(scalar.to_string(), Box::new(scalar) as Box<dyn Any>);

    let array_ref = implementing_args[0].1;

    let result = array_ref.array_function(
        &ArrayFunction::new("scirs2::array_protocol::operations::multiply_by_scalar_f32"),
        &[TypeId::of::<Box<dyn ArrayProtocol>>()],
        &[Box::new(a.box_clone())],
        &kwargs,
    )?;

    // Try to downcast the result
    match result.downcast::<Box<dyn ArrayProtocol>>() {
        Ok(array) => Ok(*array),
        Err(_) => Err(OperationError::Other(
            "Failed to downcast result to ArrayProtocol".to_string(),
        )),
    }
}

// Scalar division for f64
#[allow(dead_code)]
pub fn divide_by_scalar_f64(
    a: &dyn ArrayProtocol,
    scalar: f64,
) -> Result<Box<dyn ArrayProtocol>, OperationError> {
    // Get implementing args
    let boxed_a = Box::new(a.box_clone());
    let boxed_args: Vec<Box<dyn Any>> = vec![boxed_a];
    let implementing_args = get_implementing_args(
        "scirs2::array_protocol::operations::divide_by_scalar_f64",
        &boxed_args,
    );
    if implementing_args.is_empty() {
        // Fallback implementation for ndarray types
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>() {
            let result = a_array.as_array() / scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
            let result = a_array.as_array() / scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
            let result = a_array.as_array() / scalar;
            return Ok(Box::new(NdarrayWrapper::new(result)));
        }
        return Err(OperationError::NotImplemented(
            "divide_by_scalar not implemented for this array type".to_string(),
        ));
    }

    // Delegate to the implementation
    let mut kwargs = HashMap::new();
    kwargs.insert(scalar.to_string(), Box::new(scalar) as Box<dyn Any>);

    let array_ref = implementing_args[0].1;

    let result = array_ref.array_function(
        &ArrayFunction::new("scirs2::array_protocol::operations::divide_by_scalar_f64"),
        &[TypeId::of::<Box<dyn ArrayProtocol>>()],
        &[Box::new(a.box_clone())],
        &kwargs,
    )?;

    // Try to downcast the result
    match result.downcast::<Box<dyn ArrayProtocol>>() {
        Ok(array) => Ok(*array),
        Err(_) => Err(OperationError::Other(
            "Failed to downcast result to ArrayProtocol".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol::{self, NdarrayWrapper};
    use ::ndarray::{array, Array2};

    #[test]
    fn test_operations_with_ndarray() {
        use ::ndarray::array;

        // Initialize the array protocol system
        array_protocol::init();

        // Create regular ndarrays
        let a = Array2::<f64>::eye(3);
        let b = Array2::<f64>::ones((3, 3));

        // Wrap them in NdarrayWrapper
        let wrapped_a = NdarrayWrapper::new(a.clone());
        let wrapped_b = NdarrayWrapper::new(b.clone());

        // Test matrix multiplication
        let matmul_result = matmul(&wrapped_a, &wrapped_b).expect("matmul should succeed");
        let matmul_array = matmul_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("matmul should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(matmul_array.as_array(), &a.dot(&b));

        // Test addition
        let add_result = add(&wrapped_a, &wrapped_b).expect("add should succeed");
        let add_array = add_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("add should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(add_array.as_array(), &(a.clone() + b.clone()));

        // Test multiplication
        let multiply_result = multiply(&wrapped_a, &wrapped_b).expect("multiply should succeed");
        let multiply_array = multiply_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("multiply should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(multiply_array.as_array(), &(a.clone() * b.clone()));

        // Test sum
        let sum_result = sum(&wrapped_a, None).expect("sum should succeed");
        let sum_value = sum_result
            .downcast_ref::<f64>()
            .expect("sum should return an f64");
        assert_eq!(*sum_value, a.sum());

        // Test transpose
        let transpose_result = transpose(&wrapped_a).expect("transpose should succeed");
        let transpose_array = transpose_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("transpose should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(transpose_array.as_array(), &a.t().to_owned());

        // Test reshape. `reshape`'s dispatch passes `shape` as a `Vec<usize>`
        // (a runtime-length container), and `ndarray`'s `IntoDimension` maps
        // `Vec<usize>` to `IxDyn` regardless of how many elements it holds —
        // not to a fixed-size `Ix1`/`Ix2`/etc., even when the vec happens to
        // have exactly 1 element. So the result is a `NdarrayWrapper<f64,
        // IxDyn>`, not `NdarrayWrapper<f64, Ix1>`.
        let c = array![[1., 2., 3.], [4., 5., 6.]];
        let wrapped_c = NdarrayWrapper::new(c.clone());
        let reshape_result = reshape(&wrapped_c, &[6]).expect("reshape should succeed");
        let result_array = reshape_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
            .expect("reshape should return a NdarrayWrapper<f64, IxDyn>");
        let expected = c
            .clone()
            .into_shape_with_order(6)
            .expect("Operation failed")
            .into_dyn();
        assert_eq!(result_array.as_array(), &expected);
    }

    /// Regression test for the `get_implementing_args` dispatch bug: before
    /// the fix, `implementing_args` was never empty for `NdarrayWrapper`
    /// arguments (since `NdarrayWrapper` boxed as `Box<dyn ArrayProtocol>`
    /// always downcast successfully), so `subtract`/`multiply_by_scalar_f64`/
    /// `divide_by_scalar_f64` always delegated straight into
    /// `NdarrayWrapper::array_function` — which doesn't have a match arm for
    /// any of these operations — and so always returned
    /// `Err(OperationError::NotImplemented(..))` instead of ever reaching
    /// the correct hand-written fallback code below. Uses non-constant data
    /// so a fabricated all-same-value result couldn't pass by accident.
    #[test]
    fn test_subtract_and_scalar_ops_reach_fallback() {
        array_protocol::init();

        let a = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64 + 1.0);
        let b = Array2::from_shape_fn((3, 3), |(i, j)| (i as f64) * 0.5 - (j as f64) * 0.25);

        let wrapped_a = NdarrayWrapper::new(a.clone());
        let wrapped_b = NdarrayWrapper::new(b.clone());

        let subtract_result = subtract(&wrapped_a, &wrapped_b).expect("subtract should succeed");
        let subtract_array = subtract_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("subtract should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(subtract_array.as_array(), &(a.clone() - b.clone()));

        let scaled =
            multiply_by_scalar_f64(&wrapped_a, 2.5).expect("multiply_by_scalar_f64 should succeed");
        let scaled_array = scaled
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("multiply_by_scalar_f64 should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(scaled_array.as_array(), &(a.clone() * 2.5));

        let divided =
            divide_by_scalar_f64(&wrapped_a, 4.0).expect("divide_by_scalar_f64 should succeed");
        let divided_array = divided
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("divide_by_scalar_f64 should return a NdarrayWrapper<f64, Ix2>");
        assert_eq!(divided_array.as_array(), &(a.clone() / 4.0));
    }

    /// Regression test for the fake `svd`/`inverse` placeholders: before the
    /// fix, both always returned an identity/ones "decomposition" regardless
    /// of the input, and — until the `get_implementing_args` dispatch fix —
    /// were also unreachable dead code. Uses a non-symmetric, non-constant
    /// matrix so an identity/ones fabrication provably fails reconstruction.
    #[cfg(feature = "linalg")]
    #[test]
    fn test_svd_reconstructs_original_matrix() {
        array_protocol::init();

        let a = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 1) as f64) * 1.7 - (j as f64) * 0.9);
        let wrapped_a = NdarrayWrapper::new(a.clone());

        let (u, s, vt) = svd(&wrapped_a).expect("svd should succeed");
        let u = u
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("U should be a NdarrayWrapper<f64, Ix2>")
            .as_array()
            .clone();
        let s = s
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ::ndarray::Ix1>>()
            .expect("S should be a NdarrayWrapper<f64, Ix1>")
            .as_array()
            .clone();
        let vt = vt
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("Vt should be a NdarrayWrapper<f64, Ix2>")
            .as_array()
            .clone();

        // A real SVD must NOT be the identity/ones placeholder: the old
        // placeholder returned `s` as all-ones regardless of input, which a
        // non-constant, non-orthogonal input matrix like `a` can never
        // actually produce.
        assert!(
            s.iter().any(|&v| (v - 1.0).abs() > 1e-6),
            "singular values look like the old all-ones placeholder: {s:?}"
        );

        // Build Sigma with whatever shape U/Vt actually came back with
        // (implementations differ on full vs. economy SVD), then verify the
        // defining reconstruction property A = U * Sigma * V^T.
        let mut sigma = Array2::<f64>::zeros((u.ncols(), vt.nrows()));
        for i in 0..s.len() {
            sigma[[i, i]] = s[i];
        }
        let reconstructed = u.dot(&sigma).dot(&vt);

        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-8,
                    "SVD reconstruction mismatch at ({i}, {j}): {reconstructed_val} vs {orig_val}",
                    reconstructed_val = reconstructed[[i, j]],
                    orig_val = a[[i, j]]
                );
            }
        }
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_inverse_is_a_real_inverse() {
        array_protocol::init();

        // A non-symmetric, non-trivial invertible matrix.
        let a = array![[4.0, 3.0, 0.0], [2.0, 5.0, 1.0], [1.0, 0.0, 3.0]];
        let wrapped_a = NdarrayWrapper::new(a.clone());

        let inv_result = inverse(&wrapped_a).expect("inverse should succeed");
        let inv_array = inv_result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .expect("inverse should return a NdarrayWrapper<f64, Ix2>")
            .as_array()
            .clone();

        // The old placeholder always returned the identity matrix, which
        // would trivially (and wrongly) satisfy `a.dot(&identity) == a`, not
        // `a.dot(&inv) == identity` — assert the real property instead.
        let product = a.dot(&inv_array);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-8,
                    "A * inv(A) mismatch at ({i}, {j}): {actual}",
                    actual = product[[i, j]]
                );
            }
        }
    }
}
