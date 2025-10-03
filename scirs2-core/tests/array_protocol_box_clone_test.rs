// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

use scirs2_core::array_protocol::{ArrayFunction, ArrayProtocol, NdarrayWrapper, NotImplemented};
use scirs2_core::ndarray_ext::Array2;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

// Tests for the box_clone functionality in the ArrayProtocol trait.
//
// This file contains tests for cloning Box<dyn ArrayProtocol> objects
// using simplified test implementations.

/// A simplified mock distributed array for testing box_clone
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MockDistributedArray {
    data: Vec<f64>, // Prefixed with underscore to indicate it's unused
    shape: Vec<usize>,
}

impl MockDistributedArray {
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl ArrayProtocol for MockDistributedArray {
    fn array_function(
        &self,
        _func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        Err(NotImplemented)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

/// A simplified mock GPU array for testing box_clone
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MockGPUArray {
    data: Vec<f64>, // Prefixed with underscore to indicate it's unused
    shape: Vec<usize>,
    device: String, // Prefixed with underscore to indicate it's unused
}

impl MockGPUArray {
    fn new(data: Vec<f64>, shape: Vec<usize>, device: String) -> Self {
        Self {
            data,
            shape,
            device,
        }
    }
}

impl ArrayProtocol for MockGPUArray {
    fn array_function(
        &self,
        _func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        Err(NotImplemented)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

/// A simplified JIT-enabled array for testing box_clone
#[derive(Debug, Clone)]
struct JITEnabledArray<T, A: Clone> {
    inner: A,
    phantom: PhantomData<T>,
}

impl<T, A: Clone> JITEnabledArray<T, A> {
    fn new(inner: A) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, A> ArrayProtocol for JITEnabledArray<T, A>
where
    T: Clone + Send + Sync + 'static,
    A: ArrayProtocol + Clone + Send + Sync + 'static,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        self.inner.array_function(func, types, args, kwargs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

/// Standalone test for box_clone implementation in ArrayProtocol
#[test]
#[allow(dead_code)]
fn test_array_protocol_box_clone() {
    // Test NdarrayWrapper
    let array = Array2::<f64>::ones((3, 3));
    let wrapped = NdarrayWrapper::new(array);
    let boxed: Box<dyn ArrayProtocol> = Box::new(wrapped);
    let cloned = boxed.clone();

    // Verify the clone is correct
    assert_eq!(cloned.shape(), &[3, 3]);

    // Test MockGPUArray
    let array = MockGPUArray::new(vec![1.0, 2.0, 3.0], vec![3], "cuda:0".to_string());
    let boxed: Box<dyn ArrayProtocol> = Box::new(array);
    let cloned = boxed.clone();

    // Verify the clone is correct
    assert_eq!(cloned.shape(), &[3]);

    // Test MockDistributedArray
    let array = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);
    let boxed: Box<dyn ArrayProtocol> = Box::new(array);
    let cloned = boxed.clone();

    // Verify the clone is correct
    assert_eq!(cloned.shape(), &[3]);
}

/// Test box_clone for NdarrayWrapper
#[test]
#[allow(dead_code)]
fn test_ndarray_wrapper_box_clone() {
    // Create a simple ndarray
    let array = Array2::<f64>::ones((3, 3));
    let wrapped = NdarrayWrapper::new(array);

    // Box the wrapper as an ArrayProtocol
    let boxed: Box<dyn ArrayProtocol> = Box::new(wrapped);

    // Clone the boxed wrapper
    let cloned = boxed.clone();

    // Verify the clone works correctly
    assert_eq!(cloned.shape(), &[3, 3]);

    // Try to downcast the clone to the original type
    let unwrapped = cloned
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>();
    assert!(unwrapped.is_some());
}

/// Test box_clone for MockDistributedArray
#[test]
#[allow(dead_code)]
fn test_mock_distributed_array_box_clone() {
    // Create a mock distributed array
    let array = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);

    // Box the array as an ArrayProtocol
    let boxed: Box<dyn ArrayProtocol> = Box::new(array);

    // Clone the boxed array
    let cloned = boxed.clone();

    // Verify the clone works correctly
    assert_eq!(cloned.shape(), &[3]);

    // Try to downcast the clone to the original type
    let unwrapped = cloned.as_any().downcast_ref::<MockDistributedArray>();
    assert!(unwrapped.is_some());
}

/// Test box_clone for MockGPUArray
#[test]
#[allow(dead_code)]
fn test_mock_gpu_array_box_clone() {
    // Create a mock GPU array
    let array = MockGPUArray::new(vec![1.0, 2.0, 3.0], vec![3], "cuda:0".to_string());

    // Box the array as an ArrayProtocol
    let boxed: Box<dyn ArrayProtocol> = Box::new(array);

    // Clone the boxed array
    let cloned = boxed.clone();

    // Verify the clone works correctly
    assert_eq!(cloned.shape(), &[3]);

    // Try to downcast the clone to the original type
    let unwrapped = cloned.as_any().downcast_ref::<MockGPUArray>();
    assert!(unwrapped.is_some());
}

/// Test box_clone for JIT-enabled arrays
#[test]
#[allow(dead_code)]
fn test_jit_array_box_clone() {
    // Create a regular array
    let array = Array2::<f64>::ones((10, 5));
    let wrapped = NdarrayWrapper::new(array);

    // Create a JIT-enabled array
    let jit_array = JITEnabledArray::<f64, _>::new(wrapped);

    // Box the array as an ArrayProtocol
    let boxed: Box<dyn ArrayProtocol> = Box::new(jit_array);

    // Clone the boxed array
    let cloned = boxed.clone();

    // Verify the clone works correctly
    assert_eq!(cloned.shape(), &[10, 5]);

    // Try to downcast the clone to the original type
    let unwrapped = cloned
        .as_any()
        .downcast_ref::<JITEnabledArray<f64, NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>>();
    assert!(unwrapped.is_some());
}

/// Test chained cloning of different array types
#[test]
#[allow(dead_code)]
fn test_chained_box_clone() {
    // Create a variety of array types
    let ndarray = NdarrayWrapper::new(Array2::<f64>::ones((3, 3)));
    let mock_distributed = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);
    let mock_gpu = MockGPUArray::new(vec![1.0, 2.0, 3.0], vec![3], "cuda:0".to_string());

    // Create a vector of boxed arrays
    let mut boxed_arrays: Vec<Box<dyn ArrayProtocol>> = vec![
        Box::new(ndarray),
        Box::new(mock_distributed),
        Box::new(mock_gpu),
    ];

    // Clone the entire vector
    let cloned_arrays = boxed_arrays.clone();

    // Verify all clones work correctly
    assert_eq!(boxed_arrays.len(), cloned_arrays.len());

    for (i, cloned) in cloned_arrays.iter().enumerate() {
        // Verify shapes match
        assert_eq!(cloned.shape(), boxed_arrays[i].shape());
    }

    // Push a cloned array back into the original vector (demonstrates composability)
    let another_clone = boxed_arrays[0].clone();
    boxed_arrays.push(another_clone);

    // Verify the new array was added
    assert_eq!(boxed_arrays.len(), 4);
}
