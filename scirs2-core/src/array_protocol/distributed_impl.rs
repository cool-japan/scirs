// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Distributed array implementation using the array protocol.
//!
//! This module provides a more complete implementation of distributed arrays
//! than the mock version in the main `array_protocol` module.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::array_protocol::{ArrayFunction, ArrayProtocol, DistributedArray, NotImplemented};
use crate::error::CoreResult;
use ::ndarray::{Array, Axis, Dimension, Slice};

/// A configuration for distributed array operations
#[derive(Debug, Clone, Default)]
pub struct DistributedConfig {
    /// Number of chunks to split the array into
    pub chunks: usize,

    /// Whether to balance the chunks across devices/nodes
    pub balance: bool,

    /// Strategy for distributing the array
    pub strategy: DistributionStrategy,

    /// Communication backend to use
    pub backend: DistributedBackend,
}

/// Strategies for distributing an array
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionStrategy {
    /// Split along the first axis
    RowWise,

    /// Split along the second axis
    ColumnWise,

    /// Split along all axes
    Blocks,

    /// Automatically determine the best strategy
    Auto,
}

impl Default for DistributionStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Communication backends for distributed arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBackend {
    /// Local multi-threading only
    Threaded,

    /// MPI-based distributed computing
    MPI,

    /// Custom TCP/IP based communication
    TCP,
}

impl Default for DistributedBackend {
    fn default() -> Self {
        Self::Threaded
    }
}

/// A chunk of a distributed array
#[derive(Debug, Clone)]
pub struct ArrayChunk<T, D>
where
    T: Clone + 'static,
    D: Dimension + 'static,
{
    /// The data in this chunk
    pub data: Array<T, D>,

    /// The global index of this chunk
    pub global_index: Vec<usize>,

    /// The node ID that holds this chunk
    pub nodeid: usize,
}

/// A distributed array implementation
pub struct DistributedNdarray<T, D>
where
    T: Clone + 'static,
    D: Dimension + 'static,
{
    /// Configuration for this distributed array
    pub config: DistributedConfig,

    /// The chunks that make up this array
    chunks: Vec<ArrayChunk<T, D>>,

    /// The global shape of the array
    shape: Vec<usize>,

    /// The unique ID of this distributed array
    id: String,
}

impl<T, D> Debug for DistributedNdarray<T, D>
where
    T: Clone + Debug + 'static,
    D: Dimension + Debug + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedNdarray")
            .field("config", &self.config)
            .field("chunks", &self.chunks.len())
            .field("shape", &self.shape)
            .field("id", &self.id)
            .finish()
    }
}

impl<T, D> DistributedNdarray<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + Clone + Send + Sync + 'static + crate::ndarray::RemoveAxis,
{
    /// Create a new distributed array from chunks.
    #[must_use]
    pub fn new(
        chunks: Vec<ArrayChunk<T, D>>,
        shape: Vec<usize>,
        config: DistributedConfig,
    ) -> Self {
        let uuid = uuid::Uuid::new_v4();
        let id = format!("uuid_{uuid}");
        Self {
            config,
            chunks,
            shape,
            id,
        }
    }

    /// Create a distributed array by splitting an existing array.
    ///
    /// Every chunk gets a genuinely disjoint slice of the original data,
    /// split along axis 0 and recorded with its real starting offset in
    /// `global_index`, so [`Self::to_array`] can reconstruct the original
    /// array exactly. The communication backend (`config.backend`) is still
    /// simulated (`nodeid` is a simple round-robin label, not a real network
    /// destination) — only the actual data partitioning needed for correct
    /// round-tripping is implemented here.
    #[must_use]
    pub fn from_array(array: &Array<T, D>, config: DistributedConfig) -> Self
    where
        T: Clone,
    {
        let shape = array.shape().to_vec();
        let num_chunks = config.chunks.max(1);
        let axis0_len = shape.first().copied().unwrap_or(1).max(1);
        let rows_per_chunk = axis0_len.div_ceil(num_chunks).max(1);

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut nodeid = 0usize;
        while start < axis0_len {
            let end = (start + rows_per_chunk).min(axis0_len);
            let chunk_data = array
                .slice_axis(Axis(0), Slice::from(start..end))
                .to_owned();

            chunks.push(ArrayChunk {
                data: chunk_data,
                global_index: vec![start],
                nodeid: nodeid % 3, // Simulate distribution across 3 nodes
            });

            start = end;
            nodeid += 1;
        }

        Self::new(chunks, shape, config)
    }

    /// Get the number of chunks in this distributed array.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get the shape of this distributed array.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a reference to the chunks in this distributed array.
    #[must_use]
    pub fn chunks(&self) -> &[ArrayChunk<T, D>] {
        &self.chunks
    }

    /// Convert this distributed array back to a regular array.
    ///
    /// Reconstructs the array by copying each chunk's real data back to the
    /// axis-0 offset recorded in its `global_index` (set by
    /// [`Self::from_array`]) — this only correctly inverts chunks produced by
    /// `from_array`'s own axis-0 splitting; arbitrary externally-constructed
    /// chunks (via [`Self::new`]) with different placement semantics are not
    /// supported.
    ///
    /// # Errors
    /// Returns `CoreError` if array conversion fails.
    pub fn to_array(&self) -> CoreResult<Array<T, crate::ndarray::IxDyn>> {
        // `T: Zero` is already guaranteed by this impl block's own bounds.
        let mut result =
            Array::<T, crate::ndarray::IxDyn>::zeros(crate::ndarray::IxDyn(&self.shape));

        for chunk in &self.chunks {
            let start = chunk.global_index.first().copied().unwrap_or(0);
            let chunk_view = chunk.data.view().into_dyn();
            let len = chunk_view.shape().first().copied().unwrap_or(0);
            if len == 0 {
                continue;
            }
            let mut dest = result.slice_axis_mut(Axis(0), Slice::from(start..start + len));
            dest.assign(&chunk_view);
        }

        Ok(result)
    }

    /// Execute a function on each chunk in parallel.
    #[must_use]
    pub fn map<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&ArrayChunk<T, D>) -> R + Send + Sync,
        R: Send + 'static,
    {
        // In a real distributed system, this would execute functions on different nodes
        // For now, use a simple iterator instead of parallel execution
        self.chunks.iter().map(f).collect()
    }

    /// Reduce the results of mapping a function across all chunks.
    ///
    /// # Panics
    ///
    /// Panics if the chunks collection is empty and no initial value can be reduced.
    #[must_use]
    pub fn map_reduce<F, R, G>(&self, map_fn: F, reducefn: G) -> R
    where
        F: Fn(&ArrayChunk<T, D>) -> R + Send + Sync,
        G: Fn(R, R) -> R + Send + Sync,
        R: Send + Clone + 'static,
    {
        // Map phase
        let results = self.map(map_fn);

        // Reduce phase
        // In a real distributed system, this might happen on a single node
        results
            .into_iter()
            .reduce(reducefn)
            .expect("Operation failed")
    }
}

impl<T, D> ArrayProtocol for DistributedNdarray<T, D>
where
    T: Clone
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + std::ops::Div<f64, Output = T>
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
    D: Dimension + Clone + Send + Sync + 'static + crate::ndarray::RemoveAxis,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::array_protocol::operations::sum" => {
                // Distributed implementation of sum
                let axis = kwargs.get("axis").and_then(|a| a.downcast_ref::<usize>());

                if let Some(&ax) = axis {
                    // Sum along a specific axis - use map-reduce across chunks
                    // In a simplified implementation, we'll use a dummy array
                    let dummy_array = self.chunks[0].data.clone();
                    let sum_array = dummy_array.sum_axis(crate::ndarray::Axis(ax));

                    // Create a new distributed array with the result
                    Ok(Box::new(super::NdarrayWrapper::new(sum_array)))
                } else {
                    // Sum all elements using map-reduce
                    let sum = self.map_reduce(|chunk| chunk.data.sum(), |a, b| a + b);
                    Ok(Box::new(sum))
                }
            }
            "scirs2::array_protocol::operations::mean" => {
                // Distributed implementation of mean
                // Get total sum across chunks
                let sum = self.map_reduce(|chunk| chunk.data.sum(), |a, b| a + b);

                // Calculate the total number of elements across all chunks
                #[allow(clippy::cast_precision_loss)]
                let count = self.shape.iter().product::<usize>() as f64;

                // Calculate mean
                let mean = sum / count;

                Ok(Box::new(mean))
            }
            "scirs2::array_protocol::operations::add" => {
                // Element-wise addition
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a distributed array
                if let Some(other) = args[1].downcast_ref::<Self>() {
                    // Check shapes match
                    if self.shape() != other.shape() {
                        return Err(NotImplemented);
                    }

                    // Create a new distributed array with chunks that represent addition
                    let mut new_chunks = Vec::with_capacity(self.chunks.len());

                    // For simplicity, assume number of chunks matches
                    // In a real implementation, we would handle different chunk distributions
                    for (self_chunk, other_chunk) in self.chunks.iter().zip(other.chunks.iter()) {
                        let result_data = &self_chunk.data + &other_chunk.data;
                        new_chunks.push(ArrayChunk {
                            data: result_data,
                            global_index: self_chunk.global_index.clone(),
                            nodeid: self_chunk.nodeid,
                        });
                    }

                    let result = Self::new(new_chunks, self.shape.clone(), self.config.clone());

                    return Ok(Box::new(result));
                }

                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::multiply" => {
                // Element-wise multiplication
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a distributed array
                if let Some(other) = args[1].downcast_ref::<Self>() {
                    // Check shapes match
                    if self.shape() != other.shape() {
                        return Err(NotImplemented);
                    }

                    // Create a new distributed array with chunks that represent multiplication
                    let mut new_chunks = Vec::with_capacity(self.chunks.len());

                    // For simplicity, assume number of chunks matches
                    // In a real implementation, we would handle different chunk distributions
                    for (self_chunk, other_chunk) in self.chunks.iter().zip(other.chunks.iter()) {
                        let result_data = &self_chunk.data * &other_chunk.data;
                        new_chunks.push(ArrayChunk {
                            data: result_data,
                            global_index: self_chunk.global_index.clone(),
                            nodeid: self_chunk.nodeid,
                        });
                    }

                    let result = Self::new(new_chunks, self.shape.clone(), self.config.clone());

                    return Ok(Box::new(result));
                }

                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::matmul" => {
                // Matrix multiplication
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // We can only handle matrix multiplication for 2D arrays
                if self.shape.len() != 2 {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a distributed array
                if let Some(other) = args[1].downcast_ref::<Self>() {
                    // Check that shapes are compatible
                    if self.shape.len() != 2
                        || other.shape.len() != 2
                        || self.shape[1] != other.shape[0]
                    {
                        return Err(NotImplemented);
                    }

                    // In a real implementation, we would perform a distributed matrix multiplication
                    // For this simplified version, we'll return a dummy result with the correct shape

                    let resultshape = vec![self.shape[0], other.shape[1]];

                    // Create a dummy result array
                    // Using a simpler approach with IxDyn directly
                    let dummyshape = crate::ndarray::IxDyn(&resultshape);
                    let dummy_array = Array::<T, crate::ndarray::IxDyn>::zeros(dummyshape);

                    // Create a new distributed array with the dummy result
                    let chunk = ArrayChunk {
                        data: dummy_array,
                        global_index: vec![0],
                        nodeid: 0,
                    };

                    let result =
                        DistributedNdarray::new(vec![chunk], resultshape, self.config.clone());

                    return Ok(Box::new(result));
                }

                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::transpose" => {
                // Transpose operation
                if self.shape.len() != 2 {
                    return Err(NotImplemented);
                }

                // Create a new shape for the transposed array
                let transposedshape = vec![self.shape[1], self.shape[0]];

                // In a real implementation, we would transpose each chunk and reconstruct
                // the distributed array with the correct chunk distribution
                // For this simplified version, we'll just create a single dummy chunk

                // Create a dummy result array
                // Using a simpler approach with IxDyn directly
                let dummyshape = crate::ndarray::IxDyn(&transposedshape);
                let dummy_array = Array::<T, crate::ndarray::IxDyn>::zeros(dummyshape);

                // Create a new distributed array with the dummy result
                let chunk = ArrayChunk {
                    data: dummy_array,
                    global_index: vec![0],
                    nodeid: 0,
                };

                let result =
                    DistributedNdarray::new(vec![chunk], transposedshape, self.config.clone());

                Ok(Box::new(result))
            }
            "scirs2::array_protocol::operations::reshape" => {
                // Reshape operation
                if let Some(shape) = kwargs
                    .get("shape")
                    .and_then(|s| s.downcast_ref::<Vec<usize>>())
                {
                    // Check that total size matches
                    let old_size: usize = self.shape.iter().product();
                    let new_size: usize = shape.iter().product();

                    if old_size != new_size {
                        return Err(NotImplemented);
                    }

                    // In a real implementation, we would need to redistribute the chunks
                    // For this simplified version, we'll just create a single dummy chunk

                    // Create a dummy result array
                    // Using a simpler approach with IxDyn directly
                    let dummyshape = crate::ndarray::IxDyn(shape);
                    let dummy_array = Array::<T, crate::ndarray::IxDyn>::zeros(dummyshape);

                    // Create a new distributed array with the dummy result
                    let chunk = ArrayChunk {
                        data: dummy_array,
                        global_index: vec![0],
                        nodeid: 0,
                    };

                    let result =
                        DistributedNdarray::new(vec![chunk], shape.clone(), self.config.clone());

                    return Ok(Box::new(result));
                }

                Err(NotImplemented)
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(Self {
            config: self.config.clone(),
            chunks: self.chunks.clone(),
            shape: self.shape.clone(),
            id: self.id.clone(),
        })
    }
}

impl<T, D> DistributedArray for DistributedNdarray<T, D>
where
    T: Clone
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + std::ops::Div<f64, Output = T>
        + Default
        + num_traits::One,
    D: Dimension + Clone + Send + Sync + 'static + crate::ndarray::RemoveAxis,
{
    fn distribution_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".to_string(), "distributed_ndarray".to_string());
        info.insert("chunks".to_string(), self.chunks.len().to_string());
        info.insert("shape".to_string(), format!("{:?}", self.shape));
        info.insert("id".to_string(), self.id.clone());
        info.insert(
            "strategy".to_string(),
            format!("{:?}", self.config.strategy),
        );
        info.insert("backend".to_string(), format!("{:?}", self.config.backend));
        info
    }

    /// # Errors
    /// Returns `CoreError` if gathering fails.
    fn gather(&self) -> CoreResult<Box<dyn ArrayProtocol>>
    where
        D: crate::ndarray::RemoveAxis,
        T: Default + Clone + num_traits::One,
    {
        // In a real implementation, this would gather data from all nodes
        // Get a properly shaped array with the right dimensions
        let array_dyn = self.to_array()?;

        // Wrap it in NdarrayWrapper
        Ok(Box::new(super::NdarrayWrapper::new(array_dyn)))
    }

    /// # Errors
    /// Returns `CoreError` if scattering fails.
    fn scatter(&self, chunks: usize) -> CoreResult<Box<dyn DistributedArray>> {
        // Create a new distributed array with a different number of chunks, but since
        // to_array requires complex trait bounds, we'll do a simplified version
        // that just creates a new array directly

        let mut config = self.config.clone();
        config.chunks = chunks;

        // Create a new distributed array with the specified number of chunks
        // For simplicity, we'll just create a copy of the existing chunks
        let new_dist_array = Self {
            config,
            chunks: self.chunks.clone(),
            shape: self.shape.clone(),
            id: {
                let uuid = uuid::Uuid::new_v4();
                format!("uuid_{uuid}")
            },
        };

        Ok(Box::new(new_dist_array))
    }

    fn is_distributed(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::Array2;

    #[test]
    fn test_distributed_ndarray_creation() {
        let array = Array2::<f64>::ones((10, 5));
        let config = DistributedConfig {
            chunks: 3,
            ..Default::default()
        };

        let dist_array = DistributedNdarray::from_array(&array, config);

        // Check that the array was split correctly
        assert_eq!(dist_array.num_chunks(), 3);
        assert_eq!(dist_array.shape(), &[10, 5]);

        // Chunks are genuinely disjoint row-wise slices of the original
        // array (see `from_array`), so their total element count must equal
        // the original array's size exactly — not a multiple of it.
        let total_elements: usize = dist_array
            .chunks()
            .iter()
            .map(|chunk| chunk.data.len())
            .sum();
        assert_eq!(total_elements, array.len());
    }

    #[test]
    fn test_distributed_ndarray_to_array() {
        // Non-constant data (not all-ones): a stub that fabricates a
        // plausible-looking placeholder (e.g. all-ones, matching a former
        // version of this test) would fail this check, whereas an all-ones
        // input would not have caught it.
        let array = Array2::from_shape_fn((10, 5), |(i, j)| (i * 5 + j) as f64);
        let config = DistributedConfig {
            chunks: 3,
            ..Default::default()
        };

        let dist_array = DistributedNdarray::from_array(&array, config);

        // Convert back to a regular array
        let result = dist_array.to_array().expect("Operation failed");

        // Check that the result matches the original array's shape and content.
        assert_eq!(result.shape(), array.shape());
        assert_eq!(result, array.into_dyn());
    }

    #[test]
    fn test_distributed_ndarray_map_reduce() {
        let array = Array2::<f64>::ones((10, 5));
        let config = DistributedConfig {
            chunks: 3,
            ..Default::default()
        };

        let dist_array = DistributedNdarray::from_array(&array, config);

        // Chunks are disjoint slices, so summing across all of them via
        // map_reduce should reconstruct the original array's total sum
        // exactly once — not a multiple of it.
        let expected_sum = array.sum();

        // Calculate the sum using map_reduce
        let sum = dist_array.map_reduce(|chunk| chunk.data.sum(), |a, b| a + b);

        assert_eq!(sum, expected_sum);
    }
}
