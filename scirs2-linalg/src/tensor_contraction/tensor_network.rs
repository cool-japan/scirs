//! Tensor Network Operations
//!
//! This module provides functionality for representing and manipulating tensor networks,
//! which are a way to represent high-dimensional tensors as a network of smaller tensors
//! connected by shared indices.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{ArrayD, Dimension, IxDyn};
use num_traits::{Float, NumAssign, Zero};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::iter::Sum;

/// Represents a tensor network node.
///
/// A tensor network node is a tensor with named indices, which can be connected
/// to other tensor nodes through shared indices.
#[derive(Debug, Clone)]
pub struct TensorNode<A>
where
    A: Clone + Float + Debug,
{
    /// The tensor data
    pub data: ArrayD<A>,
    /// The names of the indices, in the order they appear in the tensor
    pub indices: Vec<String>,
}

impl<A> TensorNode<A>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + Send + Sync + 'static,
{
    /// Creates a new tensor node.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor data
    /// * `indices` - The names of the indices, in the order they appear in the tensor
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node
    ///
    /// # Errors
    ///
    /// * If the number of indices does not match the number of dimensions in the tensor
    /// * If there are duplicate index names
    pub fn new(data: ArrayD<A>, indices: Vec<String>) -> LinalgResult<Self> {
        // Check that the number of indices matches the number of dimensions
        if indices.len() != data.ndim() {
            return Err(LinalgError::ShapeError(format!(
                "Number of indices ({}) does not match number of tensor dimensions ({})",
                indices.len(),
                data.ndim()
            )));
        }

        // Check for duplicate index names
        let mut unique_indices = HashSet::new();
        for index in &indices {
            if !unique_indices.insert(index) {
                return Err(LinalgError::ValueError(format!(
                    "Duplicate index name: {}",
                    index
                )));
            }
        }

        Ok(TensorNode { data, indices })
    }

    /// Gets the shape of the tensor node.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - The shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Gets the dimensionality of the tensor node.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of dimensions in the tensor
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Transposes the tensor node by reordering its indices.
    ///
    /// # Arguments
    ///
    /// * `new_order` - The new order of indices, specified by their names
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node with reordered indices
    ///
    /// # Errors
    ///
    /// * If any index name in `new_order` does not exist in the tensor node
    /// * If `new_order` does not contain all the indices of the tensor node
    pub fn transpose(&self, new_order: &[String]) -> LinalgResult<Self> {
        // Check that the number of indices in new_order matches the number of dimensions
        if new_order.len() != self.ndim() {
            return Err(LinalgError::ShapeError(format!(
                "Number of indices in new_order ({}) does not match number of tensor dimensions ({})",
                new_order.len(),
                self.ndim()
            )));
        }

        // Check that all indices in new_order are present in the tensor node
        let unique_new_indices: HashSet<_> = new_order.iter().collect();
        let current_indices: HashSet<_> = self.indices.iter().collect();
        if unique_new_indices != current_indices {
            return Err(LinalgError::ValueError(
                "New order must contain exactly the same indices as the tensor node".to_string(),
            ));
        }

        // Map from current indices to their positions
        let index_positions: HashMap<_, _> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, idx)| (idx.as_str(), i))
            .collect();

        // Create the permutation
        let mut permutation = Vec::with_capacity(self.ndim());
        for idx in new_order {
            permutation.push(index_positions[idx.as_str()]);
        }

        // Permute the data
        let permuted_data = self.data.clone().permuted_axes(permutation.as_slice());

        // Create the new tensor node
        TensorNode::new(permuted_data, new_order.to_vec())
    }

    /// Contracts this tensor node with another tensor node along shared indices.
    ///
    /// # Arguments
    ///
    /// * `other` - Another tensor node to contract with
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node resulting from the contraction
    ///
    /// # Errors
    ///
    /// * If the dimensions of shared indices do not match
    pub fn contract(&self, other: &TensorNode<A>) -> LinalgResult<TensorNode<A>> {
        // Find shared indices
        let self_indices: HashSet<_> = self.indices.iter().collect();
        let other_indices: HashSet<_> = other.indices.iter().collect();
        let shared_indices: Vec<_> = self_indices
            .intersection(&other_indices)
            .map(|&idx| idx.clone())
            .collect();

        if shared_indices.is_empty() {
            return Err(LinalgError::ValueError(
                "No shared indices found for contraction".to_string(),
            ));
        }

        // Get positions of shared indices in both tensors
        let mut self_contract_axes = Vec::new();
        let mut other_contract_axes = Vec::new();

        for idx in &shared_indices {
            let self_pos = self
                .indices
                .iter()
                .position(|x| x == idx)
                .expect("Index not found");
            let other_pos = other
                .indices
                .iter()
                .position(|x| x == idx)
                .expect("Index not found");

            // Check that dimensions match
            if self.data.shape()[self_pos] != other.data.shape()[other_pos] {
                return Err(LinalgError::ShapeError(format!(
                    "Dimension mismatch for index '{}': {} != {}",
                    idx,
                    self.data.shape()[self_pos],
                    other.data.shape()[other_pos]
                )));
            }

            self_contract_axes.push(self_pos);
            other_contract_axes.push(other_pos);
        }

        // Perform contraction using tensor_contraction function
        let result_data = crate::tensor_contraction::contract(
            &self.data.view(),
            &other.data.view(),
            &self_contract_axes,
            &other_contract_axes,
        )?;

        // Determine the indices of the result tensor
        let mut result_indices = Vec::new();

        // Add non-contracted indices from self
        for (i, idx) in self.indices.iter().enumerate() {
            if !self_contract_axes.contains(&i) {
                result_indices.push(idx.clone());
            }
        }

        // Add non-contracted indices from other
        for (i, idx) in other.indices.iter().enumerate() {
            if !other_contract_axes.contains(&i) {
                result_indices.push(idx.clone());
            }
        }

        // Create the resulting tensor node
        TensorNode::new(result_data, result_indices)
    }

    /// Creates an outer product of this tensor node with another tensor node.
    ///
    /// # Arguments
    ///
    /// * `other` - Another tensor node for outer product
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node resulting from the outer product
    ///
    /// # Errors
    ///
    /// * If there are shared indices between the tensors
    pub fn outer_product(&self, other: &TensorNode<A>) -> LinalgResult<TensorNode<A>> {
        // Check for shared indices
        let self_indices: HashSet<_> = self.indices.iter().collect();
        let other_indices: HashSet<_> = other.indices.iter().collect();
        let shared_indices: Vec<_> = self_indices
            .intersection(&other_indices)
            .map(|&idx| idx.clone())
            .collect();

        if !shared_indices.is_empty() {
            return Err(LinalgError::ValueError(format!(
                "Tensors have shared indices {:?}, which is not allowed for outer product",
                shared_indices
            )));
        }

        // Compute shapes of the result tensor
        let mut result_shape = Vec::new();
        result_shape.extend_from_slice(self.data.shape());
        result_shape.extend_from_slice(other.data.shape());

        // Create result tensor
        let mut result_data = ArrayD::zeros(IxDyn(&result_shape));

        // Compute outer product
        for self_idx in ndarray::indices(self.data.shape()) {
            for other_idx in ndarray::indices(other.data.shape()) {
                let mut result_idx = Vec::new();
                for &i in self_idx.as_array_view().iter() {
                    result_idx.push(i);
                }
                for &i in other_idx.as_array_view().iter() {
                    result_idx.push(i);
                }

                result_data[result_idx.as_slice()] = self.data[&self_idx] * other.data[&other_idx];
            }
        }

        // Combine indices
        let mut result_indices = self.indices.clone();
        result_indices.extend(other.indices.clone());

        // Create the resulting tensor node
        TensorNode::new(result_data, result_indices)
    }

    /// Traces (contracts) a tensor along a pair of its own indices.
    ///
    /// # Arguments
    ///
    /// * `index1` - The name of the first index to trace
    /// * `index2` - The name of the second index to trace
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node with the specified indices traced
    ///
    /// # Errors
    ///
    /// * If either index does not exist in the tensor
    /// * If the dimensions of the indices do not match
    pub fn trace(&self, index1: &str, index2: &str) -> LinalgResult<TensorNode<A>> {
        // Find positions of the indices
        let pos1 = match self.indices.iter().position(|x| x == index1) {
            Some(p) => p,
            None => {
                return Err(LinalgError::ValueError(format!(
                    "Index '{}' not found in tensor",
                    index1
                )))
            }
        };

        let pos2 = match self.indices.iter().position(|x| x == index2) {
            Some(p) => p,
            None => {
                return Err(LinalgError::ValueError(format!(
                    "Index '{}' not found in tensor",
                    index2
                )))
            }
        };

        // Check that dimensions match
        if self.data.shape()[pos1] != self.data.shape()[pos2] {
            return Err(LinalgError::ShapeError(format!(
                "Dimension mismatch for traced indices '{}' and '{}': {} != {}",
                index1,
                index2,
                self.data.shape()[pos1],
                self.data.shape()[pos2]
            )));
        }

        // Determine the shape of the result tensor
        let mut result_shape = Vec::new();
        let mut result_indices = Vec::new();

        for (i, idx) in self.indices.iter().enumerate() {
            if i != pos1 && i != pos2 {
                result_shape.push(self.data.shape()[i]);
                result_indices.push(idx.clone());
            }
        }

        // Create result tensor
        let mut result_data = ArrayD::zeros(IxDyn(&result_shape));

        // Perform trace operation
        // Note: This is a naive implementation for clarity; more efficient implementations exist
        let trace_dim = self.data.shape()[pos1];

        // Build indices for iteration
        let mut non_trace_axes = Vec::new();
        for i in 0..self.ndim() {
            if i != pos1 && i != pos2 {
                non_trace_axes.push(i);
            }
        }

        // Iterate over each result index
        for result_idx in ndarray::indices(result_data.shape()) {
            let mut sum = A::zero();

            // Sum over the traced dimension
            for k in 0..trace_dim {
                // Build the corresponding index in the original tensor
                let mut self_idx = vec![0; self.ndim()];

                // Fill in non-traced indices
                for (result_pos, &axis) in non_trace_axes.iter().enumerate() {
                    self_idx[axis] = result_idx[result_pos];
                }

                // Fill in traced indices
                self_idx[pos1] = k;
                self_idx[pos2] = k;

                // Add to the sum
                sum += self.data[self_idx.as_slice()];
            }

            // Store the result
            result_data[&result_idx] = sum;
        }

        // Create the resulting tensor node
        TensorNode::new(result_data, result_indices)
    }

    /// Adds a dummy index to the tensor node.
    ///
    /// # Arguments
    ///
    /// * `index_name` - The name of the new index
    /// * `position` - The position where to insert the new index
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node with the added index
    ///
    /// # Errors
    ///
    /// * If the index name already exists
    /// * If the position is out of bounds
    pub fn add_dummy_index(
        &self,
        index_name: &str,
        position: usize,
    ) -> LinalgResult<TensorNode<A>> {
        // Check that the index name doesn't already exist
        if self.indices.contains(&index_name.to_string()) {
            return Err(LinalgError::ValueError(format!(
                "Index name '{}' already exists in tensor",
                index_name
            )));
        }

        // Check that the position is valid
        if position > self.ndim() {
            return Err(LinalgError::ValueError(format!(
                "Position {} out of bounds for tensor with {} dimensions",
                position,
                self.ndim()
            )));
        }

        // Create a new shape with the dummy dimension added
        let mut new_shape = self.data.shape().to_vec();
        new_shape.insert(position, 1);

        // Reshape the data to add the dummy dimension
        let mut new_data = self.data.clone();
        new_data = new_data.into_shape_with_order(new_shape).map_err(|e| {
            LinalgError::ComputationError(format!("Failed to reshape tensor: {}", e))
        })?;

        // Create new indices list with the new index added
        let mut new_indices = self.indices.clone();
        new_indices.insert(position, index_name.to_string());

        // Create the new tensor node
        TensorNode::new(new_data, new_indices)
    }

    /// Removes an index from the tensor node.
    ///
    /// This function removes an index by summing over it, effectively reducing
    /// the dimensionality of the tensor by 1.
    ///
    /// # Arguments
    ///
    /// * `index_name` - The name of the index to remove
    ///
    /// # Returns
    ///
    /// * `TensorNode` - A new tensor node with the index removed
    ///
    /// # Errors
    ///
    /// * If the index does not exist in the tensor
    pub fn remove_index(&self, index_name: &str) -> LinalgResult<TensorNode<A>> {
        // Find the position of the index
        let position = match self.indices.iter().position(|x| x == index_name) {
            Some(p) => p,
            None => {
                return Err(LinalgError::ValueError(format!(
                    "Index '{}' not found in tensor",
                    index_name
                )))
            }
        };

        // Sum over the specified axis
        let new_data = self.data.sum_axis(ndarray::Axis(position));

        // Create new indices list with the index removed
        let mut new_indices = self.indices.clone();
        new_indices.remove(position);

        // Create the new tensor node
        TensorNode::new(new_data.into_dyn(), new_indices)
    }
}

/// Represents a tensor network, which is a collection of tensor nodes.
#[derive(Debug, Clone)]
pub struct TensorNetwork<A>
where
    A: Clone + Float + Debug,
{
    /// The tensor nodes in the network
    pub nodes: Vec<TensorNode<A>>,
}

impl<A> TensorNetwork<A>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + Send + Sync + 'static,
{
    /// Creates a new tensor network.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The tensor nodes in the network
    ///
    /// # Returns
    ///
    /// * `TensorNetwork` - A new tensor network
    pub fn new(nodes: Vec<TensorNode<A>>) -> Self {
        TensorNetwork { nodes }
    }

    /// Adds a tensor node to the network.
    ///
    /// # Arguments
    ///
    /// * `node` - The tensor node to add
    pub fn add_node(&mut self, node: TensorNode<A>) {
        self.nodes.push(node);
    }

    /// Contracts two tensor nodes in the network.
    ///
    /// # Arguments
    ///
    /// * `node1_idx` - The index of the first node
    /// * `node2_idx` - The index of the second node
    ///
    /// # Returns
    ///
    /// * `TensorNetwork` - A new tensor network with the contracted node
    ///
    /// # Errors
    ///
    /// * If either node index is out of bounds
    /// * If the nodes do not share any indices
    /// * If the contraction fails
    pub fn contract_nodes(
        &self,
        node1_idx: usize,
        node2_idx: usize,
    ) -> LinalgResult<TensorNetwork<A>> {
        // Check that indices are valid
        if node1_idx >= self.nodes.len() || node2_idx >= self.nodes.len() {
            return Err(LinalgError::ValueError(format!(
                "Node indices out of bounds: {} and/or {} >= {}",
                node1_idx,
                node2_idx,
                self.nodes.len()
            )));
        }

        if node1_idx == node2_idx {
            return Err(LinalgError::ValueError(
                "Cannot contract a node with itself".to_string(),
            ));
        }

        // Get the nodes
        let node1 = &self.nodes[node1_idx];
        let node2 = &self.nodes[node2_idx];

        // Contract the nodes
        let contracted_node = node1.contract(node2)?;

        // Create a new network with the contracted node
        let mut new_nodes = Vec::new();

        // Add all nodes except the contracted ones
        for (i, node) in self.nodes.iter().enumerate() {
            if i != node1_idx && i != node2_idx {
                new_nodes.push(node.clone());
            }
        }

        // Add the contracted node
        new_nodes.push(contracted_node);

        Ok(TensorNetwork::new(new_nodes))
    }

    /// Contracts the entire tensor network into a single tensor node.
    ///
    /// This function contracts the tensor network using a greedy algorithm
    /// that repeatedly contracts the pair of nodes with the most shared indices.
    ///
    /// # Returns
    ///
    /// * `TensorNode` - The result of contracting the entire network
    ///
    /// # Errors
    ///
    /// * If the network is empty
    /// * If the network cannot be fully contracted
    /// * If any contraction fails
    pub fn contract_all(&self) -> LinalgResult<TensorNode<A>> {
        if self.nodes.is_empty() {
            return Err(LinalgError::ValueError(
                "Cannot contract an empty tensor network".to_string(),
            ));
        }

        if self.nodes.len() == 1 {
            return Ok(self.nodes[0].clone());
        }

        // Create a working copy of the network
        let mut network = self.clone();

        // Repeatedly contract pairs of nodes until only one remains
        while network.nodes.len() > 1 {
            // Find the pair of nodes with the most shared indices
            let (node1_idx, node2_idx) = network.find_best_contraction_pair()?;

            // Contract these nodes
            network = network.contract_nodes(node1_idx, node2_idx)?;
        }

        // Return the final node
        Ok(network.nodes[0].clone())
    }

    /// Finds the best pair of nodes to contract next, based on the number of shared indices.
    ///
    /// # Returns
    ///
    /// * `(usize, usize)` - Indices of the best pair of nodes to contract
    ///
    /// # Errors
    ///
    /// * If no contractible pair is found
    fn find_best_contraction_pair(&self) -> LinalgResult<(usize, usize)> {
        let mut best_pair = None;
        let mut max_shared = 0;

        // Check all pairs of nodes
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                // Count shared indices
                let node1_indices: HashSet<_> = self.nodes[i].indices.iter().collect();
                let node2_indices: HashSet<_> = self.nodes[j].indices.iter().collect();
                let shared_indices = node1_indices.intersection(&node2_indices).count();

                // Update best pair if this one is better
                if shared_indices > 0 && shared_indices > max_shared {
                    max_shared = shared_indices;
                    best_pair = Some((i, j));
                }
            }
        }

        match best_pair {
            Some(pair) => Ok(pair),
            None => Err(LinalgError::ValueError(
                "No contractible pair of nodes found in the network".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tensor_node_creation() {
        // Create a 2x3 tensor
        let data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices = vec!["i".to_string(), "j".to_string()];

        let node = TensorNode::new(data, indices).unwrap();

        assert_eq!(node.shape(), vec![2, 3]);
        assert_eq!(node.ndim(), 2);
        assert_eq!(node.indices, vec!["i".to_string(), "j".to_string()]);
    }

    #[test]
    fn test_tensor_node_transpose() {
        // Create a 2x3 tensor
        let data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices = vec!["i".to_string(), "j".to_string()];

        let node = TensorNode::new(data, indices).unwrap();

        // Transpose the tensor
        let transposed = node.transpose(&["j".to_string(), "i".to_string()]).unwrap();

        assert_eq!(transposed.shape(), vec![3, 2]);
        assert_eq!(transposed.indices, vec!["j".to_string(), "i".to_string()]);

        // Check data
        assert_eq!(transposed.data[[0, 0]], 1.0);
        assert_eq!(transposed.data[[0, 1]], 4.0);
        assert_eq!(transposed.data[[1, 0]], 2.0);
        assert_eq!(transposed.data[[1, 1]], 5.0);
        assert_eq!(transposed.data[[2, 0]], 3.0);
        assert_eq!(transposed.data[[2, 1]], 6.0);
    }

    #[test]
    fn test_tensor_node_contraction() {
        // Create two tensors
        let data1 =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices1 = vec!["i".to_string(), "j".to_string()];
        let node1 = TensorNode::new(data1, indices1).unwrap();

        let data2 = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let indices2 = vec!["j".to_string(), "k".to_string()];
        let node2 = TensorNode::new(data2, indices2).unwrap();

        // Contract the nodes
        let result = node1.contract(&node2).unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.indices, vec!["i".to_string(), "k".to_string()]);

        // Check result data (matrix multiplication)
        assert_abs_diff_eq!(result.data[[0, 0]], 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.data[[0, 1]], 64.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.data[[1, 0]], 139.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.data[[1, 1]], 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tensor_node_outer_product() {
        // Create two tensors
        let data1 = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let indices1 = vec!["i".to_string()];
        let node1 = TensorNode::new(data1, indices1).unwrap();

        let data2 = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![3.0, 4.0, 5.0]).unwrap();
        let indices2 = vec!["j".to_string()];
        let node2 = TensorNode::new(data2, indices2).unwrap();

        // Compute outer product
        let result = node1.outer_product(&node2).unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.indices, vec!["i".to_string(), "j".to_string()]);

        // Check result data
        assert_eq!(result.data[[0, 0]], 3.0);
        assert_eq!(result.data[[0, 1]], 4.0);
        assert_eq!(result.data[[0, 2]], 5.0);
        assert_eq!(result.data[[1, 0]], 6.0);
        assert_eq!(result.data[[1, 1]], 8.0);
        assert_eq!(result.data[[1, 2]], 10.0);
    }

    #[test]
    fn test_tensor_node_trace() {
        // Create a 2x2 tensor
        let data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = vec!["i".to_string(), "j".to_string()];
        let node = TensorNode::new(data, indices).unwrap();

        // Compute trace
        let result = node.trace("i", "j").unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![] as Vec<usize>);
        assert_eq!(result.indices.len(), 0);

        // Check result data (should be trace of matrix)
        assert_abs_diff_eq!(result.data[[]], 5.0, epsilon = 1e-10); // 1.0 + 4.0 = 5.0
    }

    #[test]
    fn test_add_dummy_index() {
        // Create a 2x3 tensor
        let data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices = vec!["i".to_string(), "j".to_string()];
        let node = TensorNode::new(data, indices).unwrap();

        // Add a dummy index
        let result = node.add_dummy_index("k", 1).unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![2, 1, 3]);
        assert_eq!(
            result.indices,
            vec!["i".to_string(), "k".to_string(), "j".to_string()]
        );

        // Check result data
        assert_eq!(result.data[[0, 0, 0]], 1.0);
        assert_eq!(result.data[[0, 0, 1]], 2.0);
        assert_eq!(result.data[[0, 0, 2]], 3.0);
        assert_eq!(result.data[[1, 0, 0]], 4.0);
        assert_eq!(result.data[[1, 0, 1]], 5.0);
        assert_eq!(result.data[[1, 0, 2]], 6.0);
    }

    #[test]
    fn test_remove_index() {
        // Create a 2x3 tensor
        let data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices = vec!["i".to_string(), "j".to_string()];
        let node = TensorNode::new(data, indices).unwrap();

        // Remove an index by summing over it
        let result = node.remove_index("j").unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![2]);
        assert_eq!(result.indices, vec!["i".to_string()]);

        // Check result data (should be sum along j axis)
        assert_abs_diff_eq!(result.data[[0]], 6.0, epsilon = 1e-10); // 1.0 + 2.0 + 3.0 = 6.0
        assert_abs_diff_eq!(result.data[[1]], 15.0, epsilon = 1e-10); // 4.0 + 5.0 + 6.0 = 15.0
    }

    #[test]
    fn test_tensor_network_contraction() {
        // Create three tensors
        let data1 =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let indices1 = vec!["a".to_string(), "b".to_string()];
        let node1 = TensorNode::new(data1, indices1).unwrap();

        let data2 = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let indices2 = vec!["b".to_string(), "c".to_string()];
        let node2 = TensorNode::new(data2, indices2).unwrap();

        let data3 = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let indices3 = vec!["c".to_string(), "d".to_string()];
        let node3 = TensorNode::new(data3, indices3).unwrap();

        // Create tensor network
        let network = TensorNetwork::new(vec![node1, node2, node3]);

        // Contract the entire network
        let result = network.contract_all().unwrap();

        // Check result shape and indices
        assert_eq!(result.shape(), vec![2, 2]);
        // The indices might be in either order depending on contraction order
        assert!(
            result.indices == vec!["a".to_string(), "d".to_string()]
                || result.indices == vec!["d".to_string(), "a".to_string()],
            "Expected indices to be [a, d] or [d, a], got {:?}",
            result.indices
        );

        // The result should be equivalent to matrix multiplication: node1 @ node2 @ node3
        // But we don't check specific values here due to different possible contraction orders
    }
}
