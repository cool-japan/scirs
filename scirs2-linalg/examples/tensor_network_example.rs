//! This example demonstrates the Tensor Network functionality.
//!
//! To run this example with the tensor_contraction feature:
//! ```bash
//! cargo run --example tensor_network_example --features tensor_contraction
//! ```

#[cfg(feature = "tensor_contraction")]
#[allow(dead_code)]
fn main() -> scirs2_linalg::error::LinalgResult<()> {
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use scirs2_linalg::tensor_contraction::tensor_network::{TensorNetwork, TensorNode};

    println!("Tensor Network Example");
    println!("=====================\n");

    // --- Creating Tensor Nodes ---

    println!("1. Creating Tensor Nodes");
    println!("-----------------------");

    // Create a 2×3 matrix A
    let data_a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?;
    let indices_a = vec!["i".to_string(), "j".to_string()];
    let tensor_a = TensorNode::new(data_a, indices_a)?;

    println!(
        "Tensor A: shape {:?}, indices {:?}",
        tensor_a.shape(),
        tensor_a.indices
    );

    // Create a 3×4 matrix B
    let data_b = ArrayD::from_shape_vec(
        IxDyn(&[3, 4]),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?;
    let indices_b = vec!["j".to_string(), "k".to_string()];
    let tensor_b = TensorNode::new(data_b, indices_b)?;

    println!(
        "Tensor B: shape {:?}, indices {:?}",
        tensor_b.shape(),
        tensor_b.indices
    );

    // Create a 4×2 matrix C
    let data_c =
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?;
    let indices_c = vec!["k".to_string(), "l".to_string()];
    let tensor_c = TensorNode::new(data_c, indices_c)?;

    println!(
        "Tensor C: shape {:?}, indices {:?}\n",
        tensor_c.shape(),
        tensor_c.indices
    );

    // --- Tensor Node Operations ---

    println!("2. Tensor Node Operations");
    println!("------------------------");

    // Transpose a tensor
    println!("* Transposing Tensor A:");
    let transposed_a = tensor_a.transpose(&["j".to_string(), "i".to_string()])?;
    println!(
        "  Transposed: shape {:?}, indices {:?}",
        transposed_a.shape(),
        transposed_a.indices
    );

    // Add a dummy index
    println!("* Adding a dummy index to Tensor A:");
    let dummy_a = tensor_a.add_dummy_index("m", 1)?;
    println!(
        "  With dummy: shape {:?}, indices {:?}",
        dummy_a.shape(),
        dummy_a.indices
    );

    // Contract two tensors
    println!("* Contracting Tensor A and B:");
    let ab = tensor_a.contract(&tensor_b)?;
    println!("  Result: shape {:?}, indices {:?}", ab.shape(), ab.indices);
    println!("  This is equivalent to matrix multiplication A@B");

    // Outer product
    println!("* Outer product of small tensors:");
    let v1 = TensorNode::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0])
            .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?,
        vec!["i".to_string()],
    )?;
    let v2 = TensorNode::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 4.0, 5.0])
            .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?,
        vec!["j".to_string()],
    )?;

    let outer = v1.outer_product(&v2)?;
    println!(
        "  Result: shape {:?}, indices {:?}",
        outer.shape(),
        outer.indices
    );
    println!("  Data of outer product:");
    println!("  [");
    for i in 0..2 {
        println!(
            "    [{}, {}, {}]",
            outer.data[[i, 0]],
            outer.data[[i, 1]],
            outer.data[[i, 2]]
        );
    }
    println!("  ]\n");

    // Trace operation
    println!("* Trace operation on a matrix:");
    let matrix = TensorNode::new(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .map_err(|e| scirs2_linalg::error::LinalgError::ShapeError(e.to_string()))?,
        vec!["i".to_string(), "j".to_string()],
    )?;

    let traced = matrix.trace("i", "j")?;
    println!("  Original matrix: 3×3");
    println!("  Trace result: {}", traced.data[[]]);
    println!("  This is the sum of diagonal elements: 1 + 5 + 9 = 15\n");

    // --- Tensor Network Operations ---

    println!("3. Tensor Network Operations");
    println!("---------------------------");

    // Create a tensor network
    let network = TensorNetwork::new(vec![tensor_a.clone(), tensor_b.clone(), tensor_c.clone()]);
    println!("Created a tensor network with 3 nodes");
    println!("Nodes sharing indices:");
    println!("  A and B share index 'j'");
    println!("  B and C share index 'k'");

    // Contract specific nodes
    println!("* Contracting nodes A and B:");
    let network_ab = network.contract_nodes(0, 1)?;
    println!("  Network now has {} nodes", network_ab.nodes.len());
    println!(
        "  Contracted node has shape {:?} and indices {:?}",
        network_ab.nodes[2].shape(),
        network_ab.nodes[2].indices
    );

    // Contract the entire network
    println!("* Contracting the entire network:");
    let result = network.contract_all()?;
    println!(
        "  Final result: shape {:?}, indices {:?}",
        result.shape(),
        result.indices
    );
    println!("  This is equivalent to matrix chain multiplication A@B@C");

    // Comparing with direct matrix multiplication
    let expectedshape = vec![2, 2];
    println!("  Expected shape: {:?}", expectedshape);
    assert_eq!(result.shape(), expectedshape);

    // Print some values
    println!("  Some values from the result:");
    println!("    result[0,0] = {}", result.data[[0, 0]]);
    println!("    result[0,1] = {}", result.data[[0, 1]]);
    println!("    result[1,0] = {}", result.data[[1, 0]]);
    println!("    result[1,1] = {}", result.data[[1, 1]]);

    Ok(())
}

#[cfg(not(feature = "tensor_contraction"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'tensor_contraction' feature.");
    println!(
        "Please run with: cargo run --example tensor_network_example --features tensor_contraction"
    );
}
