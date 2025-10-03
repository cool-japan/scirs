#[cfg(feature = "tensor_contraction")]
mod tensor_example {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::tensor_contraction::{batch_matmul, contract, einsum, hosvd};

    pub fn run() {
        println!("Tensor Contraction Operations for Deep Learning");
        println!("===============================================\n");

        // Example 1: Basic matrix multiplication as tensor contraction
        println!("Example 1: Matrix Multiplication as Tensor Contraction");

        // Create two matrices
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        // Perform matrix multiplication using contract
        let c = contract(&a.view(), &b.view(), &[1], &[0]).unwrap();

        println!("Matrix A:");
        println!("{:?}", a);
        println!("Matrix B:");
        println!("{:?}", b);
        println!("A * B using contract:");
        println!("{:?}", c);
        println!();

        // Example 2: Batch Matrix Multiplication
        println!("Example 2: Batch Matrix Multiplication");

        // Create a batch of 2 matrices, each 2x2
        let batch_a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

        // Create another batch of 2 matrices, each 2x2
        let batch_b = array![[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]];

        // Perform batch matrix multiplication
        let batch_c = batch_matmul(&batch_a.view(), &batch_b.view(), 1).unwrap();

        println!("Batch A (shape: {:?}):", batch_a.shape());
        println!("Batch B (shape: {:?}):", batch_b.shape());
        println!("Batch A * Batch B (shape: {:?}):", batch_c.shape());
        println!("Result:");
        println!("{:?}", batch_c);
        println!();

        // Example 3: Einstein Summation Notation
        println!("Example 3: Einstein Summation Notation (einsum)");

        // Matrix multiplication using einsum
        let d = einsum("ij,jk->ik", &[&a.view().into_dyn(), &b.view().into_dyn()]).unwrap();

        println!("Matrix multiplication using einsum:");
        println!("{:?}", d);
        println!();

        // Batch matrix multiplication using einsum
        let batch_d = einsum(
            "bij,bjk->bik",
            &[&batch_a.view().into_dyn(), &batch_b.view().into_dyn()],
        )
        .unwrap();

        println!("Batch matrix multiplication using einsum:");
        println!("{:?}", batch_d);
        println!();

        // Trace of a matrix using einsum
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let trace = einsum("ii->", &[&matrix.view().into_dyn()]).unwrap();

        println!("Matrix:");
        println!("{:?}", matrix);
        println!("Trace using einsum: {}", trace[0]);
        println!();

        // Outer product using einsum
        let vec1 = array![1.0, 2.0, 3.0];
        let vec2 = array![4.0, 5.0];
        let outer = einsum(
            "i,j->ij",
            &[&vec1.view().into_dyn(), &vec2.view().into_dyn()],
        )
        .unwrap();

        println!("Vector 1: {:?}", vec1);
        println!("Vector 2: {:?}", vec2);
        println!("Outer product using einsum:");
        println!("{:?}", outer);
        println!();

        // Example 4: Tensor Decomposition with HOSVD
        println!("Example 4: Tensor Decomposition with Higher-Order SVD (HOSVD)");

        // Create a 3D tensor
        let tensor = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        println!("Original tensor (shape: {:?}):", tensor.shape());

        // Decompose tensor with full rank
        let (core, factors) = hosvd(&tensor.view(), &[2, 2, 2]).unwrap();

        println!("Core tensor (shape: {:?}):", core.shape());
        for (i, factor) in factors.iter().enumerate() {
            println!("Factor {} (shape: {:?}):", i, factor.shape());
            println!("{:?}", factor);
        }

        // Example 5: Advanced einsum operations
        println!("\nExample 5: Advanced einsum operations");

        // Create a 3D tensor
        let tensor3d = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ];

        // Create a 2D matrix
        let matrix2d = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];

        // Contract tensor3d and matrix2d: tensor3d[i,j,k] * matrix2d[k,l] -> result[i,j,l]
        let result = einsum(
            "ijk,kl->ijl",
            &[&tensor3d.view().into_dyn(), &matrix2d.view().into_dyn()],
        )
        .unwrap();

        println!("Tensor (shape: {:?}):", tensor3d.shape());
        println!("Matrix (shape: {:?}):", matrix2d.shape());
        println!(
            "Result of einsum \"ijk,kl->ijl\" (shape: {:?}):",
            result.shape()
        );
        println!("{:?}", result);
    }
}

#[allow(dead_code)]
fn main() {
    #[cfg(feature = "tensor_contraction")]
    {
        tensor_example::run();
    }

    #[cfg(not(feature = "tensor_contraction"))]
    {
        println!("This example requires tensor_contraction feature.");
        println!(
            "Run with: cargo run --example tensor_contraction_example --features tensor_contraction"
        );
    }
}
